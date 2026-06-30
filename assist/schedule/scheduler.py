"""The in-process scheduler — a producer (cron poll loop) feeding a consumer (submit).

In-process is REQUIRED, not just convenient: THREAD_QUEUE is a per-process singleton, so
only an in-process producer shares the single turn slot (→ "overlap waits"), and "don't
fire while the service/LLM is down" falls out for free. The poll loop runs on a daemon
thread; dispatch runs on a single-worker executor; nothing here touches the asyncio loop.

``submit(schedule)`` is the one consumer entry — it owns the health gate, the per-thread
in-flight dedup, and the executor. A future event-trigger becomes a second producer that
computes a wakeup and calls the same dispatch path, with no change here.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from assist.schedule import cadence

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Scheduler:
    def __init__(self, store, dispatch, health_check, *, tick_seconds: float = 30.0,
                 now_fn=_utcnow):
        self._store = store
        self._dispatch = dispatch          # (tid, prompt, tz) -> None; blocking, runs on executor
        self._health_check = health_check  # () -> bool; LLM reachable?
        self._tick = tick_seconds
        self._now = now_fn
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="schedule-run")
        self._inflight: set[str] = set()   # tids with a wakeup queued/running (dedup)
        self._inflight_lock = threading.Lock()

    # --- lifecycle (started/stopped by the web lifespan) ------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="schedule-poll")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._executor.shutdown(wait=False)

    def _run(self) -> None:
        # Reconcile then poll, both on THIS thread (never the async lifespan body): the
        # disk scan would block the event loop, and same-thread sequencing removes any
        # reconcile/tick race.
        try:
            self.reconcile()
        except Exception:
            log.exception("schedule reconcile failed")
        while not self._stop.wait(self._tick):
            try:
                self.poll()
            except Exception:
                log.exception("schedule poll failed")

    # --- producer: cron --------------------------------------------------------
    def reconcile(self) -> None:
        """On startup, advance any missing/past next_fire_at to the next FUTURE instant
        WITHOUT firing — the 'no catch-up' guarantee (a service down across N windows
        resumes at the next future window, firing none of the missed N)."""
        now = self._now()
        for s in self._store.all():
            if not s.enabled:
                continue
            if not s.next_fire_at or datetime.fromisoformat(s.next_fire_at) <= now:
                self._advance(s, now)

    def poll(self) -> None:
        now = self._now()
        for s in self._store.due(now):
            self._fire(s, now)

    def _fire(self, s, now) -> None:
        # advance-persist-THEN-dispatch: persisting next_fire_at first means a crash
        # before dispatch loses the fire (consistent with no-catch-up) but never
        # double-fires; the only double-fire path is dispatch-then-fail-to-persist.
        try:
            self._advance(s, now)
        except Exception:
            log.exception("advance failed for %s/%s; skipping dispatch", s.thread_id, s.id)
            return
        if not self._claim(s.thread_id):
            log.info("schedule %s/%s due but thread busy; skipping (already in-flight)",
                     s.thread_id, s.id)
            return
        if not self._health_check():
            log.info("schedule %s/%s due but LLM unreachable; skipping (no catch-up)",
                     s.thread_id, s.id)
            self._release(s.thread_id)
            return
        self._executor.submit(self._run_wakeup, s.thread_id, s.prompt, s.tz)

    # --- consumer: the wakeup --------------------------------------------------
    def _run_wakeup(self, tid: str, prompt: str, tz: str) -> None:
        try:
            self._dispatch(tid, prompt, tz)
        except Exception:
            log.warning("scheduled run failed for %s (fail-silently)", tid, exc_info=True)
        finally:
            self._release(tid)

    def _advance(self, s, now) -> None:
        nxt = cadence.next_after(s, now).isoformat()
        self._store.update(s.thread_id, s.id, lambda x: x.with_next_fire(nxt))

    def _claim(self, tid: str) -> bool:
        with self._inflight_lock:
            if tid in self._inflight:
                return False
            self._inflight.add(tid)
            return True

    def _release(self, tid: str) -> None:
        with self._inflight_lock:
            self._inflight.discard(tid)
