"""The in-process scheduler — a cron poll loop feeding a dispatch consumer.

In-process is REQUIRED, not just convenient: THREAD_QUEUE is a per-process singleton, so
only an in-process producer shares the single turn slot (→ "overlap waits"), and "don't
fire while the service/LLM is down" falls out for free. The poll loop runs on a daemon
thread; dispatch runs on a single-worker executor; nothing here touches the asyncio loop.

``_fire`` is the consumer tail — it advances the next fire, applies a per-SCHEDULE
in-flight dedup + the health gate, and hands the wakeup to the executor. A future
event-trigger would split this into a shared ``submit(wakeup)`` and call the same
dispatch path; that refactor is deferred until the first such producer exists.
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
        self._inflight: set[str] = set()   # schedule ids with a wakeup queued/running (dedup)
        self._inflight_lock = threading.Lock()

    # --- lifecycle (started/stopped by the web lifespan) ------------------------
    def start(self) -> None:
        # Idempotent + restartable: a second start while running is a no-op; a start
        # after stop() clears the stop flag and rebuilds the (shut-down) executor, so a
        # re-run lifespan (common in tests) doesn't leave a no-op or extra poll threads.
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="schedule-run")
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
        due = self._store.due(now)
        if not due:
            return
        healthy = self._health_check()   # probe once per tick, only when something's due
        for s in due:
            self._fire(s, now, healthy)

    def _fire(self, s, now, healthy: bool) -> None:
        # advance-persist-THEN-dispatch: persisting next_fire_at first means a crash
        # before dispatch loses the fire (consistent with no-catch-up) but never
        # double-fires; the only double-fire path is dispatch-then-fail-to-persist.
        try:
            self._advance(s, now)
        except Exception:
            log.exception("advance failed for %s/%s; skipping dispatch", s.thread_id, s.id)
            return
        # Dedup by SCHEDULE id (not thread): this bounds a single schedule's own backlog
        # (a slow run vs a short cadence) while letting two distinct schedules on the same
        # thread both fire — THREAD_QUEUE serializes them harmlessly.
        if not self._claim(s.id):
            log.info("schedule %s/%s due but already in-flight; skipping", s.thread_id, s.id)
            return
        if not healthy:
            log.info("schedule %s/%s due but LLM unreachable; skipping (no catch-up)",
                     s.thread_id, s.id)
            self._release(s.id)
            return
        self._executor.submit(self._run_wakeup, s.id, s.thread_id, s.prompt, s.tz)

    # --- consumer: the wakeup --------------------------------------------------
    def _run_wakeup(self, sid: str, tid: str, prompt: str, tz: str) -> None:
        try:
            self._dispatch(tid, prompt, tz)
        except Exception:
            log.warning("scheduled run failed for %s/%s (fail-silently)", tid, sid, exc_info=True)
        finally:
            self._release(sid)

    def _advance(self, s, now) -> None:
        # Recompute next_fire from the CURRENT record under the store lock (not the
        # possibly-stale due() snapshot ``s``), so a concurrent modify of the cadence/tz
        # isn't clobbered with a next_fire computed from the old cadence.
        self._store.update(
            s.thread_id, s.id,
            lambda x: x.with_next_fire(cadence.next_after(x, now).isoformat()))

    def _claim(self, sid: str) -> bool:
        with self._inflight_lock:
            if sid in self._inflight:
                return False
            self._inflight.add(sid)
            return True

    def _release(self, sid: str) -> None:
        with self._inflight_lock:
            self._inflight.discard(sid)
