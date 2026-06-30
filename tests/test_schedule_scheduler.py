"""Scheduler decision logic with a real store + fake dispatch/health/clock. No LLM."""
import os
import threading
from datetime import datetime, timezone

import pytest

from assist.schedule.model import Cadence, Schedule
from assist.schedule.scheduler import Scheduler
from assist.schedule.store import ScheduleStore

NOW = datetime(2026, 6, 15, 18, 0, tzinfo=timezone.utc)   # a Monday, 11:00 LA
PAST = "2026-06-15T00:00:00+00:00"
FUTURE = "2999-01-01T00:00:00+00:00"


class _Dispatch:
    def __init__(self):
        self.calls = []
        self._lock = threading.Lock()

    def __call__(self, tid, prompt, tz):
        with self._lock:
            self.calls.append((tid, prompt))


def _store(tmp_path, *scheds):
    s = ScheduleStore(str(tmp_path))
    for sc in scheds:
        os.makedirs(os.path.join(str(tmp_path), sc.thread_id), exist_ok=True)
        s.add(sc)
    return s


def _sched(tid="t1", sid="a", *, enabled=True, next_fire=PAST):
    return Schedule(id=sid, thread_id=tid, prompt="do it", cadence=Cadence(hour=7, minute=0),
                    tz="America/Los_Angeles", enabled=enabled, next_fire_at=next_fire)


def _sched_run(store, *, health=True):
    d = _Dispatch()
    sch = Scheduler(store, d, lambda: health, now_fn=lambda: NOW)
    return d, sch


def _flush(sch):
    sch._executor.shutdown(wait=True)


def test_due_fires_once_and_advances(tmp_path):
    store = _store(tmp_path, _sched())
    d, sch = _sched_run(store)
    sch.poll()
    _flush(sch)
    assert d.calls == [("t1", "do it")]
    nxt = store.for_thread("t1")[0].next_fire_at
    assert datetime.fromisoformat(nxt) > NOW            # advanced to the future


def test_disabled_never_fires(tmp_path):
    store = _store(tmp_path, _sched(enabled=False))
    d, sch = _sched_run(store)
    sch.poll()
    _flush(sch)
    assert d.calls == []


def test_health_down_skips_silently_but_advances(tmp_path):
    store = _store(tmp_path, _sched())
    d, sch = _sched_run(store, health=False)
    sch.poll()
    _flush(sch)
    assert d.calls == []                                 # did not fire
    assert datetime.fromisoformat(store.for_thread("t1")[0].next_fire_at) > NOW  # no catch-up


def test_dedup_skips_when_thread_inflight(tmp_path):
    store = _store(tmp_path, _sched())
    d, sch = _sched_run(store)
    sch._claim("t1")                                     # pretend a wakeup is already running
    sch._fire(store.for_thread("t1")[0], NOW)
    _flush(sch)
    assert d.calls == []                                 # skipped (deduped)
    assert datetime.fromisoformat(store.for_thread("t1")[0].next_fire_at) > NOW  # still advanced


def test_reconcile_advances_past_missed_without_firing(tmp_path):
    store = _store(tmp_path, _sched(next_fire=PAST))
    d, sch = _sched_run(store)
    sch.reconcile()
    _flush(sch)
    assert d.calls == []                                 # missed windows are NOT replayed
    assert datetime.fromisoformat(store.for_thread("t1")[0].next_fire_at) > NOW


def test_persist_failure_blocks_dispatch(tmp_path):
    store = _store(tmp_path, _sched())
    d, sch = _sched_run(store)
    def boom(*a, **k):
        raise OSError("disk full")
    store.update = boom                                  # advance can't persist
    sch._fire(store.for_thread("t1")[0], NOW)
    _flush(sch)
    assert d.calls == []                                 # no dispatch when next_fire can't persist


def test_future_schedule_not_due(tmp_path):
    store = _store(tmp_path, _sched(next_fire=FUTURE))
    d, sch = _sched_run(store)
    sch.poll()
    _flush(sch)
    assert d.calls == []
