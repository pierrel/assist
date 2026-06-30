"""ScheduleStore — disk persistence, cap, and the un-mocked read-modify-write race."""
import os
import threading
from datetime import datetime, timezone

import pytest

from assist.schedule.model import Cadence, Schedule
from assist.schedule.store import (CAP_PER_THREAD, ScheduleCapExceeded,
                                   ScheduleNotFound, ScheduleStore)


def _mk(root, tid):
    os.makedirs(os.path.join(root, tid), exist_ok=True)


def _sched(tid, sid, *, enabled=True, next_fire="2026-06-15T14:00:00+00:00"):
    return Schedule(id=sid, thread_id=tid, prompt="p", cadence=Cadence(hour=7, minute=0),
                    tz="America/Los_Angeles", enabled=enabled, next_fire_at=next_fire)


@pytest.fixture
def store(tmp_path):
    return ScheduleStore(str(tmp_path))


def test_add_and_read_round_trip(tmp_path, store):
    _mk(str(tmp_path), "t1")
    store.add(_sched("t1", "a"))
    got = store.for_thread("t1")
    assert len(got) == 1 and got[0].id == "a"


def test_cap_enforced(tmp_path, store):
    _mk(str(tmp_path), "t1")
    for i in range(CAP_PER_THREAD):
        store.add(_sched("t1", f"s{i}"))
    with pytest.raises(ScheduleCapExceeded):
        store.add(_sched("t1", "overflow"))


def test_survives_restart_via_disk(tmp_path):
    _mk(str(tmp_path), "t1")
    ScheduleStore(str(tmp_path)).add(_sched("t1", "a"))
    # A brand-new store instance (= a process restart) reads the same file.
    reloaded = ScheduleStore(str(tmp_path)).for_thread("t1")
    assert len(reloaded) == 1 and reloaded[0].id == "a"


def test_update_advances_next_fire(tmp_path, store):
    _mk(str(tmp_path), "t1")
    store.add(_sched("t1", "a"))
    store.update("t1", "a", lambda s: s.with_next_fire("2026-06-16T14:00:00+00:00"))
    assert store.for_thread("t1")[0].next_fire_at == "2026-06-16T14:00:00+00:00"


def test_remove(tmp_path, store):
    _mk(str(tmp_path), "t1")
    store.add(_sched("t1", "a"))
    store.remove("t1", "a")
    assert store.for_thread("t1") == []
    with pytest.raises(ScheduleNotFound):
        store.remove("t1", "missing")


def test_all_across_threads(tmp_path, store):
    _mk(str(tmp_path), "t1")
    _mk(str(tmp_path), "t2")
    store.add(_sched("t1", "a"))
    store.add(_sched("t2", "b"))
    assert {s.id for s in store.all()} == {"a", "b"}


def test_due_filters_disabled_and_future(tmp_path, store):
    _mk(str(tmp_path), "t1")
    store.add(_sched("t1", "past", next_fire="2026-06-15T14:00:00+00:00"))
    store.add(_sched("t1", "future", next_fire="2999-01-01T00:00:00+00:00"))
    store.add(_sched("t1", "off", enabled=False, next_fire="2026-06-15T14:00:00+00:00"))
    now = datetime(2026, 6, 15, 18, 0, tzinfo=timezone.utc)
    assert {s.id for s in store.due(now)} == {"past"}


def test_concurrent_updates_no_lost_update(tmp_path, store):
    """The advance/patch/delete race: many threads update DIFFERENT schedules on the
    same thread file concurrently; the lock must serialize RMW so none is lost."""
    _mk(str(tmp_path), "t1")
    # Pre-seed 5 schedules (the cap), then hammer each with a concurrent next_fire bump.
    for i in range(CAP_PER_THREAD):
        store.add(_sched("t1", f"s{i}", next_fire="2026-06-15T00:00:00+00:00"))
    barrier = threading.Barrier(CAP_PER_THREAD)

    def bump(sid):
        barrier.wait()  # maximize contention
        store.update("t1", sid, lambda s: s.with_next_fire("2026-12-31T00:00:00+00:00"))

    threads = [threading.Thread(target=bump, args=(f"s{i}",)) for i in range(CAP_PER_THREAD)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Every schedule's update must have landed — no lost write.
    assert all(s.next_fire_at == "2026-12-31T00:00:00+00:00" for s in store.for_thread("t1"))


def test_all_ignores_non_dir_files_in_root(tmp_path, store):
    # The thread root also holds non-dir files (e.g. ThreadManager's threads.db);
    # all()/due() must not choke on them.
    _mk(str(tmp_path), "t1")
    store.add(_sched("t1", "a"))
    (tmp_path / "threads.db").write_text("not a thread dir")
    assert {s.id for s in store.all()} == {"a"}
