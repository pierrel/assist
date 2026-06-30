"""Schedule tools — thread-scoped, server-side sparse-delta modify. Faked run config."""
import os
from types import SimpleNamespace

import pytest

from assist.context_rider import CONTEXT_RIDER_KEY
from assist.schedule import tools as tools_mod
from assist.schedule.store import ScheduleStore


@pytest.fixture
def tools(tmp_path, monkeypatch):
    os.makedirs(os.path.join(str(tmp_path), "t1"), exist_ok=True)
    store = ScheduleStore(str(tmp_path))
    cfg = {"configurable": {"thread_id": "t1",
                            CONTEXT_RIDER_KEY: SimpleNamespace(tz="America/Los_Angeles")}}
    monkeypatch.setattr(tools_mod, "get_config", lambda: cfg)
    fns = {f.__name__: f for f in tools_mod.schedule_tools(store)}
    return SimpleNamespace(store=store, **fns)


def test_create_then_list(tools):
    out = tools.create_schedule("morning review", hour=7, minute=0)
    assert "Scheduled" in out and "every day at 7:00 AM" in out
    assert "morning review" in tools.list_schedules()


def test_modify_is_sparse_delta(tools):
    tools.create_schedule("review", hour=7, minute=15, weekdays=[0, 1, 2, 3, 4])
    sid = tools.store.for_thread("t1")[0].id
    out = tools.modify_schedule(sid, hour=5)
    assert "Updated" in out
    cad = tools.store.for_thread("t1")[0].cadence
    assert cad.hour == 5 and cad.minute == 15 and cad.weekdays == (0, 1, 2, 3, 4)


def test_modify_unknown_id(tools):
    assert "No schedule" in tools.modify_schedule("deadbeef", hour=5)


def test_pause_resume(tools):
    tools.create_schedule("x", hour=8)
    sid = tools.store.for_thread("t1")[0].id
    assert "Paused" in tools.pause_schedule(sid)
    assert tools.store.for_thread("t1")[0].enabled is False
    assert "Resumed" in tools.resume_schedule(sid)
    assert tools.store.for_thread("t1")[0].enabled is True


def test_delete(tools):
    tools.create_schedule("x", hour=8)
    sid = tools.store.for_thread("t1")[0].id
    assert "Deleted" in tools.delete_schedule(sid)
    assert tools.store.for_thread("t1") == []


def test_cap_message(tools):
    for i in range(5):
        tools.create_schedule(f"s{i}", hour=i + 1)
    out = tools.create_schedule("overflow", hour=9)
    assert "already has 5 schedules" in out


def test_invalid_cadence_declined(tools):
    out = tools.create_schedule("bad", hour=7, every_n_minutes=30)   # interval + clock
    assert "Couldn't schedule" in out


def test_no_timezone_declines(tools, monkeypatch):
    monkeypatch.setattr(tools_mod, "get_config",
                        lambda: {"configurable": {"thread_id": "t1"}})   # no rider/tz
    assert "timezone" in tools.create_schedule("x", hour=7)
