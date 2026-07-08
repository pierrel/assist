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


def test_resume_recomputes_next_fire_no_catchup(tools):
    # A schedule paused past its fire time must NOT fire immediately on resume.
    tools.create_schedule("x", hour=8)
    sid = tools.store.for_thread("t1")[0].id
    tools.pause_schedule(sid)
    tools.store.update("t1", sid, lambda s: s.with_next_fire("2020-01-01T00:00:00+00:00"))
    tools.resume_schedule(sid)
    from datetime import datetime, timezone
    nxt = datetime.fromisoformat(tools.store.for_thread("t1")[0].next_fire_at)
    assert nxt > datetime.now(timezone.utc)   # recomputed forward, not the stale past value


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


def test_create_monthly_defaults_anchor_to_current_month(tools):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    before = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m")
    out = tools.create_schedule("rent", day_of_month=1, hour=9)
    after = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m")
    assert "Scheduled" in out and "on the 1st at 9:00 AM" in out
    saved = tools.store.for_thread("t1")[0]
    # the tool always sets an anchor for a monthly schedule; default = current month, user
    # tz (accept either side of a midnight month-boundary so the test can't flake in CI)
    assert saved.cadence.anchor_month in {before, after}


def test_create_monthly_skip_months(tools):
    out = tools.create_schedule("report", day_of_month=25, month_interval=2, hour=7)
    assert "on the 25th of every 2 months at 7:00 AM" in out
    saved = tools.store.for_thread("t1")[0]
    assert saved.cadence.month_interval == 2 and saved.cadence.anchor_month is not None


def test_create_monthly_explicit_anchor(tools):
    out = tools.create_schedule("q", day_of_month=25, month_interval=2, hour=7,
                                anchor_month="2026-03")
    assert "Scheduled" in out
    saved = tools.store.for_thread("t1")[0]
    assert saved.cadence.anchor_month == "2026-03"


def test_create_monthly_invalid_returns_corrective(tools):
    # day_of_month with weekdays is incoherent -> a corrective string, never a raise
    out = tools.create_schedule("x", day_of_month=5, weekdays=[0], hour=7)
    assert out.startswith("Couldn't schedule:")
