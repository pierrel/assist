"""Cadence engine — pure date/DST math + the relative-edit patch. No LLM."""
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from assist.schedule import cadence
from assist.schedule.cadence import InvalidCadence
from assist.schedule.model import Cadence, Schedule

LA = ZoneInfo("America/Los_Angeles")


def _sched(cad: Cadence, tz="America/Los_Angeles") -> Schedule:
    return Schedule(id="s1", thread_id="t1", prompt="hi", cadence=cad, tz=tz)


def _next_local(cad: Cadence, now_local: datetime, tz="America/Los_Angeles"):
    now_utc = now_local.replace(tzinfo=ZoneInfo(tz))
    return cadence.next_after(_sched(cad, tz), now_utc).astimezone(ZoneInfo(tz))


def test_daily_same_day_then_next_day():
    cad = Cadence(hour=7, minute=0)
    assert _next_local(cad, datetime(2026, 6, 15, 6, 0)) == datetime(2026, 6, 15, 7, 0, tzinfo=LA)
    assert _next_local(cad, datetime(2026, 6, 15, 8, 0)) == datetime(2026, 6, 16, 7, 0, tzinfo=LA)


def test_strictly_after_at_exact_match():
    # At exactly 7:00 the next fire is tomorrow, never "now".
    cad = Cadence(hour=7, minute=0)
    assert _next_local(cad, datetime(2026, 6, 15, 7, 0)) == datetime(2026, 6, 16, 7, 0, tzinfo=LA)


def test_weekly_single_weekday():
    cad = Cadence(hour=9, minute=0, weekdays=(0,))  # Mondays
    # 2026-06-16 is a Tuesday -> next Monday 2026-06-22.
    nxt = _next_local(cad, datetime(2026, 6, 16, 10, 0))
    assert nxt == datetime(2026, 6, 22, 9, 0, tzinfo=LA) and nxt.weekday() == 0


def test_weekdays_skip_weekend():
    cad = Cadence(hour=7, minute=0, weekdays=(0, 1, 2, 3, 4))
    # 2026-06-13 is a Saturday -> next weekday Monday 2026-06-15.
    assert _next_local(cad, datetime(2026, 6, 13, 8, 0)) == datetime(2026, 6, 15, 7, 0, tzinfo=LA)


def test_hourly_at_minute():
    cad = Cadence(hour=None, minute=30)
    assert _next_local(cad, datetime(2026, 6, 15, 10, 15)) == datetime(2026, 6, 15, 10, 30, tzinfo=LA)
    assert _next_local(cad, datetime(2026, 6, 15, 10, 45)) == datetime(2026, 6, 15, 11, 30, tzinfo=LA)


def test_interval_every_30_clock_aligned():
    cad = Cadence(every_n_minutes=30)
    assert _next_local(cad, datetime(2026, 6, 15, 10, 10)) == datetime(2026, 6, 15, 10, 30, tzinfo=LA)
    assert _next_local(cad, datetime(2026, 6, 15, 10, 40)) == datetime(2026, 6, 15, 11, 0, tzinfo=LA)


def test_dst_spring_forward_skips_nonexistent_local_time():
    # 2026-03-08: LA jumps 02:00 -> 03:00, so 02:30 never occurs that day.
    cad = Cadence(hour=2, minute=30)
    nxt = _next_local(cad, datetime(2026, 3, 8, 1, 0))
    assert nxt == datetime(2026, 3, 9, 2, 30, tzinfo=LA)  # skipped to the next valid day


def test_next_after_is_always_in_the_future():
    cad = Cadence(hour=7, minute=0)
    now = datetime(2020, 1, 1, 12, 0, tzinfo=LA)  # next_fire_at "in the past" -> recompute forward
    assert cadence.next_after(_sched(cad), now) > now


def test_apply_patch_changes_only_the_delta():
    base = _sched(Cadence(hour=7, minute=15, weekdays=(0, 1, 2, 3, 4)))
    patched = cadence.apply_patch(base, hour=5)
    assert patched.cadence.hour == 5
    assert patched.cadence.minute == 15                      # unchanged
    assert patched.cadence.weekdays == (0, 1, 2, 3, 4)       # unchanged


def test_apply_patch_explicit_none_means_every():
    base = _sched(Cadence(hour=7, minute=0, weekdays=(0,)))
    assert cadence.apply_patch(base, weekdays=None).cadence.weekdays is None


def test_describe_labels():
    assert cadence.describe(Cadence(hour=7, minute=0)) == "every day at 7:00 AM"
    assert cadence.describe(Cadence(hour=7, minute=0, weekdays=(0, 1, 2, 3, 4))) == "Mon–Fri at 7:00 AM"
    assert cadence.describe(Cadence(every_n_minutes=30)) == "every 30 minutes"
    assert cadence.describe(Cadence(hour=None, minute=30)) == "hourly at :30"


@pytest.mark.parametrize("cad", [
    Cadence(every_n_minutes=30, hour=7),   # interval + clock = incoherent
    Cadence(minute=60),                    # out of range
    Cadence(hour=25, minute=0),            # out of range
    Cadence(weekdays=()),                  # empty set
    Cadence(every_n_minutes=0),            # out of range
])
def test_validate_rejects_incoherent(cad):
    with pytest.raises(InvalidCadence):
        cadence.validate(cad)


def test_model_round_trip():
    s = Schedule(id="x", thread_id="t", prompt="p",
                 cadence=Cadence(hour=7, minute=5, weekdays=(0, 2)), tz="America/Los_Angeles",
                 next_fire_at="2026-06-15T14:00:00+00:00", created_at="2026-06-15T00:00:00+00:00")
    assert Schedule.from_dict(s.to_dict()) == s


# --- monthly (day_of_month + month_interval + anchor_month) ---

def _monthly(day, hour=7, minute=0, month_interval=1, anchor=None):
    # anchor=None -> the engine uses the "now" month as the anchor (every-month / phase-now);
    # skip-month + future-anchor tests pass an explicit anchor.
    return Cadence(minute=minute, hour=hour, day_of_month=day,
                   month_interval=month_interval, anchor_month=anchor)


def test_monthly_every_month_this_month_then_next():
    cad = _monthly(25, month_interval=1)
    assert _next_local(cad, datetime(2026, 7, 1, 9, 0)) == datetime(2026, 7, 25, 7, 0, tzinfo=LA)
    assert _next_local(cad, datetime(2026, 7, 26, 9, 0)) == datetime(2026, 8, 25, 7, 0, tzinfo=LA)


def test_monthly_skip_months_anchored():
    cad = _monthly(25, month_interval=2, anchor="2026-07")  # Jul, Sep, Nov, Jan…
    assert _next_local(cad, datetime(2026, 7, 1, 9, 0)) == datetime(2026, 7, 25, 7, 0, tzinfo=LA)
    assert _next_local(cad, datetime(2026, 7, 26, 9, 0)) == datetime(2026, 9, 25, 7, 0, tzinfo=LA)
    # crosses the year boundary
    assert _next_local(cad, datetime(2026, 11, 30, 9, 0)) == datetime(2027, 1, 25, 7, 0, tzinfo=LA)


def test_monthly_future_anchor_never_fires_before_it():
    cad = _monthly(25, month_interval=2, anchor="2026-09")
    assert _next_local(cad, datetime(2026, 7, 1, 9, 0)) == datetime(2026, 9, 25, 7, 0, tzinfo=LA)


@pytest.mark.parametrize("now,expect", [
    (datetime(2025, 2, 1, 9, 0), datetime(2025, 2, 28, 8, 0)),   # non-leap Feb
    (datetime(2024, 2, 1, 9, 0), datetime(2024, 2, 29, 8, 0)),   # leap Feb
    (datetime(2026, 4, 1, 9, 0), datetime(2026, 4, 30, 8, 0)),   # 30-day month
])
def test_monthly_31st_clamps_to_last_day(now, expect):
    cad = _monthly(31, hour=8, month_interval=1)
    assert _next_local(cad, now) == expect.replace(tzinfo=LA)


def test_monthly_clamp_and_skip_together():
    cad = _monthly(31, hour=8, month_interval=2, anchor="2026-01")  # Jan, Mar, May…
    assert _next_local(cad, datetime(2026, 1, 1, 9, 0)) == datetime(2026, 1, 31, 8, 0, tzinfo=LA)
    assert _next_local(cad, datetime(2026, 2, 1, 9, 0)) == datetime(2026, 3, 31, 8, 0, tzinfo=LA)


def test_describe_monthly():
    assert cadence.describe(_monthly(1, month_interval=1)) == "on the 1st at 7:00 AM"
    assert cadence.describe(_monthly(25, month_interval=2)) == "on the 25th of every 2 months at 7:00 AM"
    assert cadence.describe(_monthly(22, month_interval=1)) == "on the 22nd at 7:00 AM"


@pytest.mark.parametrize("n,s", [(1, "1st"), (2, "2nd"), (3, "3rd"), (11, "11th"),
                                 (21, "21st"), (22, "22nd"), (31, "31st")])
def test_ordinal(n, s):
    assert cadence._ordinal(n) == s


@pytest.mark.parametrize("cad", [
    Cadence(day_of_month=0), Cadence(day_of_month=32),
    Cadence(day_of_month=5, weekdays=(0,)),
    Cadence(day_of_month=5, every_n_minutes=30),
    Cadence(day_of_month=5, month_interval=0),
    Cadence(day_of_month=5, anchor_month="not-a-month"),
    Cadence(month_interval=2),  # month_interval without day_of_month
])
def test_monthly_validate_rejects(cad):
    with pytest.raises(InvalidCadence):
        cadence.validate(cad)


def test_apply_patch_monthly_fields():
    s = _sched(_monthly(25, month_interval=1))
    # "make it every 3 months" -> month_interval only
    patched = cadence.apply_patch(s, month_interval=3)
    assert patched.cadence.month_interval == 3 and patched.cadence.day_of_month == 25
    # "change it to the 1st"
    patched = cadence.apply_patch(s, day_of_month=1)
    assert patched.cadence.day_of_month == 1
