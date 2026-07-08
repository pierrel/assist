"""Pure cadence math — no I/O, no clock reads (``now`` is always passed in).

``next_after`` steps in UTC and matches against the LOCAL wall-clock, so DST is handled
by construction (a spring-forward-skipped local time simply never matches). The v1
cadence set (PRD D3): daily / weekly / specific-weekdays / hourly / every-N-minutes —
anything else is declined by ``validate`` rather than silently approximated.
"""
from __future__ import annotations

import calendar
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from assist.schedule.model import Cadence, Schedule

_UNSET = object()  # "field not supplied" — distinct from an explicit None (= "every")
_WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# 8 days of minute steps bounds the search: covers weekly (≤7d) + any clock/interval
# combo in the v1 set (no monthly). ~11.5k iterations worst case.
_MAX_STEPS = 8 * 24 * 60 + 60


class InvalidCadence(ValueError):
    """A cadence the v1 field set can't represent or that's internally incoherent.
    The message is a corrective the agent can relay/act on."""


def validate(cad: Cadence) -> None:
    """Raise InvalidCadence on an out-of-range or incoherent cadence."""
    if not (0 <= cad.minute <= 59):
        raise InvalidCadence(f"minute must be 0-59, got {cad.minute}")
    if cad.hour is not None and not (0 <= cad.hour <= 23):
        raise InvalidCadence(f"hour must be 0-23 or omitted, got {cad.hour}")
    if cad.weekdays is not None:
        if not cad.weekdays:
            raise InvalidCadence("weekdays cannot be an empty set; omit it for every day")
        for d in cad.weekdays:
            if not (0 <= d <= 6):
                raise InvalidCadence(f"weekday must be 0(Mon)-6(Sun), got {d}")
    if cad.every_n_minutes is not None:
        if cad.hour is not None:
            raise InvalidCadence(
                "every_n_minutes is an interval; don't combine it with a specific hour")
        if not (1 <= cad.every_n_minutes <= 1440):
            raise InvalidCadence(
                f"every_n_minutes must be 1-1440, got {cad.every_n_minutes}")
    if cad.day_of_month is not None:
        if not (1 <= cad.day_of_month <= 31):
            raise InvalidCadence(f"day_of_month must be 1-31, got {cad.day_of_month}")
        if cad.weekdays is not None or cad.every_n_minutes is not None:
            raise InvalidCadence(
                "day_of_month is a monthly schedule; don't combine it with "
                "weekdays or every_n_minutes")
        if cad.month_interval < 1:
            raise InvalidCadence(
                f"month_interval must be >= 1, got {cad.month_interval}")
        if cad.anchor_month is not None:
            _anchor_index(cad.anchor_month)  # raises InvalidCadence on a bad "YYYY-MM"
    elif cad.month_interval != 1:
        raise InvalidCadence(
            "month_interval only applies to a day_of_month (monthly) schedule")


def _matches(cad: Cadence, local: datetime) -> bool:
    if cad.weekdays is not None and local.weekday() not in cad.weekdays:
        return False
    if cad.every_n_minutes is not None:
        return (local.hour * 60 + local.minute) % cad.every_n_minutes == 0
    if local.minute != cad.minute:
        return False
    if cad.hour is not None and local.hour != cad.hour:
        return False
    return True


def next_after(sched: Schedule, now_utc: datetime) -> datetime:
    """The next UTC instant strictly after ``now_utc`` that matches the cadence."""
    try:
        tz = ZoneInfo(sched.tz)
    except ZoneInfoNotFoundError as e:
        raise InvalidCadence(f"unknown timezone {sched.tz!r}") from e
    now = now_utc.astimezone(timezone.utc).replace(second=0, microsecond=0)
    if sched.cadence.day_of_month is not None:
        return _next_monthly(sched.cadence, tz, now)
    cand = now + timedelta(minutes=1)
    for _ in range(_MAX_STEPS):
        if _matches(sched.cadence, cand.astimezone(tz)):
            return cand
        cand += timedelta(minutes=1)
    raise InvalidCadence("no cadence match within 8 days (unsupported pattern?)")


def _month_index(year: int, month: int) -> int:
    """A monotonic month counter (…, 2026-01 -> 24312, 2026-02 -> 24313, …), so month
    arithmetic and the every-N-months phase are plain integer math."""
    return year * 12 + (month - 1)


def _anchor_index(anchor_month: str) -> int:
    """Parse an ISO ``"YYYY-MM"`` anchor into a month index. Raises InvalidCadence on a
    malformed value (the corrective the agent relays)."""
    try:
        y, m = anchor_month.split("-")
        year, mon = int(y), int(m)
        if not (1 <= mon <= 12):
            raise ValueError
    except (ValueError, AttributeError):
        raise InvalidCadence(
            f"anchor_month must be ISO \"YYYY-MM\", got {anchor_month!r}") from None
    return _month_index(year, mon)


def _next_monthly(cad: Cadence, tz: ZoneInfo, now_utc: datetime) -> datetime:
    """The next fire for a monthly (day_of_month) cadence, computed directly (no minute
    search — a monthly date is well past the 8-day search bound). Walks qualifying months
    from max(now, anchor), clamping day_of_month to each month's length, and returns the
    first whose fire instant is strictly after ``now``."""
    now_local = now_utc.astimezone(tz)
    now_idx = _month_index(now_local.year, now_local.month)
    anchor_idx = _anchor_index(cad.anchor_month) if cad.anchor_month else now_idx
    hour = cad.hour or 0
    # Never fire before the anchor; align the start month up to the anchor's phase.
    start = max(now_idx, anchor_idx)
    while (start - anchor_idx) % cad.month_interval != 0:
        start += 1
    # At most two qualifying months to check (this month's day may already be past).
    for k in range(3):
        idx = start + k * cad.month_interval
        year, mon = divmod(idx, 12)
        mon += 1
        day = min(cad.day_of_month, calendar.monthrange(year, mon)[1])
        fire_utc = datetime(year, mon, day, hour, cad.minute, tzinfo=tz) \
            .astimezone(timezone.utc).replace(second=0, microsecond=0)
        if fire_utc > now_utc:
            return fire_utc
    raise InvalidCadence("could not compute next monthly fire")


def apply_patch(sched: Schedule, *, minute=_UNSET, hour=_UNSET, weekdays=_UNSET,
                every_n_minutes=_UNSET, day_of_month=_UNSET, month_interval=_UNSET,
                anchor_month=_UNSET) -> Schedule:
    """Return ``sched`` with ONLY the supplied cadence fields changed (relative edit).
    Unsupplied fields keep their current value; an explicit ``None`` for hour/weekdays
    means "every hour"/"every day". Validates the result."""
    c = sched.cadence
    new_cad = Cadence(
        minute=c.minute if minute is _UNSET else minute,
        hour=c.hour if hour is _UNSET else hour,
        weekdays=c.weekdays if weekdays is _UNSET else (
            tuple(weekdays) if weekdays is not None else None),
        every_n_minutes=c.every_n_minutes if every_n_minutes is _UNSET else every_n_minutes,
        day_of_month=c.day_of_month if day_of_month is _UNSET else day_of_month,
        month_interval=c.month_interval if month_interval is _UNSET else month_interval,
        anchor_month=c.anchor_month if anchor_month is _UNSET else anchor_month,
    )
    validate(new_cad)
    from dataclasses import replace
    return replace(sched, cadence=new_cad)


def describe(cad: Cadence) -> str:
    """A human-readable label, derived from the structured fields (the web view + the
    agent's confirm-back use this — never a model-supplied string)."""
    if cad.day_of_month is not None:
        day = _ordinal(cad.day_of_month)
        every = "" if cad.month_interval == 1 else f" of every {cad.month_interval} months"
        return f"on the {day}{every} at {_clock(cad.hour, cad.minute)}"
    if cad.every_n_minutes is not None:
        base = (f"every {cad.every_n_minutes} minutes" if cad.every_n_minutes != 60
                else "hourly")
        return f"{base}{_days_suffix(cad.weekdays)}"
    at = f"{_clock(cad.hour, cad.minute)}"
    if cad.hour is None:
        return f"hourly at :{cad.minute:02d}{_days_suffix(cad.weekdays)}"
    if cad.weekdays is None:
        return f"every day at {at}"
    return f"{_days_label(cad.weekdays)} at {at}"


def fmt_instant(iso_utc: str | None, tz: str, *, empty: str = "—") -> str:
    """Format a stored UTC instant as a local wall-clock ("Mon Jun 15, 7:00 AM").
    Shared by the agent's confirm-back and the web view so they never diverge."""
    if not iso_utc:
        return empty
    local = datetime.fromisoformat(iso_utc).astimezone(ZoneInfo(tz))
    h12 = local.hour % 12 or 12
    return f"{local:%a %b %d}, {h12}:{local.minute:02d} {local:%p}"


def _ordinal(n: int) -> str:
    """1 -> "1st", 2 -> "2nd", 11 -> "11th", 22 -> "22nd", 31 -> "31st"."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _clock(hour: int | None, minute: int) -> str:
    h = 0 if hour is None else hour
    h12 = h % 12 or 12
    ampm = "AM" if h < 12 else "PM"
    return f"{h12}:{minute:02d} {ampm}"


def _days_label(weekdays: tuple[int, ...]) -> str:
    wd = sorted(weekdays)
    if wd == [0, 1, 2, 3, 4]:
        return "Mon–Fri"
    if wd == [5, 6]:
        return "weekends"
    return ", ".join(_WEEKDAY_NAMES[d] for d in wd)


def _days_suffix(weekdays: tuple[int, ...] | None) -> str:
    return "" if weekdays is None else f" on {_days_label(weekdays)}"
