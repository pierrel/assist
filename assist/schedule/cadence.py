"""Pure cadence math — no I/O, no clock reads (``now`` is always passed in).

``next_after`` steps in UTC and matches against the LOCAL wall-clock, so DST is handled
by construction (a spring-forward-skipped local time simply never matches). The v1
cadence set (PRD D3): daily / weekly / specific-weekdays / hourly / every-N-minutes —
anything else is declined by ``validate`` rather than silently approximated.
"""
from __future__ import annotations

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
    cand = now_utc.astimezone(timezone.utc).replace(second=0, microsecond=0) \
        + timedelta(minutes=1)
    for _ in range(_MAX_STEPS):
        if _matches(sched.cadence, cand.astimezone(tz)):
            return cand
        cand += timedelta(minutes=1)
    raise InvalidCadence("no cadence match within 8 days (unsupported pattern?)")


def apply_patch(sched: Schedule, *, minute=_UNSET, hour=_UNSET, weekdays=_UNSET,
                every_n_minutes=_UNSET) -> Schedule:
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
    )
    validate(new_cad)
    from dataclasses import replace
    return replace(sched, cadence=new_cad)


def describe(cad: Cadence) -> str:
    """A human-readable label, derived from the structured fields (the web view + the
    agent's confirm-back use this — never a model-supplied string)."""
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
