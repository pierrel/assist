from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import List

from dateutil.relativedelta import relativedelta
from langchain_core.tools import tool


@tool
def current_date() -> str:
    """Return the current date in ISO format."""
    return date.today().isoformat()


@tool
def date_offset(base_date: str, days: int = 0, weeks: int = 0, months: int = 0) -> str:
    """Return a date offset from ``base_date`` by the given amount."""
    dt = datetime.fromisoformat(base_date).date()
    dt = dt + relativedelta(days=days, weeks=weeks, months=months)
    return dt.isoformat()


@tool
def date_diff(
    start_date: str,
    end_date: str,
    unit: str = "days",
    mode: str = "all",
) -> str:
    """Return the difference between two dates.

    ``unit`` may be ``days``, ``weeks`` or ``months``. ``mode`` controls which
    days are counted: ``all`` (default), ``weekdays`` or ``weekends``.
    """
    start = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()

    if mode == "all":
        delta_days = abs((end - start).days)
    else:
        step = 1 if end >= start else -1
        delta_days = 0
        current = start
        while current != end:
            is_weekend = current.weekday() >= 5
            if mode == "weekdays" and not is_weekend:
                delta_days += 1
            elif mode == "weekends" and is_weekend:
                delta_days += 1
            current += timedelta(days=step)

    if unit == "weeks":
        return str(delta_days // 7)
    if unit == "months":
        rd = relativedelta(end, start)
        months = abs(rd.years * 12 + rd.months)
        return str(months)
    return str(delta_days)


__all__ = ["current_date", "date_offset", "date_diff"]
