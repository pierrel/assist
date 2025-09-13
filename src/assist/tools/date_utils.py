from __future__ import annotations

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from langchain_core.tools import tool


@tool
def get_current_date() -> dict:
    """Returns today's date in ISO format (YYYY-MM-DD).

    when_to_use:
    - Need the current date for logging or scheduling.
    - Anchor calculations relative to today.
    - Validate date-based conditions.
    when_not_to_use:
    - Require current time-of-day or timezone conversions.
    - Working with non-Gregorian calendars.
    args_schema: {}
    preconditions_permissions: {}
    side_effects:
    - None; idempotent: true; retry_safe: true.
    cost_latency: "<1ms; free"
    pagination_cursors:
    - input_cursor: none
    - next_cursor: none
    errors: {}
    returns:
    - date (str): ISO date string.
    - brief_summary (str): Same as ``date``.
    examples:
    - input: {}
      output: {"date": "2025-09-01", "brief_summary": "2025-09-01"}
    """
    today = date.today().isoformat()
    return {"date": today, "brief_summary": today}


@tool
def offset_date(base_date: str, days: int = 0, weeks: int = 0, months: int = 0) -> dict:
    """Returns the ISO date offset from ``base_date`` by the given days, weeks, or months.

    when_to_use:
    - Calculate a deadline or reminder date.
    - Determine past or future dates for planning.
    - Adjust schedules by fixed intervals.
    when_not_to_use:
    - Need time-of-day adjustments.
    - ``base_date`` format is uncertain.
    args_schema:
    - base_date (str): ISO date, e.g. "2025-09-01".
    - days (int, default=0): [-365, 365].
    - weeks (int, default=0): [-52, 52].
    - months (int, default=0): [-12, 12].
    preconditions_permissions:
    - ``base_date`` must parse as ISO format.
    side_effects:
    - None; idempotent: true; retry_safe: true.
    cost_latency: "<1ms; free"
    pagination_cursors:
    - input_cursor: none
    - next_cursor: none
    errors:
    - invalid_date: ``base_date`` not ISO; use YYYY-MM-DD.
    returns:
    - date (str): Resulting ISO date.
    - brief_summary (str): Same as ``date``.
    examples:
    - input: {"base_date": "2025-09-01", "days": 7}
      output: {"date": "2025-09-08", "brief_summary": "2025-09-08"}
    """
    dt = datetime.fromisoformat(base_date).date()
    dt = dt + relativedelta(days=days, weeks=weeks, months=months)
    result = dt.isoformat()
    return {"date": result, "brief_summary": result}


@tool
def diff_dates(
    start_date: str,
    end_date: str,
    unit: str = "days",
    mode: str = "all",
) -> dict:
    """Returns the difference between two dates in the specified unit.

    when_to_use:
    - Measure duration between two dates.
    - Evaluate schedule gaps or delays.
    - Count business days or weekends.
    when_not_to_use:
    - Need sub-day precision.
    - Date formats are unknown.
    args_schema:
    - start_date (str): ISO date, e.g. "2025-09-01".
    - end_date (str): ISO date, e.g. "2025-09-10".
    - unit (str, enum['days','weeks','months'], default='days').
    - mode (str, enum['all','weekdays','weekends'], default='all').
    preconditions_permissions:
    - Dates must parse as ISO format.
    side_effects:
    - None; idempotent: true; retry_safe: true.
    cost_latency: "<1ms; free"
    pagination_cursors:
    - input_cursor: none
    - next_cursor: none
    errors:
    - invalid_date: Inputs not ISO format.
    - invalid_choice: ``unit`` or ``mode`` not recognized.
    returns:
    - difference (int): Numeric difference in ``unit``.
    - brief_summary (str): e.g. "9 days".
    examples:
    - input: {"start_date": "2025-09-01", "end_date": "2025-09-10"}
      output: {"difference": 9, "brief_summary": "9 days"}
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
        diff = delta_days // 7
    elif unit == "months":
        rd = relativedelta(end, start)
        diff = abs(rd.years * 12 + rd.months)
    else:
        diff = delta_days

    summary = f"{diff} {unit}"
    return {"difference": diff, "brief_summary": summary}


__all__ = ["get_current_date", "offset_date", "diff_dates"]
