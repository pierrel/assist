"""The agent-facing schedule tools (the conversational surface).

Thread-scoped: each reads its ``thread_id`` (and the rider's tz) from the run config,
so a schedule belongs to the thread it's created in. ``modify`` merges a SPARSE delta
server-side — the model passes only the field(s) it's changing + the id, never the whole
cadence (the relative-edit requirement, and the shape this small model gets right).

Built by ``schedule_tools(store)`` and wired into the web ``AgentSpec`` (not core
built-ins): a schedule's effect needs a co-resident Scheduler, which only the web runs.
Tools never raise into the agent loop — every failure returns a corrective string.
"""
from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime, timezone

from langgraph.config import get_config

from assist.context_rider import CONTEXT_RIDER_KEY
from assist.schedule import cadence
from assist.schedule.cadence import InvalidCadence
from assist.schedule.model import Cadence, Schedule
from assist.schedule.store import ScheduleCapExceeded, ScheduleNotFound


def _cfg() -> dict:
    return (get_config() or {}).get("configurable") or {}


def _thread_id() -> str | None:
    return _cfg().get("thread_id")


def _tz() -> str | None:
    rider = _cfg().get(CONTEXT_RIDER_KEY)
    return getattr(rider, "tz", None)


def _line(s: Schedule) -> str:
    return (f"[{s.id}] {cadence.describe(s.cadence)} — \"{s.prompt}\""
            f"{'' if s.enabled else ' (paused)'}; "
            f"next: {cadence.fmt_instant(s.next_fire_at, s.tz, empty='not scheduled')}")


def schedule_tools(store) -> list:
    """Return the six thread-scoped schedule tools closing over ``store``."""

    def create_schedule(prompt: str, minute: int = 0, hour: int | None = None,
                        weekdays: list[int] | None = None,
                        every_n_minutes: int | None = None) -> str:
        """Schedule PROMPT to run automatically on a recurring cadence, in THIS thread.

        Use ONE cadence shape:
        - daily at a time: hour=7, minute=0
        - specific weekdays at a time: hour=7, minute=0, weekdays=[0,1,2,3,4] (0=Mon..6=Sun)
        - hourly at a given minute: minute=30 (omit hour)
        - every N minutes: every_n_minutes=30 (omit hour)
        Times are in the user's local timezone. Returns the saved schedule + next run.
        """
        tid = _thread_id()
        if not tid:
            return "Couldn't schedule: no active thread."
        tz = _tz()
        if not tz:
            return "Couldn't schedule: I don't know your timezone for this message."
        cad = Cadence(minute=minute, hour=hour,
                      weekdays=tuple(weekdays) if weekdays is not None else None,
                      every_n_minutes=every_n_minutes)
        try:
            cadence.validate(cad)
        except InvalidCadence as e:
            return f"Couldn't schedule: {e}"
        sched = Schedule(id=os.urandom(6).hex(), thread_id=tid, prompt=prompt, cadence=cad,
                         tz=tz, created_at=datetime.now(timezone.utc).isoformat())
        try:
            sched = sched.with_next_fire(cadence.next_after(sched, datetime.now(timezone.utc)).isoformat())
        except InvalidCadence as e:
            return f"Couldn't schedule: {e}"
        try:
            store.add(sched)
        except ScheduleCapExceeded as e:
            return f"Couldn't schedule: {e}"
        return f"Scheduled. {_line(sched)}"

    def list_schedules() -> str:
        """List THIS thread's schedules (id, cadence, prompt, next run, paused state)."""
        tid = _thread_id()
        if not tid:
            return "No active thread."
        scheds = store.for_thread(tid)
        if not scheds:
            return "This thread has no schedules."
        return "\n".join(_line(s) for s in scheds)

    def modify_schedule(schedule_id: str, minute: int | None = None, hour: int | None = None,
                        weekdays: list[int] | None = None,
                        every_n_minutes: int | None = None) -> str:
        """Change an existing schedule. Pass ONLY the field(s) you're changing + the id;
        omitted fields keep their current value (e.g. "fire at 5am" -> hour=5 only).
        To switch between a clock schedule and an every-N-minutes one, delete and recreate.
        """
        tid = _thread_id()
        if not tid:
            return "No active thread."
        delta = {}
        if minute is not None:
            delta["minute"] = minute
        if hour is not None:
            delta["hour"] = hour
        if weekdays is not None:
            delta["weekdays"] = tuple(weekdays)
        if every_n_minutes is not None:
            delta["every_n_minutes"] = every_n_minutes
        if not delta:
            return "Nothing to change — pass the field(s) to update."
        try:
            current = next((s for s in store.for_thread(tid) if s.id == schedule_id), None)
            if current is None:
                return f"No schedule {schedule_id} on this thread."
            patched = cadence.apply_patch(current, **delta)
            patched = patched.with_next_fire(
                cadence.next_after(patched, datetime.now(timezone.utc)).isoformat())
            saved = store.update(tid, schedule_id, lambda _s: patched)
        except InvalidCadence as e:
            return f"Couldn't change it: {e}"
        except ScheduleNotFound:
            return f"No schedule {schedule_id} on this thread."
        return f"Updated. {_line(saved)}"

    def _set_enabled(schedule_id: str, enabled: bool, verb: str) -> str:
        tid = _thread_id()
        if not tid:
            return "No active thread."

        def _apply(s: Schedule) -> Schedule:
            s = replace(s, enabled=enabled)
            if enabled:  # resume: recompute forward so a long-paused schedule doesn't
                # fire immediately for missed windows (the no-catch-up guarantee).
                s = s.with_next_fire(cadence.next_after(s, datetime.now(timezone.utc)).isoformat())
            return s
        try:
            saved = store.update(tid, schedule_id, _apply)
        except ScheduleNotFound:
            return f"No schedule {schedule_id} on this thread."
        return f"{verb}. {_line(saved)}"

    def pause_schedule(schedule_id: str) -> str:
        """Stop a schedule from running (keep it; can be resumed later)."""
        return _set_enabled(schedule_id, False, "Paused")

    def resume_schedule(schedule_id: str) -> str:
        """Resume a paused schedule."""
        return _set_enabled(schedule_id, True, "Resumed")

    def delete_schedule(schedule_id: str) -> str:
        """Delete a schedule from this thread permanently."""
        tid = _thread_id()
        if not tid:
            return "No active thread."
        try:
            store.remove(tid, schedule_id)
        except ScheduleNotFound:
            return f"No schedule {schedule_id} on this thread."
        return f"Deleted schedule {schedule_id}."

    return [create_schedule, list_schedules, modify_schedule,
            pause_schedule, resume_schedule, delete_schedule]
