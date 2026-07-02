"""Schedule persistence — disk IS the source of truth (no in-memory cache).

Each thread's schedules live in ``<root_dir>/<tid>/schedules.json``; the shared
:class:`assist.record_store.PerThreadJsonStore` provides the lock-serialized, atomic
tmp+rename read-modify-write (three writers race: the scheduler advancing ``next_fire_at``
on the poll thread, a ``modify``/``pause`` tool in the thread's turn, and a web delete).
This subclass adds only the schedule record type, cap, and the ``due`` query.
"""
from __future__ import annotations

from datetime import datetime

from assist.record_store import (
    PerThreadJsonStore,
    RecordCapExceeded,
    RecordNotFound,
)
from assist.schedule.model import Schedule

SCHEDULES_FILE = "schedules.json"
CAP_PER_THREAD = 5  # PRD: a small cap so schedules can't run away


class ScheduleCapExceeded(RecordCapExceeded):
    """Creating would exceed CAP_PER_THREAD on this thread."""


class ScheduleNotFound(RecordNotFound):
    """No schedule with that id on that thread."""


class ScheduleStore(PerThreadJsonStore[Schedule]):
    """Disk-backed schedule store. ``root_dir`` is the thread root (``MANAGER.root_dir``);
    a thread's dir is ``<root_dir>/<tid>``, matching ``ThreadManager.thread_dir``."""

    FILENAME = SCHEDULES_FILE
    CAP = CAP_PER_THREAD
    CAP_EXC = ScheduleCapExceeded
    NOTFOUND_EXC = ScheduleNotFound

    @staticmethod
    def _from_dict(d: dict) -> Schedule:
        return Schedule.from_dict(d)

    def due(self, now_utc: datetime) -> list[Schedule]:
        """Enabled schedules whose next_fire_at is at or before now."""
        return [s for s in self.all()
                if s.enabled and s.next_fire_at
                and datetime.fromisoformat(s.next_fire_at) <= now_utc]
