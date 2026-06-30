"""Schedule persistence — disk IS the source of truth (no in-memory cache).

Each thread's schedules live in ``<root_dir>/<tid>/schedules.json`` (atomic tmp+rename,
like ``status.json``), so they survive restart and die with the thread via the thread
dir's rmtree — no eviction hook needed. One process-wide lock serializes every
read-modify-write, because three writers race on a thread's file: the scheduler
advancing ``next_fire_at`` (poll thread), a ``modify``/``pause`` tool (in the thread's
turn), and a web delete. The lock is held only for the fast file op, never across a
dispatch or a turn.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime

from assist.schedule.model import Schedule

SCHEDULES_FILE = "schedules.json"
CAP_PER_THREAD = 5  # PRD: a small cap so schedules can't run away


class ScheduleCapExceeded(Exception):
    """Creating would exceed CAP_PER_THREAD on this thread."""


class ScheduleNotFound(Exception):
    """No schedule with that id on that thread."""


class ScheduleStore:
    """Disk-backed schedule store. ``root_dir`` is the thread root (``MANAGER.root_dir``);
    a thread's dir is ``<root_dir>/<tid>``, matching ``ThreadManager.thread_dir``."""

    def __init__(self, root_dir: str):
        self._root = root_dir
        self._lock = threading.Lock()

    def _path(self, tid: str) -> str:
        return os.path.join(self._root, tid, SCHEDULES_FILE)

    def _read(self, tid: str) -> list[Schedule]:
        """Read a thread's schedules (callers hold the lock). Missing/corrupt → []."""
        try:
            with open(self._path(tid)) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        return [Schedule.from_dict(d) for d in data]

    def _write(self, tid: str, scheds: list[Schedule]) -> None:
        """Atomically persist a thread's schedules (callers hold the lock)."""
        path = self._path(tid)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump([s.to_dict() for s in scheds], f)
        os.replace(tmp, path)

    # --- per-thread (used by the tools) -----------------------------------------
    def for_thread(self, tid: str) -> list[Schedule]:
        with self._lock:
            return self._read(tid)

    def add(self, sched: Schedule) -> Schedule:
        with self._lock:
            scheds = self._read(sched.thread_id)
            if len(scheds) >= CAP_PER_THREAD:
                raise ScheduleCapExceeded(
                    f"this thread already has {CAP_PER_THREAD} schedules; "
                    f"delete one before adding another")
            scheds.append(sched)
            self._write(sched.thread_id, scheds)
            return sched

    def update(self, tid: str, sid: str, fn) -> Schedule:
        """Read-modify-write a single schedule under the lock. ``fn`` maps the found
        Schedule to its replacement (used for patch / enable / advance next_fire_at)."""
        with self._lock:
            scheds = self._read(tid)
            for i, s in enumerate(scheds):
                if s.id == sid:
                    scheds[i] = fn(s)
                    self._write(tid, scheds)
                    return scheds[i]
            raise ScheduleNotFound(sid)

    def remove(self, tid: str, sid: str) -> None:
        with self._lock:
            scheds = self._read(tid)
            kept = [s for s in scheds if s.id != sid]
            if len(kept) == len(scheds):
                raise ScheduleNotFound(sid)
            self._write(tid, kept)

    # --- across all threads (used by the scheduler + the web view) --------------
    def all(self) -> list[Schedule]:
        with self._lock:
            out: list[Schedule] = []
            for tid in self._list_tids():
                out.extend(self._read(tid))
            return out

    def _list_tids(self) -> list[str]:
        try:
            entries = os.listdir(self._root)
        except FileNotFoundError:
            return []
        return [e for e in entries if os.path.isfile(self._path(e))]

    def due(self, now_utc: datetime) -> list[Schedule]:
        """Enabled schedules whose next_fire_at is at or before now."""
        return [s for s in self.all()
                if s.enabled and s.next_fire_at
                and datetime.fromisoformat(s.next_fire_at) <= now_utc]
