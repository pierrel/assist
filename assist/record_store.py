"""Per-thread JSON record store — the disk-is-truth persistence shared by schedules and
message-event subscriptions.

Each thread's records live in ``<root_dir>/<tid>/<FILENAME>`` (atomic tmp+rename, like
``status.json``), so they survive restart and die with the thread via the thread dir's
rmtree — no eviction hook. One process-wide lock serializes every read-modify-write,
because multiple writers race on a thread's file (a poll thread, a tool in the thread's
turn, a web delete). The lock is held only for the fast file op, never across a dispatch or
a turn. Subclasses set ``FILENAME``/``CAP``, the record ``from_dict``, and the exception
types to raise; records must expose ``id``, ``thread_id``, and ``to_dict()``.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class RecordCapExceeded(Exception):
    """Creating would exceed the per-thread cap."""


class RecordNotFound(Exception):
    """No record with that id on that thread."""


class PerThreadJsonStore(Generic[T]):
    FILENAME: str = ""          # subclass sets, e.g. "schedules.json"
    CAP: int = 0                # 0 = uncapped
    CAP_EXC: type[Exception] = RecordCapExceeded      # subclass may specialize
    NOTFOUND_EXC: type[Exception] = RecordNotFound

    def __init__(self, root_dir: str):
        self._root = root_dir
        self._lock = threading.Lock()

    @staticmethod
    def _from_dict(d: dict) -> T:
        raise NotImplementedError

    def _path(self, tid: str) -> str:
        # A tid is one safe path segment; reject a crafted id (traversal/separator) by
        # construction — user-facing routes pass tid straight in.
        if not tid or tid in (".", "..") or os.sep in tid or (os.altsep and os.altsep in tid):
            raise self.NOTFOUND_EXC(tid)
        return os.path.join(self._root, tid, self.FILENAME)

    def _read(self, tid: str) -> list[T]:
        """Read a thread's records (callers hold the lock). Missing/corrupt → []."""
        try:
            with open(self._path(tid)) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        return [self._from_dict(d) for d in data]

    def _write(self, tid: str, recs: list[T]) -> None:
        """Atomically persist a thread's records (callers hold the lock)."""
        path = self._path(tid)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump([r.to_dict() for r in recs], f)
        os.replace(tmp, path)

    # --- per-thread (used by the tools) -----------------------------------------
    def for_thread(self, tid: str) -> list[T]:
        with self._lock:
            return self._read(tid)

    def add(self, rec: T) -> T:
        with self._lock:
            recs = self._read(rec.thread_id)
            if self.CAP and len(recs) >= self.CAP:
                raise self.CAP_EXC(
                    f"this thread already has {self.CAP} {self.FILENAME.split('.')[0]}; "
                    f"delete one before adding another")
            recs.append(rec)
            self._write(rec.thread_id, recs)
            return rec

    def update(self, tid: str, rid: str, fn: Callable[[T], T]) -> T:
        """Read-modify-write a single record under the lock. ``fn`` maps the found record
        to its replacement."""
        with self._lock:
            recs = self._read(tid)
            for i, r in enumerate(recs):
                if r.id == rid:
                    recs[i] = fn(r)
                    self._write(tid, recs)
                    return recs[i]
            raise self.NOTFOUND_EXC(rid)

    def remove(self, tid: str, rid: str) -> None:
        with self._lock:
            recs = self._read(tid)
            kept = [r for r in recs if r.id != rid]
            if len(kept) == len(recs):
                raise self.NOTFOUND_EXC(rid)
            self._write(tid, kept)

    # --- across all threads -----------------------------------------------------
    def all(self) -> list[T]:
        with self._lock:
            out: list[T] = []
            for tid in self._list_tids():
                out.extend(self._read(tid))
            return out

    def _list_tids(self) -> list[str]:
        try:
            entries = os.listdir(self._root)
        except FileNotFoundError:
            return []
        # Only descend into actual thread dirs — the root also holds non-dir files
        # (e.g. threads.db); isdir-first avoids stat'ing file/<FILENAME>.
        return [e for e in entries
                if os.path.isdir(os.path.join(self._root, e))
                and os.path.isfile(self._path(e))]
