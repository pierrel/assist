"""Read-only thread catalog — the client-agnostic "what threads exist and what's
their state" surface (shared session API; docs/2026-07-21-voice-call-tech-design.org
§3).

The receptionist (voice) consumes this WITHOUT loading any conversation: no
checkpoint access, no ``Thread`` construction, no model call, no message read.
Thread CONTENTS are unreachable by construction — this module never reads
``threads.db`` or the langgraph checkpoint (it holds a ``ThreadManager``, which owns
them, but only calls its directory operations — ``list`` / ``thread_dir``); it reads
just the per-thread sidecar files (``description.txt`` / ``status.json`` / the urgent
marker) plus the dir mtime.

It mirrors the *file-reading halves* of the web layer's per-thread reads
(``manage/web/state.py``'s ``_get_status`` / ``_has_urgent`` / ``description.txt``)
as one checkpoint-free surface; a later increment can point the web thread-list at
it so the two stop re-deriving the same reads (web still owns them today).
Description GENERATION (which needs a model) and the web's ``DESCRIPTION_CACHE``
stay web-side — the catalog only reads the file, falling back to ``"New thread"``
on a miss (never generating one).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

# These three sidecar names are authored by the web layer (manage/web/state.py);
# it stays the source of truth until web is migrated onto this catalog. Renaming
# one there without updating here silently blanks that field (e.g. urgent=False).
_DESCRIPTION_FILE = "description.txt"
_STATUS_FILE = "status.json"
_URGENT_MARKER = "urgent_response"   # the notify() durable marker (state.py:_urgent_path)


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    description: str      # one-line summary, never the transcript; "New thread" if none
    status: str          # the status.json stage (ready / processing / error / …)
    updated_at: float    # thread-dir mtime, epoch seconds (the list sort key)
    urgent: bool         # the notify()-set durable urgent marker


class ThreadCatalog:
    """Read-only view over a ``ThreadManager``'s threads. Takes the manager (so one
    dir scan and one tid-validation path are reused) and NEVER constructs a Thread."""

    def __init__(self, manager):
        self._manager = manager

    def _dir(self, tid: str) -> str:
        return self._manager.thread_dir(tid)   # validates tid (raises on traversal/NUL)

    def _description(self, tdir: str) -> str:
        try:
            with open(os.path.join(tdir, _DESCRIPTION_FILE)) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        return line     # first non-empty line — a one-line summary, so
                                        # an embedded newline can't break a numbered/spoken list
        except Exception:
            pass
        return "New thread"     # missing/unreadable/blank ⇒ not generated here (needs a model)

    def _status(self, tdir: str) -> str:
        try:
            with open(os.path.join(tdir, _STATUS_FILE)) as f:
                stage = json.load(f).get("stage")
        except Exception:
            return "ready"          # missing/corrupt ⇒ the same default _get_status uses
        # A non-string stage (null/number in otherwise-valid JSON) is corrupt too:
        # degrade to the default rather than surface "None"/"123" to a client.
        return stage if isinstance(stage, str) and stage else "ready"

    def _urgent(self, tdir: str) -> bool:
        try:
            return os.path.isfile(os.path.join(tdir, _URGENT_MARKER))
        except OSError:
            return False

    def get(self, tid: str) -> CatalogEntry | None:
        """One entry, or None if the tid is invalid or the thread doesn't exist."""
        try:
            tdir = self._dir(tid)
        except Exception:
            return None             # invalid tid shape — never a 500 into a client
        if not os.path.isdir(tdir):
            return None
        try:
            updated_at = os.path.getmtime(tdir)
        except OSError:
            updated_at = 0.0
        return CatalogEntry(id=tid, description=self._description(tdir),
                            status=self._status(tdir), updated_at=updated_at,
                            urgent=self._urgent(tdir))

    def entries(self, limit: int | None = None) -> list[CatalogEntry]:
        """Live threads as catalog entries, in ``ThreadManager.list`` order
        (mtime-descending, soft-deleted filtered). One dir scan for the ids; a few
        small sidecar reads each. ``limit`` stops after that many entries — since the
        order is mtime-descending, that's the most-recent N, so the latency-sensitive
        listing path reads only N threads' sidecars instead of every thread's."""
        out: list[CatalogEntry] = []
        for tid in self._manager.list():
            entry = self.get(tid)
            if entry is not None:
                out.append(entry)
                if limit is not None and len(out) >= limit:
                    break
        return out
