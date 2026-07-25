"""Per-thread durable event log — the job system's internal PUSH target.

Runs push events here as they act (a continuation scheduled / dispatched /
completed / failed); readers PULL on their own schedule (currently offline
diagnostics — ``read_events`` is the read API; the web render reads the
journal + status directly today). This is the "push
architecture with a pull UI" decision (docs/2026-07-19-blank-slate-assessment
.org): event production is decoupled from run completion with zero UI push
machinery. Division of truth: the checkpoint holds MESSAGES, the run store holds
pending WORK, status.json holds the live banner, and this log
holds the durable HISTORY of what the job system did.

Format: ``<thread_dir>/events.jsonl`` — one JSON object per line with at least
``ts`` (UTC ISO) and ``kind``. Appends are single ``write()`` calls on an
O_APPEND descriptor (atomic for line-sized writes on POSIX); a reader may see
a torn FINAL line mid-append and must skip it — ``read_events`` does. The log
dies with the thread dir like every other sidecar; it is history, so nothing
ever rewrites it.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

EVENTS_FILE = "events.jsonl"


def append_event(thread_dir: str, kind: str, **fields) -> None:
    """Append one event. Best-effort: the log is observability, and a log
    failure must never fail the run pushing to it."""
    try:
        # dumps inside the guard: a non-serializable field must degrade to a
        # logged warning, not fail the run — per this function's contract.
        line = json.dumps({"ts": datetime.now(timezone.utc).isoformat(),
                           "kind": kind, **fields}, default=str)
        with open(os.path.join(thread_dir, EVENTS_FILE), "a") as f:
            f.write(line + "\n")
    except Exception:
        logger.warning("event log append failed for %s (%s)", thread_dir, kind,
                       exc_info=True)


def read_events(thread_dir: str, kind: str | None = None) -> list[dict]:
    """All events (oldest first), optionally filtered by kind. Missing log →
    []. A torn final line (reader racing an append) is skipped."""
    try:
        with open(os.path.join(thread_dir, EVENTS_FILE)) as f:
            raw = f.read()
    except OSError:
        return []
    out = []
    for line in raw.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn final line mid-append; history lines never rewrite
        if kind is None or ev.get("kind") == kind:
            out.append(ev)
    return out
