"""Small in-memory journals for the phone's provisional Run text.

The journal is deliberately not a second source of conversation truth.  It is a
bounded, process-local observation aid keyed by the durable Run work id.
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


MAX_WORKS = 4
MAX_TEXT_BYTES = 256 * 1024
MAX_DELTAS = 4096
# Worst-case JSON escaping expands each control character to six ASCII bytes.
# 7 KiB therefore leaves room for the event envelope under the 48 KiB record cap.
PHONE_DELTA_CHUNK_BYTES = 7 * 1024


@dataclass
class _Entry:
    active: bool = False
    terminal: bool = False
    gone: bool = False
    attempt: int = 1
    revision: int = 0
    deltas: list[dict[str, Any]] = field(default_factory=list)
    text_bytes: int = 0
    truncated: bool = False


class RunStreamJournal:
    """Bounded journals; all mutable state is protected by one short lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[str, str], _Entry] = OrderedDict()
        self._observers: dict[tuple[str, str], int] = {}
        self._gone_observers: set[tuple[str, str]] = set()

    def reserve(self, thread_id: str, work_id: str) -> bool:
        """Reserve an unreadable entry, evicting only an old terminal or gone entry."""
        key = (thread_id, work_id)
        with self._lock:
            if key in self._entries:
                return True
            while len(self._entries) >= MAX_WORKS:
                old_key = next((k for k, value in self._entries.items()
                                if value.terminal or value.gone), None)
                if old_key is None:
                    return False
                self._entries.pop(old_key)
            self._entries[key] = _Entry()
            return True

    def activate(self, thread_id: str, work_id: str) -> bool:
        with self._lock:
            entry = self._entries.get((thread_id, work_id))
            if entry is None:
                return False
            if not entry.active:
                entry.active = True
                entry.revision += 1
            return True

    def discard(self, thread_id: str, work_id: str) -> None:
        with self._lock:
            self._entries.pop((thread_id, work_id), None)

    def reset_attempt(self, thread_id: str, work_id: str) -> int | None:
        with self._lock:
            entry = self._entries.get((thread_id, work_id))
            if entry is None or entry.terminal:
                return None
            entry.attempt += 1
            entry.deltas.clear()
            entry.text_bytes = 0
            entry.truncated = False
            entry.revision += 1
            return entry.attempt

    def publish_delta(self, thread_id: str, work_id: str, text: str) -> dict[str, Any] | None:
        """Retain and return a current-attempt delta, or mark truncation once."""
        encoded = text.encode("utf-8")
        with self._lock:
            entry = self._entries.get((thread_id, work_id))
            if entry is None or entry.terminal or entry.truncated:
                return None
            if (len(entry.deltas) >= MAX_DELTAS
                    or entry.text_bytes + len(encoded) > MAX_TEXT_BYTES):
                entry.truncated = True
                entry.revision += 1
                return None
            value = {"attempt": entry.attempt, "index": len(entry.deltas) + 1,
                     "text": text}
            entry.deltas.append(value)
            entry.text_bytes += len(encoded)
            entry.revision += 1
            return dict(value)

    def finish(self, thread_id: str, work_id: str) -> None:
        with self._lock:
            entry = self._entries.get((thread_id, work_id))
            if entry is not None and not entry.terminal:
                entry.terminal = True
                entry.revision += 1

    def mark_thread_gone(self, thread_id: str) -> None:
        """Close every retained or currently observed Run for a deleted thread."""
        with self._lock:
            for (candidate, _), entry in self._entries.items():
                if candidate == thread_id:
                    entry.gone = True
            self._gone_observers.update(
                key for key in self._observers if key[0] == thread_id)

    def open_observer(self, thread_id: str, work_id: str) -> None:
        """Retain a bounded phone observer so deletion closes status-only streams."""
        key = (thread_id, work_id)
        with self._lock:
            self._observers[key] = self._observers.get(key, 0) + 1

    def close_observer(self, thread_id: str, work_id: str) -> None:
        """Release one phone observer and its no-longer-observable tombstone."""
        key = (thread_id, work_id)
        with self._lock:
            count = self._observers.get(key, 0)
            if count <= 1:
                self._observers.pop(key, None)
                self._gone_observers.discard(key)
            else:
                self._observers[key] = count - 1

    def is_gone(self, thread_id: str, work_id: str) -> bool:
        """Return whether this exact retained observation was deleted."""
        with self._lock:
            entry = self._entries.get((thread_id, work_id))
            return bool(entry and entry.gone) or (thread_id, work_id) in self._gone_observers

    @staticmethod
    def _snapshot(entry: _Entry) -> dict[str, Any]:
        return {"attempt": entry.attempt,
                "deltas": [dict(delta) for delta in entry.deltas],
                "truncated": entry.truncated,
                "terminal": entry.terminal}

    def snapshot_if_changed(
            self, thread_id: str, work_id: str, known_revision: int | None
    ) -> tuple[int, dict[str, Any]] | None:
        """Copy a live journal only when its provisional state changed."""
        with self._lock:
            entry = self._entries.get((thread_id, work_id))
            if (entry is None or not entry.active
                    or entry.revision == known_revision):
                return None
            return entry.revision, self._snapshot(entry)

    def read(self, thread_id: str, work_id: str) -> dict[str, Any] | None:
        """Return one immutable journal snapshot; inactive reservations are hidden."""
        with self._lock:
            entry = self._entries.get((thread_id, work_id))
            if entry is None or not entry.active:
                return None
            return self._snapshot(entry)


RUN_STREAMS = RunStreamJournal()


def encode_sse(event: str, value: dict[str, Any]) -> str:
    """Encode one safe single-line SSE record using server-selected event names."""
    return f"event: {event}\ndata: {json.dumps(value, ensure_ascii=True, separators=(',', ':'))}\n\n"
