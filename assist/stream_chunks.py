"""Defensive normalization of langgraph stream-chunk payloads.

Shared by every consumer of `Thread.stream_message` — the assist CLI
(`manage/cli.py`) and the emacsos-server NDJSON gateway (`emacsos_server`,
which installs assist editable) — so langgraph-version-specific chunk shapes
are handled in exactly one place rather than re-discovered per consumer.
"""
from __future__ import annotations

from typing import Any

from langgraph.types import Overwrite


def unwrap_messages(value: Any) -> list:
    """Return the messages carried by a node's ``messages`` channel update,
    regardless of how langgraph wrapped them.

    Under ``stream_mode=["messages", "updates"]`` an ``updates`` chunk's
    per-node ``messages`` value is normally a list.  langgraph 1.2 may instead
    deliver an ``Overwrite`` wrapper (a node *replacing* the channel rather
    than appending).  ``Overwrite`` is truthy and not iterable, so the naive
    ``for m in value or []`` raises ``TypeError: 'Overwrite' object is not
    iterable``.  This unwraps the known shapes and returns ``[]`` for anything
    unexpected — it never raises.
    """
    if isinstance(value, Overwrite):
        value = value.value
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None:
        return []
    # A lone message-like object: the messages channel is normally a list,
    # but tolerate a single message so a stray shape isn't silently dropped.
    if hasattr(value, "content") or hasattr(value, "type"):
        return [value]
    return []
