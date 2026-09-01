"""Canonical visible conversation records for the web and live captures.

The checkpointer stores both human-facing messages and host control frames.  This
module is the small, pure boundary that gives both presentation and capture the
same interpretation of those frames.  It deliberately exposes no workspace,
tool-result, or checkpoint implementation detail.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

from langchain.messages import AIMessage, HumanMessage


CONTINUATION_RIDER = "[Continuing my earlier work — background follow-up] "
TASK_COMPLETION_RIDER = "[Background task finished] "
INTERJECTION_FRAME = "[Mid-turn message from the user — sent while you were working] "
INTERJECTION_GUIDE = (
    "\n\n(This message arrived mid-turn. The user's latest word wins: if it "
    "changes what they want, redirect your remaining work now; if it adds "
    "scope, fold it in. If it asks you to stop, do no further work and reply "
    "with a brief account of what you already completed.)"
)
_RENDER_BLOCK_RE = re.compile(r"```render\s*\n.*?\n```", re.DOTALL)


@dataclass(frozen=True)
class VisibleRecord:
    """One normalized chronological record safe to show or judge."""

    id: str
    order: int
    role: str
    text: str
    source_kind: str
    capture_eligible: bool


def _as_text(value: object) -> str:
    return value if isinstance(value, str) else str(value or "")


def _assistant_text(text: str) -> str:
    """Represent a rendered workspace artifact honestly without copying it."""
    return _RENDER_BLOCK_RE.sub("[Rendered artifact unavailable to capture]", text)


def _interjection_text(text: str) -> str:
    inner = text[len(INTERJECTION_FRAME):]
    cut = inner.rfind(INTERJECTION_GUIDE)
    return inner[:cut] if cut != -1 else inner


def visible_records(raw: Iterable[object]) -> list[VisibleRecord]:
    """Project checkpoint messages into the human-visible record contract."""
    records: list[VisibleRecord] = []
    source_order = 0

    def add(role: str, text: str, source_kind: str, eligible: bool) -> None:
        records.append(VisibleRecord(
            id=f"r{len(records) + 1:04d}", order=source_order, role=role,
            text=text, source_kind=source_kind, capture_eligible=eligible,
        ))

    for message in raw:
        source_order += 1
        if isinstance(message, HumanMessage):
            text = _as_text(message.content)
            if text.startswith(CONTINUATION_RIDER):
                add("assistant", text[len(CONTINUATION_RIDER):], "continuation", False)
            elif text.startswith(TASK_COMPLETION_RIDER):
                add("assistant", text[len(TASK_COMPLETION_RIDER):], "task", False)
            elif text.startswith(INTERJECTION_FRAME):
                add("user", _interjection_text(text), "interjection", True)
            else:
                add("user", text, "user", True)
        elif isinstance(message, AIMessage):
            calls = getattr(message, "tool_calls", None)
            if calls:
                add("tools", "", "tool", False)
            text = _as_text(message.content)
            if text:
                add("assistant", _assistant_text(text), "assistant", True)
    return records


def visible_records_from_dicts(
    messages: Iterable[Mapping[str, object]], *, capture_safe: bool = False,
) -> list[VisibleRecord]:
    """Apply the frame contract to web dicts, retaining render directives by default.

    The page still needs a trusted server-side render directive to construct an
    embed.  Capture callers use :func:`visible_records` from raw messages and
    therefore always receive the artifact-unavailable marker instead.
    """
    records: list[VisibleRecord] = []
    for order, message in enumerate(messages, start=1):
        role = _as_text(message.get("role"))
        text = _as_text(message.get("content"))
        if role == "user" and text.startswith(CONTINUATION_RIDER):
            role, text, kind, eligible = "assistant", text[len(CONTINUATION_RIDER):], "continuation", False
        elif role == "user" and text.startswith(TASK_COMPLETION_RIDER):
            role, text, kind, eligible = "assistant", text[len(TASK_COMPLETION_RIDER):], "task", False
        elif role == "user" and text.startswith(INTERJECTION_FRAME):
            text, kind, eligible = _interjection_text(text), "interjection", True
        elif role == "assistant":
            text, kind, eligible = (_assistant_text(text) if capture_safe else text), "assistant", True
        elif role == "user":
            kind, eligible = "user", True
        else:
            kind, eligible = "tool", False
        records.append(VisibleRecord(
            id=f"r{len(records) + 1:04d}", order=order, role=role, text=text,
            source_kind=kind, capture_eligible=eligible,
        ))
    return records


def select_completed_turns(
    records: Iterable[VisibleRecord], *, whole_conversation: bool = False,
    turns: int = 3,
) -> tuple[VisibleRecord, ...]:
    """Return capture-eligible records from complete logical user turns only."""
    completed: list[list[VisibleRecord]] = []
    current: list[VisibleRecord] | None = None
    has_assistant = False
    for record in records:
        if not record.capture_eligible:
            continue
        if record.role == "user":
            if current is not None and has_assistant:
                completed.append(current)
            current = [record]
            has_assistant = False
        elif record.role == "assistant" and current is not None:
            current.append(record)
            has_assistant = True
    if current is not None and has_assistant:
        completed.append(current)
    selected = completed if whole_conversation else completed[-turns:]
    return tuple(record for turn in selected for record in turn)
