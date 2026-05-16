"""Strip ANSI escape sequences from tool output before it enters agent state.

Some `execute` invocations emit colorized terminal output (pytest, pip,
make, etc.).  Those ANSI CSI sequences can:

1. Trigger BadRequestError from the OpenAI-compatible endpoint on the
   next model call (some servers reject the bytes as malformed UTF-8
   inside the JSON message body).
2. Survive across turns and accumulate in the conversation history
   that ``SummarizationMiddleware`` reads — burning tokens on bytes
   the model cannot meaningfully use.

History: this used to live inside ``ContextAwareToolEvictionMiddleware``
alongside the per-result eviction logic.  That middleware was deleted
on 2026-05-16 (see docs/2026-05-16-context-management-overhaul.org)
because eviction was redundant with deepagents 0.6.1's built-in
``FilesystemMiddleware`` + ``SummarizationMiddleware``.  Sanitization,
however, is NOT redundant — neither upstream middleware strips ANSI.
Pulling the regex into ``BadRequestRetryMiddleware`` was insufficient:
that middleware only rewrites the in-flight request body on retry and
never persists the sanitized ToolMessage back to ``state["messages"]``.
The next turn would see the same raw ANSI and pay the BadRequest +
retry cost again.

This middleware sanitizes ``ToolMessage`` content in the
``wrap_tool_call`` after-path so the sanitized version lands in state.
``BadRequestRetryMiddleware`` keeps its own (broader) sanitizer as a
defense-in-depth layer for anything that slips through (e.g., ANSI
embedded in ``AIMessage`` content, which this middleware does not
touch).
"""
from __future__ import annotations

import logging
import re
from typing import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

logger = logging.getLogger(__name__)


# Full CSI matcher per ECMA-48: ESC `[` parameter-bytes (0x30-0x3f, includes
# `:;<=>?` for 24-bit color and private-mode sequences) intermediate-bytes
# (0x20-0x2f) final-byte (0x40-0x7e).  Covers SGR (colors), cursor moves,
# erase, scroll, private-mode set/reset, and DEC special sequences — i.e.
# everything a terminal emits.  The narrower `\x1b\[[0-9;]*[mGKHF]` regex
# previously embedded in the deleted ContextAwareToolEvictionMiddleware
# missed `\x1b[2J` (clear screen), 24-bit color forms using `:`, and the
# full cursor-movement set.
_CSI_RE = re.compile(r"\x1b\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]")

# OSC (Operating System Command) matcher: ESC `]` payload, terminated by
# either BEL (0x07) or ST (`ESC \\` = 0x1b 0x5c).  Real-world emitters:
# `set terminal title` from shells, hyperlink escapes from modern terminal
# tools (file://, vscode-jupyter-tab-tag etc.), iTerm2 inline images.
# Without this match, OSC payload survives as plain text after CSI-only
# stripping, e.g. `]0;title\x07` would lose the `\x1b` to the control-char
# pass but leave `]0;title` as visible noise.
_OSC_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")

# Plus a sweep for non-whitespace control chars that can still break
# JSON serialization on some endpoints (kept narrower than
# BadRequestRetry's set because we do NOT want to drop \r, \n, \t which
# are valid whitespace in tool output the agent reads back).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize(text: str) -> str:
    text = _OSC_RE.sub("", text)
    text = _CSI_RE.sub("", text)
    return _CONTROL_RE.sub("", text)


def _sanitize_content(content):
    """Sanitize ToolMessage content of either str or list[str|dict] shape.

    Returns ``(new_content, changed)``.  When ``changed`` is False, the
    returned ``new_content`` is the same object identity as the input.
    """
    if isinstance(content, str):
        sanitized = _sanitize(content)
        return (sanitized, sanitized != content)
    if isinstance(content, list):
        new_parts = []
        changed = False
        for part in content:
            if isinstance(part, str):
                cleaned = _sanitize(part)
                if cleaned != part:
                    changed = True
                new_parts.append(cleaned)
            elif isinstance(part, dict) and "text" in part and isinstance(part["text"], str):
                cleaned = _sanitize(part["text"])
                if cleaned != part["text"]:
                    new_parts.append({**part, "text": cleaned})
                    changed = True
                else:
                    new_parts.append(part)
            else:
                # Non-text blocks (image, audio, etc.) pass through.
                new_parts.append(part)
        return (new_parts if changed else content, changed)
    # Unknown content shape (None / bytes / etc.) — pass through.
    return (content, False)


def _sanitize_tool_message(msg: ToolMessage) -> ToolMessage:
    """Return msg unchanged if no sanitization needed; else a new copy
    with sanitized content.  Handles both str and list-of-blocks content."""
    new_content, changed = _sanitize_content(msg.content)
    if not changed:
        return msg
    if isinstance(msg.content, str):
        stripped_chars = len(msg.content) - len(new_content)
    else:
        stripped_chars = -1  # list-of-blocks; not worth precise accounting
    logger.debug(
        "OutputSanitization: stripped %s chars from %s tool output",
        stripped_chars if stripped_chars >= 0 else "?",
        msg.name or "tool",
    )
    return msg.model_copy(update={"content": new_content})


def _sanitize_result(result):
    """Sanitize a handler return value of ``ToolMessage`` or ``Command``.

    Pulled out as a pure function so sync and async paths share logic.
    Returns the same object instance if nothing changed (cheap pass-through).
    """
    if isinstance(result, ToolMessage):
        return _sanitize_tool_message(result)
    if isinstance(result, Command):
        update = getattr(result, "update", None)
        if not isinstance(update, dict):
            return result
        messages = update.get("messages")
        if not messages:
            return result
        mutated = False
        new_messages = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                new_msg = _sanitize_tool_message(msg)
                if new_msg is not msg:
                    mutated = True
                new_messages.append(new_msg)
            else:
                new_messages.append(msg)
        if not mutated:
            return result
        new_update = {**update, "messages": new_messages}
        # Preserve control-flow fields (goto / graph / resume) — Command
        # is a dataclass with (graph, update, resume, goto); rebuilding
        # with only `update` silently drops the others, which would
        # break any tool that combines navigation with state updates.
        return Command(
            graph=result.graph,
            update=new_update,
            resume=result.resume,
            goto=result.goto,
        )
    # Unknown return type — pass through.
    return result


class OutputSanitizationMiddleware(AgentMiddleware):
    """Strip ANSI / control chars from ``ToolMessage`` content in-state.

    Wires into both sync (``wrap_tool_call``) and async
    (``awrap_tool_call``) paths so the middleware is consistent under
    ``invoke``, ``ainvoke``, ``stream``, and ``astream`` — research /
    context subagents invoke through the async path
    (``ReferencesCleanupRunnable.ainvoke`` exists for that reason).

    The handler returns either a bare ``ToolMessage`` (most common) or
    a ``Command(update={...})`` that carries one or more messages plus
    other state updates.  We handle both shapes — anything else passes
    through untouched.
    """

    name = "OutputSanitizationMiddleware"

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        result = handler(request)
        try:
            return _sanitize_result(result)
        except Exception as e:  # never block the tool path on sanitizer bugs
            logger.warning(
                "OutputSanitization: skipped due to %s: %s",
                type(e).__name__, e,
            )
            return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        result = await handler(request)
        try:
            return _sanitize_result(result)
        except Exception as e:
            logger.warning(
                "OutputSanitization: skipped due to %s: %s",
                type(e).__name__, e,
            )
            return result
