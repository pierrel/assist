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
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)


# Full CSI matcher: ESC `[` parameter-bytes (0x30-0x3f, includes `:<=>?` for
# 24-bit color and private-mode sequences) intermediate-bytes (0x20-0x2f)
# final-byte (0x40-0x7e).  Covers SGR (colors), cursor moves, erase, scroll,
# private-mode set/reset, and DEC special sequences — i.e. everything a
# terminal emits.  The narrower `\x1b\[[0-9;]*[mGKHF]` regex previously
# embedded in the deleted ContextAwareToolEvictionMiddleware missed
# `\x1b[2J` (clear screen), 24-bit color forms using `:`, and the full
# cursor-movement set.
_CSI_RE = re.compile(r"\x1b\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]")

# Plus a sweep for non-whitespace control chars that can still break
# JSON serialization on some endpoints (kept narrower than
# BadRequestRetry's set because we do NOT want to drop \r, \n, \t which
# are valid whitespace in tool output the agent reads back).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize(text: str) -> str:
    text = _CSI_RE.sub("", text)
    return _CONTROL_RE.sub("", text)


class OutputSanitizationMiddleware(AgentMiddleware):
    """Strip ANSI / control chars from ``ToolMessage`` content in-state.

    Runs after each tool call.  Compares before/after; if anything
    changed, replaces the message content so the sanitized version is
    what lands in ``state["messages"]`` (and therefore in any future
    checkpoint, summarization read, or next-turn prompt).
    """

    name = "OutputSanitizationMiddleware"

    def wrap_tool_call(self, request, handler):
        result = handler(request)
        try:
            messages = getattr(result, "messages", None)
            if not messages:
                return result
            mutated = False
            new_messages = []
            for msg in messages:
                if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
                    sanitized = _sanitize(msg.content)
                    if sanitized != msg.content:
                        new_msg = msg.model_copy(update={"content": sanitized})
                        new_messages.append(new_msg)
                        mutated = True
                        logger.debug(
                            "OutputSanitization: stripped %d bytes from %s",
                            len(msg.content) - len(sanitized),
                            msg.name or "tool",
                        )
                        continue
                new_messages.append(msg)
            if mutated:
                return result.model_copy(update={"messages": new_messages})
            return result
        except Exception as e:  # never block the tool path on sanitizer bugs
            logger.warning("OutputSanitization: skipped due to %s: %s",
                           type(e).__name__, e)
            return result
