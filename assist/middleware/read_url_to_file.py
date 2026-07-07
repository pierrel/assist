"""Offload large ``read_url`` results to a file so the agent greps them on demand.

``read_url`` returns a page's full extracted text.  A long article floods the
research subagent's context (many reads → the slow-zone / overflow the context
work fights).  Rather than truncate — which loses everything past the cap, so the
agent researches from page-tops only — this middleware writes the full text to
``/large_tool_results/<tool_call_id>`` (the same prefix deepagents' built-in
large-result eviction uses, and which its filesystem system prompt already tells
the model to ``grep``/``read_file``) and replaces the tool result with a short
PREVIEW + the path + a grep instruction.  So the full page stays available while
the model-visible context is bounded to ~1kB per read BY CONSTRUCTION.

Why middleware, not guidance (AGENTS.md "guidance first"): offload is I/O plumbing
the model can't do itself — you cannot instruct it to shrink a ToolMessage the
framework has already appended to its context.  That is the ethos's own exception
("middleware only when guidance provably can't").  The behavioral half — actually
grepping the file — is carried by the prompt (see ``sub_research.txt.j2``) + the
framework's existing ``/large_tool_results/`` grep guidance.

Rides the exact state-write path deepagents' own ``FilesystemMiddleware`` uses:
``StateBackend.write`` during ``wrap_tool_call`` queues into the current graph
state, so a later ``read_file``/``grep`` in the same turn reads it back.
"""
from __future__ import annotations

import logging
from typing import Callable

from deepagents.backends.utils import sanitize_tool_call_id
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from assist.middleware.output_sanitization import _sanitize

logger = logging.getLogger(__name__)

# Offload only above this floor.  A short page — or an ``Error fetching URL:``
# string — is smaller than what a grep-the-file instruction would send the model
# after, so returning it inline (as today) is strictly better than making the
# small model chase an all-but-empty file.  Comfortably below any real article,
# far below deepagents' own 80k-char eviction floor.
_OFFLOAD_FLOOR_CHARS = 4000
_PREVIEW_CHARS = 600
_ERROR_PREFIX = "Error fetching URL:"   # read_url's bare-string error shape (tools.py)


def _preview_message(msg: ToolMessage, content: str, url: str, path: str) -> ToolMessage:
    # Preview from the SANITIZED content (same string written to the file), so the
    # model-visible preview can't carry raw ANSI/control bytes regardless of where
    # OutputSanitizationMiddleware sits relative to this one.
    preview = content[:_PREVIEW_CHARS]
    full_len = len(content)
    body = (
        f"Fetched {url}. Full page text ({full_len} chars) saved to {path} — this is "
        f"a PREVIEW ONLY (first {_PREVIEW_CHARS} chars):\n{preview}\n… [truncated] "
        f"To find a specific fact, number, date, name, or quote, DO NOT answer from "
        f"this preview — grep the file: grep(pattern=\"<keyword>\", path=\"{path}\") "
        f"then read_file around the match."
    )
    return msg.model_copy(update={"content": body})


class ReadUrlToFileMiddleware(AgentMiddleware):
    """Offload ``read_url`` results larger than the floor to ``/large_tool_results/``.

    Sync (``wrap_tool_call``) + async (``awrap_tool_call``) — the research/fact-check
    subagents invoke through the async path.  Anything that isn't a large ``read_url``
    ``ToolMessage`` passes through untouched, and any failure falls back to the
    original result (never break the research path on an offload bug).
    """

    name = "ReadUrlToFileMiddleware"

    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    def _offload(self, request: ToolCallRequest, result):
        if request.tool_call.get("name") != "read_url":
            return result
        if not isinstance(result, ToolMessage) or not isinstance(result.content, str):
            return result
        content = result.content
        if content.startswith(_ERROR_PREFIX) or len(content) <= _OFFLOAD_FLOOR_CHARS:
            return result
        # Sanitize before persisting so a later grep/read_file can't re-inject raw
        # ANSI/control bytes (the BadRequest-outage class) from the offloaded file.
        content = _sanitize(content)
        path = f"/large_tool_results/{sanitize_tool_call_id(result.tool_call_id)}"
        write = self.backend.write(path, content)
        if getattr(write, "error", None):
            logger.warning("ReadUrlToFile: write to %s failed: %s", path, write.error)
            return result
        url = (request.tool_call.get("args") or {}).get("url", "the page")
        return _preview_message(result, content, url, path)

    def _safe_offload(self, request, result):
        try:
            return self._offload(request, result)
        except Exception as e:  # never block the tool path on an offload bug
            logger.warning("ReadUrlToFile: skipped due to %s: %s", type(e).__name__, e)
            return result

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        return self._safe_offload(request, handler(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ):
        return self._safe_offload(request, await handler(request))
