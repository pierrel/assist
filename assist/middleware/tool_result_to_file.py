"""Offload large tool results to a file so the small model greps them on demand.

Some tools return a lot of text — ``read_url`` (a full web page), ``execute`` (a
long build/test log).  Left inline, a few of those flood the agent's context (the
slow-zone the context work fights).  Truncating loses everything past the cap.
Instead, for an opt-in set of tools, this middleware writes the full result to
``/large_tool_results/<tool_call_id>`` (the prefix deepagents' built-in eviction
uses, and which its filesystem system prompt already tells the model to
``grep``/``read_file``) and replaces the tool result with a short PREVIEW + the
path + an explicit grep instruction.  Full content stays available; model-visible
context is bounded to ~1kB per call BY CONSTRUCTION.

Deliberately NOT deepagents' ``FilesystemMiddleware`` (which we DON'T subclass):
that is a fat 3-job middleware (system-prompt injection + message eviction in
``wrap_model_call``, large-result eviction in ``wrap_tool_call``, and registering
the 7 file tools) whose one behavior we want (``wrap_tool_call`` eviction) writes
RAW content and emits a *read_file/offset* preview — whereas we require
sanitize-before-write (raw ANSI in a log is the BadRequest-outage class) and an
eval-proven *grep* preview.  Reusing it means overriding it wholesale while
coupling to three private members + fighting its shared 80k-token threshold, so a
small standalone middleware is the isolated boundary (AGENTS.md rule 14).

Why middleware, not guidance: offload is I/O plumbing the model can't do itself —
you can't instruct it to shrink a ToolMessage the framework already appended.
That is the "guidance first" ethos's own exception.  The behavioral half — actually
grepping — is carried by the returned message + the prompt + the framework's
existing ``/large_tool_results/`` guidance.

Rides deepagents' own proven state-write path: ``StateBackend.write`` during
``wrap_tool_call`` queues into the current graph state, so a later ``read_file``/
``grep`` in the same turn reads it back.
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

_DEFAULT_FLOOR_CHARS = 4000   # below this, return inline — grepping a tiny file is worse
_PREVIEW_CHARS = 600


def _preview(content: str, style: str) -> str:
    """A short slice of the content for orientation.

    ``head`` (default) — the first ``_PREVIEW_CHARS`` chars; right for an article,
    whose lede is at the top.  ``head_tail`` — the first + last half; right for a
    command log, whose decisive line (the error, the exit summary) is at the TAIL
    while the head is progress noise.
    """
    if style == "head_tail" and len(content) > _PREVIEW_CHARS:
        half = _PREVIEW_CHARS // 2
        return f"{content[:half]}\n…[middle truncated]…\n{content[-half:]}"
    return content[:_PREVIEW_CHARS]


def _preview_message(msg: ToolMessage, content: str, tool_name: str,
                     source: str, path: str, style: str) -> ToolMessage:
    # Preview from the SANITIZED content (same string written to the file), so the
    # model-visible preview can't carry raw ANSI/control bytes regardless of where
    # OutputSanitizationMiddleware sits relative to this one.
    src = f" ({source})" if source else ""
    body = (
        f"Result of {tool_name}{src} — {len(content)} chars saved to {path}. This is "
        f"a PREVIEW ONLY:\n{_preview(content, style)}\n… [truncated] To find a specific "
        f"value — a number, date, name, quote, error, or line — DO NOT answer from this "
        f"preview: grep the file — grep(pattern=\"<keyword>\", path=\"{path}\") — then "
        f"read_file around the match."
    )
    return msg.model_copy(update={"content": body})


class ToolResultToFileMiddleware(AgentMiddleware):
    """Offload results of the opt-in ``tools`` larger than ``floor_chars`` to a file.

    Sync (``wrap_tool_call``) + async (``awrap_tool_call``) — subagents invoke through
    the async path.  Anything not a large ToolMessage from a listed tool passes through
    untouched; any failure falls back to the original result (never break the tool path).

    ``tools`` is a POSITIVE allowlist — a tool not listed is never touched (containment
    by construction).  ``preview_style`` is per-instantiation: ``head`` for read_url
    (proven), ``head_tail`` for execute (log tails hold the salient line).
    """

    name = "ToolResultToFileMiddleware"

    def __init__(self, backend, *, tools: frozenset[str] | set[str],
                 floor_chars: int = _DEFAULT_FLOOR_CHARS, preview_style: str = "head"):
        super().__init__()
        self.backend = backend
        self._tools = frozenset(tools)
        self._floor = floor_chars
        self._style = preview_style

    def _offload(self, request: ToolCallRequest, result):
        if request.tool_call.get("name") not in self._tools:
            return result
        if not isinstance(result, ToolMessage) or not isinstance(result.content, str):
            return result
        content = result.content
        if len(content) <= self._floor:
            return result
        # Sanitize before persisting so a later grep/read_file can't re-inject raw
        # ANSI/control bytes (the BadRequest-outage class) from the offloaded file.
        content = _sanitize(content)
        path = f"/large_tool_results/{sanitize_tool_call_id(result.tool_call_id)}"
        write = self.backend.write(path, content)
        if getattr(write, "error", None):
            logger.warning("ToolResultToFile: write to %s failed: %s", path, write.error)
            return result
        args = request.tool_call.get("args") or {}
        source = next((str(v) for v in args.values() if isinstance(v, str)), "")[:120]
        return _preview_message(result, content, request.tool_call.get("name", "tool"),
                                source, path, self._style)

    def _safe_offload(self, request, result):
        try:
            return self._offload(request, result)
        except Exception as e:  # never block the tool path on an offload bug
            logger.warning("ToolResultToFile: skipped due to %s: %s", type(e).__name__, e)
            return result

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        return self._safe_offload(request, handler(request))

    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        return self._safe_offload(request, await handler(request))
