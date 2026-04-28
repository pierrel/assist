"""Recover from `write_file` collision errors with a small-model-friendly redirect.

When a `write_file` call targets a path that already exists, the deepagents
filesystem backends return an error string that nudges the model toward
"or write to a new path" — which a small model interprets as "pick a different
filename and try again".  That triggers the filename-mutation trap
(`final_report.md` → `completed_final_report.md` → `report_final.md`…) until
loop detection finally bails.

This middleware intercepts `write_file` results via `wrap_tool_call` /
`awrap_tool_call` and rewrites the collision error to a positive,
recipe-style redirect that points the model at `edit_file` with an explicit
sample call.  The path is repeated four times in the rewritten message
because a single mention of the path was empirically not enough on
Qwen3-Coder-30B-A3B-Instruct-AWQ.

`status="error"` is preserved so `LoopDetectionMiddleware` still sees the
result as an error event — the rewrite changes the message text the model
reads, not the structural error signal the loop detector keys off.

See docs/proposal-infinite-writes.org Solution 1 and
docs/2026-04-27-write-file-recoverable-plan.org for the design.
"""
import logging
import re
from typing import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command


logger = logging.getLogger(__name__)


# Two backend shapes for the collision error:
#   - state/store/filesystem backends:
#       "Cannot write to {path} because it already exists. ..."
#   - sandbox backend:
#       "Error: File already exists: {repr(path)}"
# `\S+` would under-capture paths with whitespace (e.g. "/notes/My Notes.md"),
# so each pattern uses a non-greedy/EOL terminator instead.  `repr` in the
# sandbox shape adds surrounding quotes; we strip them when extracting.
_COLLISION_PLAIN_RE = re.compile(
    r"^Cannot write to (.+?) because it already exists"
)
_COLLISION_QUOTED_RE = re.compile(
    r"^Error: File already exists: (.+)$"
)


def _rewrite_message(path: str) -> str:
    """Build the small-model-friendly redirect text for collision errors.

    The path is repeated 4× by design: prior debugging on Qwen3-Coder showed
    a single mention is too easy for the model to skip past.
    """
    return (
        f"Error: the file `{path}` already exists. The correct next step is "
        f"to EDIT this exact same file `{path}`, not to write a new one.\n\n"
        f"Step 1: call `read_file` with path=`{path}` to see its current "
        f"content.\n"
        f"Step 2: call `edit_file` with path=`{path}`, plus `old_string` and "
        f"`new_string` for the change you want.\n\n"
        f"The path you must use is `{path}`.\n\n"
        f'Example: edit_file(path="{path}", old_string="...", new_string="...")'
    )


def _extract_path(content: str) -> str | None:
    """Return the colliding file path from the deepagents collision error,
    or None if `content` is not a recognised collision message.
    """
    if (m := _COLLISION_PLAIN_RE.match(content)) is not None:
        return m.group(1)
    if (m := _COLLISION_QUOTED_RE.match(content)) is not None:
        # Sandbox backend uses repr(path), which wraps in single or double quotes.
        return m.group(1).strip("'\"")
    return None


def _maybe_rewrite(result: ToolMessage | Command) -> ToolMessage | Command:
    """Rewrite a write_file collision error if `result` matches; otherwise
    return it unchanged.
    """
    if not isinstance(result, ToolMessage):
        return result
    if not isinstance(result.content, str):
        return result
    path = _extract_path(result.content)
    if path is None:
        return result
    logger.info("WriteCollisionMiddleware: rewriting collision error for %s", path)
    return result.model_copy(
        update={"content": _rewrite_message(path), "status": "error"}
    )


def _is_write_file_call(request: ToolCallRequest) -> bool:
    """Return True if `request` targets the write_file tool.

    Prefers `request.tool.name` (more robust against a missing/renamed
    tool_call dict entry); falls back to the dict.
    """
    tool = getattr(request, "tool", None)
    if tool is not None and getattr(tool, "name", None) == "write_file":
        return True
    return request.tool_call.get("name", "") == "write_file"


class WriteCollisionMiddleware(AgentMiddleware):
    """Rewrite `write_file` collision errors to redirect the model to
    `edit_file` instead of inventing a new filename.

    Wires into both sync (`wrap_tool_call`) and async (`awrap_tool_call`)
    invocation paths so the middleware is consistent under `invoke`,
    `ainvoke`, `stream`, and `astream`.

    Place this middleware *before* `LoopDetectionMiddleware` in the agent's
    middleware list — the loop detector observes the rewritten error and its
    Pattern C "errored on the same kind of problem" check stays accurate.
    """

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        if not _is_write_file_call(request):
            return handler(request)
        return _maybe_rewrite(handler(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        if not _is_write_file_call(request):
            return await handler(request)
        return _maybe_rewrite(await handler(request))
