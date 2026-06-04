"""Generic guard that validates file edits before they apply.

Intercepts ``edit_file`` tool calls and runs a list of pluggable
``EditValidator``s.  A validator that recognises the file (``applies``) and
finds a problem (``check`` returns a corrective message) causes the edit to
be REJECTED before it applies: the model gets the message back as a
``status="success"``, non-error tool result and can retry with a fix.

``status="success"`` + a non-error-prefixed message is deliberate — it keeps
the corrective redirect from registering as a loop error in
``LoopDetectionMiddleware`` (whose same-tool-same-error rule would otherwise
end the turn after a couple of rejections, before the model self-corrects).
A genuinely stuck model that repeats the *same* args still trips
LoopDetection's same-tool-same-args rule, and ``recursion_limit`` bounds the
worst case.

To add a check for any file type, subclass ``EditValidator`` and pass an
instance to ``FileEditGuardMiddleware([...])`` where the agent is built —
e.g. a JSON/YAML well-formedness check, a "don't delete this marker" check,
etc.  The first shipped validator is ``OrgHeadingInsertionValidator`` (see
``org_structure_guard.py``).

Place this middleware *before* ``LoopDetectionMiddleware``.
"""
import logging
from abc import ABC, abstractmethod
from typing import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

logger = logging.getLogger(__name__)


class EditValidator(ABC):
    """A check applied to an ``edit_file`` before it runs.

    Subclass and pass to ``FileEditGuardMiddleware`` to validate edits to a
    particular kind of file.  Both methods are pure (no I/O) — they see only
    the edit's ``old_string``/``new_string`` and the target path.
    """

    @property
    def name(self) -> str:
        """Short label used in logs.  Defaults to the class name."""
        return type(self).__name__

    @abstractmethod
    def applies(self, file_path: str) -> bool:
        """Does this validator apply to edits of ``file_path``?"""

    @abstractmethod
    def check(self, old_string: str, new_string: str) -> str | None:
        """Return a corrective message (shown to the model) if the edit is
        invalid and should be rejected; return ``None`` to allow it."""


def _is_edit_file(request: ToolCallRequest) -> bool:
    tool = getattr(request, "tool", None)
    name = getattr(tool, "name", None) or request.tool_call.get("name", "")
    return name == "edit_file"


def _reject(request: ToolCallRequest, message: str) -> ToolMessage:
    # status="success" + a message that must not start with an error prefix
    # (the validator's responsibility) so LoopDetection doesn't treat the
    # corrective redirect as a loop error.
    return ToolMessage(
        content=message,
        name="edit_file",
        tool_call_id=request.tool_call.get("id", ""),
        status="success",
    )


class FileEditGuardMiddleware(AgentMiddleware):
    """Run pluggable ``EditValidator``s on every ``edit_file`` call and
    reject (with a corrective redirect) an edit any validator flags.

    Place before ``LoopDetectionMiddleware`` so a retry sees the redirect.
    """

    def __init__(self, validators):
        super().__init__()
        self.validators = list(validators)
        self.tools = []

    def _guarded(self, request: ToolCallRequest) -> ToolMessage | None:
        if not _is_edit_file(request):
            return None
        # Tool-call args land under "args" or (some shapes) "arguments" —
        # match GitPushBlockerMiddleware so a differently-shaped edit_file
        # can't slip past validation.  A non-dict (e.g. raw JSON string) is
        # skipped rather than crashing the call.
        args = request.tool_call.get("args") or request.tool_call.get("arguments") or {}
        if not isinstance(args, dict):
            return None
        path = args.get("file_path") or args.get("path") or ""
        old = args.get("old_string")
        new = args.get("new_string")
        if not isinstance(path, str) or not isinstance(old, str) or not isinstance(new, str):
            return None
        for v in self.validators:
            if not v.applies(path):
                continue
            message = v.check(old, new)
            if message:
                logger.warning(
                    "FileEditGuard: %s rejected an edit to %s", v.name, path)
                return _reject(request, message)
        return None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        guarded = self._guarded(request)
        return guarded if guarded is not None else handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        guarded = self._guarded(request)
        return guarded if guarded is not None else await handler(request)
