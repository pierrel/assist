"""Propose-only enforcer middleware.

The event-triage turn runs on UNTRUSTED input (an inbound message anyone can send),
so it must be unable to cause any effect — it may only *propose* a reply for the user
to approve. Passing a small tool list does NOT achieve this: ``create_deep_agent``
force-installs the filesystem tools (and ``execute`` under a sandbox, and ``task`` when a
sub-agent is registered) via protected middleware, and the small model ignores
"propose, don't act" prose. So the gate is enforced at the tool-call boundary here:
DENY BY DEFAULT — reject every tool except an explicit allowlist, returning an error
ToolMessage. "Can't act" is then a property of this middleware, unit-testable.

Contrast with :mod:`assist.middleware.read_only_enforcer`, which is a *deny-list* (three
mutating tools). A deny-list is unsafe for the untrusted-input gate because it silently
admits any effectful tool not on the list (``task``, ``search_internet``, ``read_url``,
future built-ins); an allowlist can only ever admit what it names.
"""
import logging
from typing import Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command


logger = logging.getLogger(__name__)


# The only tools an event-triage turn may call: propose a reply, and read-only context.
# Everything else (write_file, edit_file, execute, task, write_todos, search_internet,
# read_url, …) is rejected — deny by default.
_ALLOWED_TOOLS: frozenset[str] = frozenset({
    "propose_reply",
    "read_file",
    "ls",
    "glob",
    "grep",
})

_REJECTION_MESSAGE = (
    "Error: this is a message-triage turn and cannot call '{tool_name}'. "
    "You may only read context and call propose_reply(draft) to propose a reply for the "
    "user to approve — you cannot send, write, run, or delegate anything. Decide what to "
    "do per the rules, then either propose a reply or state that no action is needed."
)


class ProposeOnlyEnforcerMiddleware(AgentMiddleware):
    """Reject every tool call except the propose/read allowlist, at the runtime boundary."""

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        tool_name = request.tool_call.get("name", "")
        if tool_name not in _ALLOWED_TOOLS:
            logger.warning(
                "ProposeOnlyEnforcer rejected non-allowlisted tool call: %s", tool_name
            )
            return ToolMessage(
                content=_REJECTION_MESSAGE.format(tool_name=tool_name),
                tool_call_id=request.tool_call.get("id", ""),
                name=tool_name,
                status="error",
            )
        return handler(request)
