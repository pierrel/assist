"""Read-only enforcer middleware.

Rejects mutating tool calls (write_file, edit_file, execute) before they
execute, returning an error ToolMessage that reminds the model the agent is
read-only. This enforces the read-only contract at the tool layer rather
than relying on prompt-only adherence.
"""
import logging
from typing import Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command


logger = logging.getLogger(__name__)


_MUTATING_TOOLS: frozenset[str] = frozenset({
    "write_file",
    "edit_file",
    "execute",
})

_REJECTION_MESSAGE = (
    "Error: this agent is read-only and cannot call '{tool_name}'. "
    "If the user asked you to write or modify something, do not attempt the "
    "write — instead surface the relevant file path, current contents, and "
    "any format conventions the caller will need, and explicitly note that "
    "you did not perform the write."
)


class ReadOnlyEnforcerMiddleware(AgentMiddleware):
    """Reject calls to mutating tools at the runtime boundary."""

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        tool_name = request.tool_call.get("name", "")
        if tool_name in _MUTATING_TOOLS:
            logger.warning(
                "ReadOnlyEnforcer rejected mutating tool call: %s", tool_name
            )
            return ToolMessage(
                content=_REJECTION_MESSAGE.format(tool_name=tool_name),
                tool_call_id=request.tool_call.get("id", ""),
                name=tool_name,
                status="error",
            )
        return handler(request)
