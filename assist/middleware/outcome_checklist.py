"""Prompt-only reminder to reconcile independent outcomes with a temporary todo list."""
from __future__ import annotations

from typing import Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage


OUTCOME_CHECKLIST_PROMPT = """## Independent outcomes

When one user turn has multiple independently valuable outcomes and you start
tool, skill, or asynchronous work on one, use a short todo list with one
outcome per item if that work could cause another to be missed. Mark each item
complete only when the outcome is actually done, then reconcile the list before
replying. The todo list tracks this turn; it does not replace any user-requested
file, schedule, or durable thread-memory record.
"""


class OutcomeChecklistMiddleware(AgentMiddleware):
    """Append the sealed outcome-reconciliation rider to a model request.

    Deep Agents has already installed its ordinary todo middleware before
    Assist middleware runs. This class only appends the generic rider beside
    that existing guidance. It owns no tool, state, or lifecycle behavior.
    """

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse | AIMessage],
    ) -> ModelResponse | AIMessage:
        return handler(self._with_rider(request))

    async def awrap_model_call(self, request: ModelRequest, handler):
        return await handler(self._with_rider(request))

    def _with_rider(self, request: ModelRequest) -> ModelRequest:
        system = request.system_message
        if system is None:
            raise RuntimeError(
                "OutcomeChecklistMiddleware requires the ordinary todo system prompt"
            )
        blocks = [dict(block) for block in system.content_blocks]
        if any(block.get("text") == OUTCOME_CHECKLIST_PROMPT for block in blocks):
            raise RuntimeError("OutcomeChecklistMiddleware received a duplicate rider")
        replacement = system.model_copy(update={"content": [
            *blocks,
            {"type": "text", "text": f"\n\n{OUTCOME_CHECKLIST_PROMPT}"},
        ]})
        return request.override(system_message=replacement)
