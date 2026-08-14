"""Put the stock Deep Agents base before the web-main Assist core.

Deep Agents deliberately builds ``USER + BASE``.  The prompt-rewrite experiment
keeps that framework behavior everywhere except the explicitly marked ordinary
web-main profile, where this middleware changes only that first static text
block to ``BASE + ASSIST_CORE``.  Later filesystem, skills, memory, and task
blocks are owned by their existing middleware and remain untouched.
"""
from __future__ import annotations

from typing import Callable

from deepagents.graph import BASE_AGENT_PROMPT
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage


class PromptCompositionMiddleware(AgentMiddleware):
    """Fail closed unless the first system text block is the expected static pair."""

    def __init__(self, assist_core: str) -> None:
        self._expected = f"{assist_core}\n\n{BASE_AGENT_PROMPT}"
        self._replacement = f"{BASE_AGENT_PROMPT}\n\n{assist_core}"

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse | AIMessage],
    ) -> ModelResponse | AIMessage:
        return handler(self._reordered(request))

    async def awrap_model_call(self, request: ModelRequest, handler):
        return await handler(self._reordered(request))

    def _reordered(self, request: ModelRequest) -> ModelRequest:
        system = request.system_message
        blocks = [dict(block) for block in system.content_blocks] if system else []
        if (not blocks or blocks[0].get("type") != "text"
                or blocks[0].get("text") != self._expected):
            raise RuntimeError(
                "PromptCompositionMiddleware expected the web-main static "
                "Assist core followed by the Deep Agents base prompt")
        first = {**blocks[0], "text": self._replacement}
        replacement = system.model_copy(update={"content": [first, *blocks[1:]]})
        return request.override(system_message=replacement)
