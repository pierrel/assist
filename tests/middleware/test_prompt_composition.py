"""Exact, metadata-preserving coverage for the web-main prompt reorder."""
import asyncio
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from assist.middleware.prompt_composition import PromptCompositionMiddleware


def _request(core: str, blocks: list[dict]) -> ModelRequest:
    return ModelRequest(
        model=MagicMock(), messages=[HumanMessage(content="hello")],
        system_message=SystemMessage(content_blocks=blocks),
    )


def test_reorders_only_first_static_block_and_preserves_later_blocks():
    core = "Assist core"
    middleware = PromptCompositionMiddleware(core)
    first = {"type": "text", "text": middleware._expected, "cache_control": {"type": "ephemeral"}}
    trailing = {"type": "text", "text": "filesystem guidance", "cache_control": {"type": "persistent"}}
    request = _request(core, [first, trailing])
    seen = []

    result = middleware.wrap_model_call(request, lambda changed: seen.append(changed) or "ok")

    assert result == "ok"
    blocks = seen[0].system_message.content_blocks
    assert blocks[0] == {**first, "text": middleware._replacement}
    assert blocks[1] == trailing
    assert request.system_message.content_blocks == [first, trailing]


@pytest.mark.parametrize("blocks", [
    [],
    [{"type": "text", "text": "wrong prefix"}],
    [{"type": "image", "base64": "ignored"}],
    [{"type": "text", "text": "prefix"}],
])
def test_rejects_any_non_exact_static_prefix(blocks):
    middleware = PromptCompositionMiddleware("Assist core")

    with pytest.raises(RuntimeError, match="expected the web-main static"):
        middleware.wrap_model_call(
            _request("Assist core", blocks), lambda request: "must not run")


def test_async_wrapper_uses_the_same_exact_reorder():
    middleware = PromptCompositionMiddleware("Assist core")
    request = _request("Assist core", [{"type": "text", "text": middleware._expected}])

    async def handler(changed):
        return changed.system_message.content_blocks[0]["text"]

    assert asyncio.run(middleware.awrap_model_call(request, handler)) == middleware._replacement


def test_normalizes_model_content_blocks_before_reordering():
    class TextBlock(BaseModel):
        type: str
        text: str
        cache_control: dict[str, str]

    middleware = PromptCompositionMiddleware("Assist core")
    block = TextBlock(
        type="text", text=middleware._expected,
        cache_control={"type": "ephemeral"},
    )
    request = _request("Assist core", [{
        "type": "text",
        "text": middleware._expected,
        "cache_control": {"type": "ephemeral"},
    }])
    with patch.object(SystemMessage, "content_blocks", new_callable=PropertyMock,
                      return_value=[block]):
        changed = middleware._reordered(request)

    assert changed.system_message.content_blocks == [{
        "type": "text",
        "text": middleware._replacement,
        "cache_control": {"type": "ephemeral"},
    }]
