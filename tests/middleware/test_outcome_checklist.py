"""Unit coverage for the prompt-only independent-outcome rider."""
import asyncio
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import HumanMessage, SystemMessage

from assist.middleware.outcome_checklist import (
    OUTCOME_CHECKLIST_PROMPT,
    OutcomeChecklistMiddleware,
)


def _request(blocks):
    return ModelRequest(
        model=MagicMock(), messages=[HumanMessage(content="hello")],
        system_message=SystemMessage(content_blocks=blocks),
    )


def test_appends_the_rider_without_altering_prior_prompt_blocks():
    middleware = OutcomeChecklistMiddleware()
    prior = {"type": "text", "text": "framework todo guidance"}
    seen = []

    middleware.wrap_model_call(
        _request([prior]), lambda changed: seen.append(changed) or "ok")

    assert seen[0].system_message.content_blocks == [
        prior,
        {"type": "text", "text": f"\n\n{OUTCOME_CHECKLIST_PROMPT}"},
    ]


def test_rejects_missing_or_duplicate_system_rider_context():
    middleware = OutcomeChecklistMiddleware()
    no_system = ModelRequest(model=MagicMock(), messages=[HumanMessage(content="hello")])
    with pytest.raises(RuntimeError, match="ordinary todo"):
        middleware.wrap_model_call(no_system, lambda _request: "unreachable")
    with pytest.raises(RuntimeError, match="duplicate"):
        middleware.wrap_model_call(
            _request([{"type": "text", "text": OUTCOME_CHECKLIST_PROMPT}]),
            lambda _request: "unreachable",
        )


def test_async_wrapper_has_the_same_prompt_effect():
    middleware = OutcomeChecklistMiddleware()

    async def handler(changed):
        return changed.system_message.content_blocks[-1]["text"]

    assert asyncio.run(middleware.awrap_model_call(
        _request([{"type": "text", "text": "framework todo guidance"}]), handler,
    )) == f"\n\n{OUTCOME_CHECKLIST_PROMPT}"
