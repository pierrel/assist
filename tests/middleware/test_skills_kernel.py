"""Focused tests for declarations and the deferred prompt-kernel boundary."""
from unittest.mock import Mock

from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import SystemMessage

from assist.middleware.skills_middleware import SmallModelSkillsMiddleware


def _system_text(message):
    if isinstance(message.content, str):
        return message.content
    return "".join(block.get("text", "") for block in message.content
                   if block.get("type") == "text")


def test_startup_catalog_hides_declarations_and_kernel():
    middleware = SmallModelSkillsMiddleware(
        backend=Mock(), sources=["/skills/"])
    request = ModelRequest(
        model=Mock(), messages=[], system_message=SystemMessage(content="BASE"),
        tools=[],
        state={"skills_metadata": [{
            "name": "travel", "description": "Travel help.",
            "path": "/skills/travel/SKILL.md", "allowed_tools": ["travel"],
            "license": None, "compatibility": None, "metadata": {},
        }]}, runtime=None)

    updated = middleware.modify_request(request)
    text = _system_text(updated.system_message)

    assert "- **travel**: Travel help." in text
    assert "Allowed tools" not in text
    assert "allowed-tools" not in text
    assert "Kernel tools:" not in text
