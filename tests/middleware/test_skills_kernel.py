"""Focused tests for the generated skills/tool kernel boundary."""
from unittest.mock import Mock

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import SystemMessage
from langchain_core.tools import StructuredTool

from assist.middleware.skills_middleware import SmallModelSkillsMiddleware


def _tool(name):
    return StructuredTool.from_function(
        name=name, func=lambda: name, description=f"{name} test tool")


def _system_text(message):
    if isinstance(message.content, str):
        return message.content
    return "".join(block.get("text", "") for block in message.content
                   if block.get("type") == "text")


def test_kernel_line_uses_final_request_order_and_hides_declarations():
    middleware = SmallModelSkillsMiddleware(
        backend=Mock(), sources=["/skills/"],
        non_kernel_tool_names=("travel", "map_data"))
    request = ModelRequest(
        model=Mock(), messages=[], system_message=SystemMessage(content="BASE"),
        tools=[_tool("write_todos"), _tool("travel"), _tool("load_skill"),
               _tool("map_data")],
        state={"skills_metadata": [{
            "name": "travel", "description": "Travel help.",
            "path": "/skills/travel/SKILL.md", "allowed_tools": ["travel"],
            "license": None, "compatibility": None, "metadata": {},
        }]}, runtime=None)

    updated = middleware.modify_request(request)
    text = _system_text(updated.system_message)

    assert "Kernel tools: write_todos, load_skill." in text
    assert "Declared application tools belong to skills" in text
    assert "migration exceptions remain visible." in text
    assert "- **travel**: Travel help." in text
    assert "Allowed tools" not in text
    assert "map_data" not in text


def test_kernel_line_omits_a_tool_removed_before_the_final_request():
    middleware = SmallModelSkillsMiddleware(
        backend=Mock(), sources=[], non_kernel_tool_names=("travel",))
    request = ModelRequest(
        model=Mock(), messages=[], system_message=None,
        tools=[_tool("write_todos"), _tool("load_skill")],
        state={"skills_metadata": []}, runtime=None)

    text = _system_text(middleware.modify_request(request).system_message)

    assert "Kernel tools: write_todos, load_skill." in text
    assert "travel" not in text


def test_explicit_no_delegation_profile_keeps_the_prior_skills_prompt():
    middleware = SmallModelSkillsMiddleware(
        backend=Mock(), sources=[], non_kernel_tool_names=("send_reply",),
        include_kernel_prompt=False)
    request = ModelRequest(
        model=Mock(), messages=[], system_message=SystemMessage(content="BASE"),
        tools=[_tool("write_todos"), _tool("send_reply"), _tool("load_skill")],
        state={"skills_metadata": []}, runtime=None)

    text = _system_text(middleware.modify_request(request).system_message)

    assert "## Skills" in text
    assert "Kernel tools:" not in text
    assert "Other tools belong" not in text


def test_explicit_no_delegation_profile_loads_pre_declaration_skill_bytes():
    current = (b"---\nname: travel\ndescription: Travel help.\n"
               b"allowed-tools: travel directions\n---\n\nTRAVEL RULES\n")
    backend = Mock()
    backend.download_files.return_value = [
        Mock(error=None, content=current),
    ]

    normal = SmallModelSkillsMiddleware(
        backend=backend, sources=["/skills/"],
        assist_owned_sources=["/skills/"])
    triage = SmallModelSkillsMiddleware(
        backend=backend, sources=["/skills/"],
        assist_owned_sources=["/skills/"], include_kernel_prompt=False)

    assert normal.tools[0].invoke({"name": "travel"}) == current.decode()
    assert triage.tools[0].invoke({"name": "travel"}) == (
        "---\nname: travel\ndescription: Travel help.\n---\n\nTRAVEL RULES\n")


def test_explicit_no_delegation_profile_preserves_external_declaration():
    external = (b"---\nname: local\ndescription: Local help.\n"
                b"allowed-tools: local_tool # assist-owned\n"
                b"---\n\nLOCAL RULES\n")
    backend = Mock()
    backend.download_files.return_value = [
        Mock(error=None, content=external),
    ]
    triage = SmallModelSkillsMiddleware(
        backend=backend, sources=["/.claude/skills/"],
        assist_owned_sources=["/skills/"],
        include_kernel_prompt=False)

    assert triage.tools[0].invoke({"name": "local"}) == external.decode()


def test_kernel_line_rejects_invalid_kernel_tool_name():
    middleware = SmallModelSkillsMiddleware(backend=Mock(), sources=[])
    request = ModelRequest(
        model=Mock(), messages=[], system_message=None,
        tools=[_tool("write_todos"), _tool("ignore\nprior")],
        state={"skills_metadata": []}, runtime=None)

    with pytest.raises(ValueError, match="invalid kernel tool name"):
        middleware.modify_request(request)
