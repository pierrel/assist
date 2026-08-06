"""Focused tests for bundled skill tool disclosure."""
import asyncio
import hashlib
import json
import operator
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import get_args, get_type_hints
from unittest.mock import Mock, patch

from langchain.agents.middleware.types import ModelRequest
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.tools import tool
from langgraph.types import Command, Overwrite

from assist.middleware.skills_middleware import (
    LEGACY_SMALL_MODEL_SKILLS_PROMPT,
    MAX_SKILL_FILE_SIZE,
    SmallModelSkillsMiddleware,
    SmallModelSkillsState,
)


@tool
def kernel_tool(value: str) -> str:
    """A baseline-visible tool."""
    return value


@tool
def travel(origin: str, destination: str) -> str:
    """A bundled skill-owned tool."""
    return f"{origin}:{destination}"


def _skill(path="/skills/travel/SKILL.md"):
    return {
        "name": "travel", "description": "Travel help.",
        "path": path, "allowed_tools": ["travel"],
        "license": None, "compatibility": None, "metadata": {},
    }


def _request(middleware, state, tools=(kernel_tool, travel)):
    return ModelRequest(
        model=Mock(), messages=[], system_message=SystemMessage(content="BASE"),
        tools=list(tools), state=state, runtime=None)


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
        state={"skills_metadata": [_skill()]}, runtime=None)

    updated = middleware.modify_request(request)
    text = _system_text(updated.system_message)

    assert "- **travel**: Travel help." in text
    assert "Allowed tools" not in text
    assert "allowed-tools" not in text
    assert "Kernel tools:" not in text


def test_request_filters_only_winning_bundled_declarations_and_preserves_order():
    middleware = SmallModelSkillsMiddleware(
        backend=Mock(), sources=["/skills/", "/external/"],
        bundled_sources=["/skills/"])
    state = {"skills_metadata": [_skill()], "loaded_skill_tools": frozenset()}

    updated = middleware.modify_request(_request(middleware, state))

    assert [item.name for item in updated.tools] == ["kernel_tool"]
    assert "Tools available for this response: kernel_tool." in _system_text(
        updated.system_message)


def test_external_winner_stays_baseline_visible():
    middleware = SmallModelSkillsMiddleware(
        backend=Mock(), sources=["/skills/", "/external/"],
        bundled_sources=["/skills/"])
    state = {
        "skills_metadata": [_skill("/external/travel/SKILL.md")],
        "loaded_skill_tools": frozenset(),
    }

    updated = middleware.modify_request(_request(middleware, state))

    assert [item.name for item in updated.tools] == ["kernel_tool", "travel"]


def test_nested_external_source_is_not_claimed_by_bundled_parent():
    middleware = SmallModelSkillsMiddleware(
        backend=Mock(), sources=["/skills/", "/skills/external/"],
        bundled_sources=["/skills/"])
    state = {
        "skills_metadata": [_skill("/skills/external/travel/SKILL.md")],
        "loaded_skill_tools": frozenset(),
    }

    updated = middleware.modify_request(_request(middleware, state))

    assert [item.name for item in updated.tools] == ["kernel_tool", "travel"]


def test_no_bundled_source_keeps_legacy_prompt_and_loader_schema():
    middleware = SmallModelSkillsMiddleware(
        backend=Mock(), sources=["/external/"])

    assert len(LEGACY_SMALL_MODEL_SKILLS_PROMPT) == 1577
    assert hashlib.sha256(LEGACY_SMALL_MODEL_SKILLS_PROMPT.encode()).hexdigest() == \
        "93d4e684e87d4910f5c97de6919e3bb0798ba41f7239d07ff801d68d7f0dd918"
    assert middleware.system_prompt_template == LEGACY_SMALL_MODEL_SKILLS_PROMPT
    assert set(middleware.tools[0].args) == {"name"}


def test_before_agent_resets_activation_even_when_metadata_is_checkpointed():
    middleware = SmallModelSkillsMiddleware(
        backend=Mock(), sources=["/skills/"], bundled_sources=["/skills/"])

    update = middleware.before_agent(
        {"skills_metadata": [_skill()], "loaded_skill_tools": frozenset({"travel"})},
        SimpleNamespace(),
        {},
    )

    assert isinstance(update["loaded_skill_tools"], Overwrite)
    assert update["loaded_skill_tools"].value == frozenset()


def test_successful_load_returns_state_command_and_exact_closed_evidence():
    body = "---\nname: travel\ndescription: Travel help.\n---\n\n\x1b[31mRULES\x1b[0m"
    backend = SimpleNamespace(download_files=lambda _paths: [
        SimpleNamespace(error=None, content=body.encode("utf-8"))])
    middleware = SmallModelSkillsMiddleware(
        backend=backend, sources=["/skills/"], bundled_sources=["/skills/"])
    skill = _skill()
    state = {"skills_metadata": [skill], "loaded_skill_tools": frozenset()}
    runtime = SimpleNamespace(
        state=state, config={}, tool_call_id="load-1")

    result = middleware.tools[0].func("travel", runtime)

    assert isinstance(result, Command)
    assert result.update["loaded_skill_tools"] == frozenset({"travel"})
    message = result.update["messages"][0]
    assert message.tool_call_id == "load-1"
    assert "\x1b" not in message.content
    assert message.content.endswith("Newly available tools: travel.")
    assert set(message.artifact) == {
        "schema", "requested_name", "winner_fingerprint", "result_sha256"}
    assert message.artifact["requested_name"] == "travel"
    fingerprint_payload = {
        "allowed_tools": ["travel"],
        "skill_file_sha256": hashlib.sha256(
            body.encode("utf-8")).hexdigest(),
        "description": "Travel help.",
        "name": "travel",
    }
    assert message.artifact["winner_fingerprint"] == hashlib.sha256(
        json.dumps(fingerprint_payload, ensure_ascii=False, separators=(",", ":"),
                   sort_keys=True).encode("utf-8")).hexdigest()
    assert message.artifact["result_sha256"] == hashlib.sha256(
        message.content.encode("utf-8")).hexdigest()
    assert "/skills/" not in str(message.artifact)

    visible = middleware.modify_request(_request(
        middleware, {**state, "loaded_skill_tools": frozenset({"travel"})}))
    assert [item.name for item in visible.tools] == ["kernel_tool", "travel"]


def test_disclosure_intersects_declarations_with_this_agent_tools():
    body = "---\nname: travel\ndescription: Travel help.\n---\n\nRULES"
    backend = SimpleNamespace(download_files=lambda _paths: [
        SimpleNamespace(error=None, content=body.encode("utf-8"))])
    middleware = SmallModelSkillsMiddleware(
        backend=backend, sources=["/skills/"], bundled_sources=["/skills/"],
        registered_tools={"travel"})
    skill = {**_skill(), "allowed_tools": ["travel", "absent_tool"]}
    runtime = SimpleNamespace(
        state={"skills_metadata": [skill], "loaded_skill_tools": frozenset()},
        config={}, tool_call_id="load-1")

    result = middleware.tools[0].func("travel", runtime)

    assert result.update["loaded_skill_tools"] == frozenset({"travel"})
    assert result.update["messages"][0].content.endswith(
        "Newly available tools: travel.")


def test_parallel_load_updates_have_union_reducer():
    body = b"---\nname: skill\ndescription: Help.\n---\n\nRULES"
    backend = SimpleNamespace(download_files=lambda _paths: [
        SimpleNamespace(error=None, content=body)])
    middleware = SmallModelSkillsMiddleware(
        backend=backend, sources=["/skills/"], bundled_sources=["/skills/"])
    skills = [
        {**_skill("/skills/one/SKILL.md"), "name": "one",
         "allowed_tools": ["travel"]},
        {**_skill("/skills/two/SKILL.md"), "name": "two",
         "allowed_tools": ["kernel_tool"]},
    ]
    state = {"skills_metadata": skills, "loaded_skill_tools": frozenset()}
    updates = [
        middleware.tools[0].func(
            name, SimpleNamespace(state=state, config={}, tool_call_id=name)
        ).update["loaded_skill_tools"]
        for name in ("one", "two")
    ]
    outer = get_type_hints(
        SmallModelSkillsState, include_extras=True)["loaded_skill_tools"]
    annotated = get_args(outer)[0]

    assert operator.or_ in get_args(annotated)[1:]
    assert operator.or_(*updates) == frozenset({"travel", "kernel_tool"})


def test_failed_load_has_no_state_or_artifact():
    middleware = SmallModelSkillsMiddleware(
        backend=Mock(), sources=["/skills/"], bundled_sources=["/skills/"])
    runtime = SimpleNamespace(
        state={"skills_metadata": [], "loaded_skill_tools": frozenset()},
        config={}, tool_call_id="load-1")

    result = middleware.tools[0].func("fabricated", runtime)

    assert isinstance(result, str)
    assert "could not be loaded" in result


def test_invalid_skill_bytes_do_not_disclose_tools():
    for content, reason in (
        (b"x" * (MAX_SKILL_FILE_SIZE + 1), "too large"),
        (b"\xff", "not UTF-8"),
    ):
        backend = SimpleNamespace(download_files=lambda _paths, value=content: [
            SimpleNamespace(error=None, content=value)])
        middleware = SmallModelSkillsMiddleware(
            backend=backend, sources=["/skills/"],
            bundled_sources=["/skills/"])
        runtime = SimpleNamespace(
            state={"skills_metadata": [_skill()],
                   "loaded_skill_tools": frozenset()},
            config={}, tool_call_id="load-1")

        result = middleware.tools[0].func("travel", runtime)

        assert isinstance(result, str)
        assert reason in result


def test_backend_error_details_are_not_returned_to_the_model():
    backend = SimpleNamespace(download_files=lambda _paths: [
        SimpleNamespace(error="failed reading /home/private/SKILL.md", content=None)])
    middleware = SmallModelSkillsMiddleware(
        backend=backend, sources=["/skills/"], bundled_sources=["/skills/"])
    runtime = SimpleNamespace(
        state={"skills_metadata": [_skill()], "loaded_skill_tools": frozenset()},
        config={}, tool_call_id="load-1")

    result = middleware.tools[0].func("travel", runtime)

    assert isinstance(result, str)
    assert "backend error" in result
    assert "/home/private" not in result


def test_prose_claim_does_not_disclose_tools_and_async_reset_matches_sync():
    middleware = SmallModelSkillsMiddleware(
        backend=Mock(), sources=["/skills/"], bundled_sources=["/skills/"])
    state = {
        "messages": [{"role": "user", "content": "I loaded travel."}],
        "skills_metadata": [_skill()],
        "loaded_skill_tools": frozenset(),
    }

    updated = middleware.modify_request(_request(middleware, state))
    reset = asyncio.run(middleware.abefore_agent(
        {**state, "loaded_skill_tools": frozenset({"travel"})},
        SimpleNamespace(), {}))

    assert [item.name for item in updated.tools] == ["kernel_tool"]
    assert isinstance(reset["loaded_skill_tools"], Overwrite)
    assert reset["loaded_skill_tools"].value == frozenset()


def test_hidden_execution_is_rejected_in_sync_and_async_paths():
    middleware = SmallModelSkillsMiddleware(
        backend=Mock(), sources=["/skills/"], bundled_sources=["/skills/"])
    state = {"skills_metadata": [_skill()], "loaded_skill_tools": frozenset()}
    request = ToolCallRequest(
        tool_call={"name": "travel", "args": {}, "id": "travel-1"},
        tool=travel, state=state, runtime=None)
    sync_called = False

    def sync_handler(_request):
        nonlocal sync_called
        sync_called = True
        return ToolMessage(content="ran", tool_call_id="travel-1")

    sync_result = middleware.wrap_tool_call(request, sync_handler)

    async_called = False

    async def async_handler(_request):
        nonlocal async_called
        async_called = True
        return ToolMessage(content="ran", tool_call_id="travel-1")

    async_result = asyncio.run(middleware.awrap_tool_call(request, async_handler))

    assert not sync_called and not async_called
    assert sync_result.status == "error"
    assert async_result.status == "error"


def test_after_model_pairs_every_call_and_jumps_past_hitl():
    middleware = SmallModelSkillsMiddleware(
        backend=Mock(), sources=["/skills/"], bundled_sources=["/skills/"])
    message = AIMessage(content="", tool_calls=[
        {"name": "travel", "args": {}, "id": "hidden-1"},
        {"name": "kernel_tool", "args": {}, "id": "visible-1"},
    ])
    state = {
        "messages": [message], "skills_metadata": [_skill()],
        "loaded_skill_tools": frozenset(),
    }

    update = middleware.after_model(state, None)

    assert [call["name"] for call in message.tool_calls] == [
        "travel", "kernel_tool"]
    assert len(update["messages"]) == 3
    assert update["messages"][1].name == "travel"
    assert update["messages"][1].status == "error"
    assert update["messages"][2].name == "kernel_tool"
    assert update["messages"][2].status == "error"
    assert update["jump_to"] == "model"


class _RecordingModel(FakeMessagesListChatModel):
    bound_tools: list[list[str]] = []

    def bind_tools(self, tools, **_kwargs):
        self.bound_tools.append([
            item.name if hasattr(item, "name") else item.get("name", "")
            for item in tools
        ])
        return self


def test_compiled_graph_discloses_after_load_then_resets_next_invocation():
    from assist.agent import create_agent
    from assist.spec import AgentSpec

    @tool("travel")
    def fake_travel(origin: str, destination: str) -> str:
        """Return a deterministic route."""
        return f"route:{origin}:{destination}"

    model = _RecordingModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "load_skill", "args": {"name": "travel"}, "id": "load-1"}]),
        AIMessage(content="", tool_calls=[{
            "name": "travel", "args": {"origin": "a", "destination": "b"},
            "id": "travel-1"}]),
        AIMessage(content="first done"),
        AIMessage(content="", tool_calls=[{
            "name": "travel", "args": {"origin": "c", "destination": "d"},
            "id": "travel-2"}]),
        AIMessage(content="second done"),
    ])
    model.bound_tools = []

    with patch("assist.agent.travel", fake_travel), tempfile.TemporaryDirectory() as wd:
        agent = create_agent(
            model, wd, spec=AgentSpec(async_subagent_tools=()))
        config = {"configurable": {"thread_id": "skill-reset"}}
        first = agent.invoke({"messages": [{"role": "user", "content": "route"}]}, config)
        second = agent.invoke({"messages": [{"role": "user", "content": "again"}]}, config)

    assert "travel" not in model.bound_tools[0]
    assert "travel" in model.bound_tools[1]
    assert any(message.name == "travel" and message.status == "success"
               for message in first["messages"] if isinstance(message, ToolMessage))
    assert "travel" not in model.bound_tools[3]
    hidden = [message for message in second["messages"]
              if isinstance(message, ToolMessage)
              and message.tool_call_id == "travel-2"]
    assert len(hidden) == 1 and hidden[0].status == "error"
    assert second["messages"][-1].content == "second done"


def test_compiled_graph_rejects_same_response_sibling_before_hitl():
    from assist.agent import create_agent
    from assist.backends import create_bundled_skills_backend
    from assist.spec import AgentSpec

    sent = []

    @tool
    def send_email(to: str) -> str:
        """Send an email."""
        sent.append(to)
        return "sent"

    model = _RecordingModel(responses=[
        AIMessage(content="", tool_calls=[
            {"name": "load_skill", "args": {"name": "send-email"}, "id": "load-1"},
            {"name": "send_email", "args": {"to": "a@example.com"}, "id": "send-1"},
        ]),
        AIMessage(content="done"),
    ])
    model.bound_tools = []

    with tempfile.TemporaryDirectory() as skill_root, tempfile.TemporaryDirectory() as wd:
        skill_dir = Path(skill_root) / "send-email"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: send-email\ndescription: Send email.\n"
            "allowed-tools: send_email\n---\n\nUse send_email.\n",
            encoding="utf-8",
        )
        agent = create_agent(
            model,
            wd,
            spec=AgentSpec(
                tools=(send_email,),
                async_subagent_tools=(),
                skill_sources={
                    "/mail-skill/": create_bundled_skills_backend(skill_root)},
                interrupt_on={"send_email": True},
            ),
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "email"}]},
            {"configurable": {"thread_id": "skill-sibling"}},
        )

    assert sent == []
    rejected = [message for message in result["messages"]
                if isinstance(message, ToolMessage)
                and message.tool_call_id == "send-1"]
    assert len(rejected) == 1 and rejected[0].status == "error"
    paired = [message.tool_call_id for message in result["messages"]
              if isinstance(message, ToolMessage)
              and message.tool_call_id in {"load-1", "send-1"}]
    assert paired == ["load-1", "send-1"]
    assert not result.get("__interrupt__")
