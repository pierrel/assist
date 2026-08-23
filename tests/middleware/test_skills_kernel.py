"""Focused tests for bundled skill tool disclosure."""
import asyncio
import hashlib
import json
import operator
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import get_args, get_type_hints
from unittest.mock import AsyncMock, Mock, patch

from pydantic import PrivateAttr
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware.types import ModelRequest
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
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


@tool
def glob(pattern: str) -> str:
    """A framework-provided local discovery tool."""
    return pattern


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


def test_external_source_is_gated_when_disclosure_is_enabled():
    middleware = SmallModelSkillsMiddleware(
        backend=Mock(), sources=["/skills/", "/external/"],
        bundled_sources=["/skills/"], gated_sources=["/skills/", "/external/"])
    state = {
        "skills_metadata": [_skill("/external/travel/SKILL.md")],
        "loaded_skill_tools": frozenset(),
    }

    updated = middleware.modify_request(_request(middleware, state))

    assert [item.name for item in updated.tools] == ["kernel_tool"]


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


def test_legacy_checkpoint_without_activation_does_not_disclose_tools():
    middleware = SmallModelSkillsMiddleware(
        backend=SimpleNamespace(), sources=["/skills/"],
        bundled_sources=["/skills/"])

    with patch.object(middleware, "_catalog_snapshot",
                      return_value=([_skill()], [], "current")):
        update = middleware.before_agent(
            {"skills_metadata": [_skill()], "skills_catalog_fingerprint": "current",
             "loaded_skill_tools": frozenset({"travel"})},
            SimpleNamespace(),
            {},
        )

    assert isinstance(update["loaded_skill_tools"], Overwrite)
    assert update["loaded_skill_tools"].value == frozenset()


def test_activation_must_match_the_current_skill_catalog_and_schema():
    middleware = SmallModelSkillsMiddleware(
        backend=Mock(), sources=["/skills/"], bundled_sources=["/skills/"],
        tool_definitions=(travel,))
    valid = {
        "travel": {
            "schema_fingerprint": middleware._schema_fingerprint({"travel"}),
            "tools": frozenset({"travel"}),
        },
    }
    base_state = {
        "skills_catalog_fingerprint": "current",
        "active_skills": valid,
        "historical_gated_tools": frozenset({"travel"}),
    }

    retained = middleware._activation_update(base_state, [_skill()], "current")
    assert retained["loaded_skill_tools"].value == frozenset({"travel"})

    forged = middleware._activation_update(
        {**base_state, "active_skills": {"forged": valid["travel"]}},
        [_skill()], "current")
    assert forged["active_skills"].value == {}
    assert forged["loaded_skill_tools"].value == frozenset()

    catalog_changed = middleware._activation_update(
        base_state, [_skill()], "new-catalog")
    assert catalog_changed["active_skills"].value == {}

    malformed = middleware._activation_update(
        {**base_state, "active_skills": {"travel": {}}}, [_skill()], "current")
    assert malformed["active_skills"].value == {}

    @tool("travel")
    def changed_travel(origin: str, destination: str, mode: str) -> str:
        """Route with an additional required transport mode."""
        return f"{mode}:{origin}:{destination}"

    middleware._tool_definitions["travel"] = changed_travel
    schema_changed = middleware._activation_update(
        base_state, [_skill()], "current")
    assert schema_changed["active_skills"].value == {}
    assert schema_changed["loaded_skill_tools"].value == frozenset()


def _write_skill(root, source, name, body="Follow the current procedure."):
    skill = root / source.strip("/") / name / "SKILL.md"
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text(
        f"---\nname: {name}\ndescription: {name} help.\n---\n\n{body}\n",
        encoding="utf-8",
    )


def test_existing_checkpoint_refreshes_changed_bundled_and_domain_catalog(tmp_path):
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    initial_sources = ["/skills/", "/render-skill/"]
    _write_skill(tmp_path, "/skills/", "travel")
    middleware = SmallModelSkillsMiddleware(
        backend=backend, sources=initial_sources,
        bundled_sources=["/skills/", "/render-skill/"],
        gated_sources=initial_sources,
    )
    initial = middleware.before_agent({}, SimpleNamespace(), {})
    checkpoint = {
        "skills_metadata": initial["skills_metadata"],
        "skills_catalog_fingerprint": initial["skills_catalog_fingerprint"],
        "loaded_skill_tools": frozenset({"travel"}),
    }

    _write_skill(tmp_path, "/render-skill/", "render", "Render a file.")
    _write_skill(tmp_path, "/.claude/skills/", "domain-notes", "Read notes.")
    sources = [*initial_sources, "/.claude/skills/"]
    middleware = SmallModelSkillsMiddleware(
        backend=backend, sources=sources,
        bundled_sources=["/skills/", "/render-skill/"],
        gated_sources=sources,
    )

    refreshed = middleware.before_agent(checkpoint, SimpleNamespace(), {})

    assert {skill["name"] for skill in refreshed["skills_metadata"]} == {
        "travel", "render", "domain-notes"}
    assert refreshed["skills_catalog_fingerprint"] != \
        checkpoint["skills_catalog_fingerprint"]
    assert isinstance(refreshed["loaded_skill_tools"], Overwrite)
    assert refreshed["loaded_skill_tools"].value == frozenset()
    state = {**checkpoint, **refreshed, "loaded_skill_tools": frozenset()}
    for name in ("render", "domain-notes"):
        result = middleware.tools[0].func(
            name, SimpleNamespace(state=state, config={}, tool_call_id=name))
        assert isinstance(result, Command)
        assert name in result.update["messages"][0].content

    unchanged = middleware.before_agent(state, SimpleNamespace(), {})

    assert "skills_metadata" not in unchanged
    assert isinstance(unchanged["loaded_skill_tools"], Overwrite)

    _write_skill(tmp_path, "/render-skill/", "render", "Render the latest file.")
    body_refresh = middleware.before_agent(state, SimpleNamespace(), {})

    assert body_refresh["skills_metadata"] == state["skills_metadata"]
    assert body_refresh["skills_catalog_fingerprint"] != \
        state["skills_catalog_fingerprint"]

    _write_skill(tmp_path, "/render-skill/", "render", "Render only the final file.")
    async_refresh = asyncio.run(middleware.abefore_agent(
        {**state, **body_refresh}, SimpleNamespace(), {}))

    assert async_refresh["skills_catalog_fingerprint"] != \
        body_refresh["skills_catalog_fingerprint"]


def test_existing_checkpoint_discovers_the_web_render_skill(tmp_path):
    from assist.agent import map_data
    from assist.backends import SKILLS_ROUTE, create_composite_backend
    from assist.thread_manager import _RENDER_SKILL_ROUTE, _web_skill_sources

    backend = create_composite_backend(
        fs_root=str(tmp_path), extra_routes=_web_skill_sources())
    old = SmallModelSkillsMiddleware(
        backend=backend, sources=[SKILLS_ROUTE],
        bundled_sources=[SKILLS_ROUTE])
    initial = old.before_agent({}, SimpleNamespace(), {})
    current = SmallModelSkillsMiddleware(
        backend=backend, sources=[SKILLS_ROUTE, _RENDER_SKILL_ROUTE],
        bundled_sources=[SKILLS_ROUTE, _RENDER_SKILL_ROUTE],
        tool_definitions=(map_data,))

    refreshed = current.before_agent({
        "skills_metadata": initial["skills_metadata"],
        "skills_catalog_fingerprint": initial["skills_catalog_fingerprint"],
    }, SimpleNamespace(), {})
    state = {**refreshed, "loaded_skill_tools": frozenset()}
    result = current.tools[0].func(
        "render", SimpleNamespace(state=state, config={}, tool_call_id="render"))

    assert "render" in {skill["name"] for skill in refreshed["skills_metadata"]}
    assert isinstance(result, Command)
    assert "render" in result.update["messages"][0].content.lower()


def test_catalog_fingerprint_changes_when_a_source_becomes_unavailable():
    available = SmallModelSkillsMiddleware._snapshot_result([
        ("/skills/", [_skill()], None, [])])
    unavailable = SmallModelSkillsMiddleware._snapshot_result([
        ("/skills/", [_skill()], "source unavailable", [])])

    assert available[2] != unavailable[2]


def test_catalog_refresh_tolerates_a_short_skill_download_batch():
    middleware = SmallModelSkillsMiddleware(
        backend=Mock(), sources=["/skills/"], bundled_sources=["/skills/"])

    metadata, error, entries, records = middleware._catalog_from_responses(
        "/skills/", [{"path": "/skills/travel", "is_dir": True}], [])

    assert metadata == []
    assert error == "Cannot load skills from '/skills/': " + \
        "skill download response count mismatch"
    assert entries == [{
        "path": "/skills/travel/SKILL.md",
        "content_sha256": None,
        "error": True,
    }]
    assert records == []


def test_rebuilt_graph_refreshes_a_checkpointed_domain_catalog(tmp_path):
    from assist.agent import create_agent
    from assist.spec import AgentSpec
    from langgraph.checkpoint.memory import InMemorySaver

    model = _RecordingModel(responses=[
        AIMessage(content="before"),
        AIMessage(content="", tool_calls=[{
            "name": "load_skill", "args": {"name": "domain-notes"},
            "id": "load-domain-notes",
        }]),
        AIMessage(content="after"),
    ])
    checkpointer = InMemorySaver()
    config = {"configurable": {"thread_id": "catalog-refresh"}}
    spec = AgentSpec(async_subagent_tools=())

    initial = create_agent(model, str(tmp_path), checkpointer=checkpointer, spec=spec)
    initial.invoke({"messages": [{"role": "user", "content": "hello"}]}, config)
    _write_skill(tmp_path, "/.claude/skills/", "domain-notes", "Read notes.")
    rebuilt = create_agent(model, str(tmp_path), checkpointer=checkpointer, spec=spec)

    result = rebuilt.invoke(
        {"messages": [{"role": "user", "content": "use my notes"}]}, config)

    loaded = [message for message in result["messages"]
              if isinstance(message, ToolMessage)
              and message.tool_call_id == "load-domain-notes"]
    assert len(loaded) == 1
    assert loaded[0].status == "success"
    assert "domain-notes" in loaded[0].content


def test_successful_load_returns_state_command_and_exact_closed_evidence():
    body = "---\nname: travel\ndescription: Travel help.\n---\n\n\x1b[31mRULES\x1b[0m"
    backend = SimpleNamespace(download_files=lambda _paths: [
        SimpleNamespace(error=None, content=body.encode("utf-8"))])
    middleware = SmallModelSkillsMiddleware(
        backend=backend, sources=["/skills/"], bundled_sources=["/skills/"],
        tool_definitions=(travel,))
    skill = _skill()
    state = {"skills_metadata": [skill], "loaded_skill_tools": frozenset()}
    runtime = SimpleNamespace(
        state=state, config={}, tool_call_id="load-1")

    result = middleware.tools[0].func("travel", runtime)

    assert isinstance(result, Command)
    assert result.update["loaded_skill_tools"] == frozenset({"travel"})
    assert result.update["active_skills"] == {
        "travel": {
            "schema_fingerprint": middleware._schema_fingerprint({"travel"}),
            "tools": frozenset({"travel"}),
        },
    }
    message = result.update["messages"][0]
    assert message.tool_call_id == "load-1"
    assert "\x1b" not in message.content
    assert "## Tool contracts" in message.content
    assert '"name": "travel"' in message.content
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


def test_retained_active_tool_uses_a_compact_native_schema_without_replay():
    middleware = SmallModelSkillsMiddleware(
        backend=Mock(), sources=["/skills/"], bundled_sources=["/skills/"],
        tool_definitions=(travel,))
    state = {
        "skills_metadata": [_skill()],
        "historical_gated_tools": frozenset({"travel"}),
        "loaded_skill_tools": frozenset({"travel"}),
        "active_skills": {
            "travel": {
                "schema_fingerprint": middleware._schema_fingerprint({"travel"}),
                "tools": frozenset({"travel"}),
            },
        },
    }
    original = ToolMessage(
        content="full prior skill\n\n## Tool contracts\nvery long schema",
        name="load_skill", tool_call_id="load-1")
    request = ModelRequest(
        model=Mock(), messages=[original], system_message=SystemMessage(content="BASE"),
        tools=[travel], state=state, runtime=None)

    updated = middleware.modify_request(request)

    assert updated.messages == [original]
    assert len(updated.tools) == 1
    schema = updated.tools[0]
    assert schema["function"]["name"] == "travel"
    assert "description" not in schema["function"]
    assert schema["function"]["parameters"]["required"] == ["origin", "destination"]


def test_runtime_injected_callable_has_a_load_contract_and_compact_schema():
    def send_email(to: str, subject: str, runtime: ToolRuntime) -> str:
        """Send a message after approval."""
        return f"{to}:{subject}:{runtime.tool_call_id}"

    middleware = SmallModelSkillsMiddleware(
        backend=Mock(), sources=["/skills/"], bundled_sources=["/skills/"],
        tool_definitions=(send_email,))

    contract = middleware._tool_contract({"send_email"})
    compact = middleware._compact_schema(send_email)

    assert contract is not None
    assert '"name": "send_email"' in contract
    assert '"runtime"' not in contract
    assert compact["function"]["parameters"]["required"] == ["to", "subject"]
    assert "runtime" not in compact["function"]["parameters"]["properties"]


def test_disclosure_intersects_declarations_with_this_agent_tools():
    body = "---\nname: travel\ndescription: Travel help.\n---\n\nRULES"
    backend = SimpleNamespace(download_files=lambda _paths: [
        SimpleNamespace(error=None, content=body.encode("utf-8"))])
    middleware = SmallModelSkillsMiddleware(
        backend=backend, sources=["/skills/"], bundled_sources=["/skills/"],
        registered_tools={"travel"}, tool_definitions=(travel,))
    skill = {**_skill(), "allowed_tools": ["travel", "absent_tool"]}
    runtime = SimpleNamespace(
        state={"skills_metadata": [skill], "loaded_skill_tools": frozenset()},
        config={}, tool_call_id="load-1")

    result = middleware.tools[0].func("travel", runtime)

    assert result.update["loaded_skill_tools"] == frozenset({"travel"})
    assert "Newly available tools: travel." in result.update["messages"][0].content
    assert "Unavailable declared tools ignored: absent_tool." in \
        result.update["messages"][0].content


def test_disclosure_reports_unregistered_names_without_exposing_them():
    body = b"---\nname: travel\ndescription: Travel help.\n---\n\nRULES"
    backend = SimpleNamespace(download_files=lambda _paths: [
        SimpleNamespace(error=None, content=body)])
    middleware = SmallModelSkillsMiddleware(
        backend=backend, sources=["/external/"],
        gated_sources=["/external/"], registered_tools={"travel"},
        tool_definitions=(travel,))
    skill = {**_skill("/external/travel/SKILL.md"),
             "allowed_tools": ["travel", "not_registered"]}
    runtime = SimpleNamespace(
        state={"skills_metadata": [skill], "loaded_skill_tools": frozenset()},
        config={}, tool_call_id="load-1")

    result = middleware.tools[0].func("travel", runtime)

    message = result.update["messages"][0]
    assert result.update["loaded_skill_tools"] == frozenset({"travel"})
    assert "Unavailable declared tools ignored: not_registered." in message.content
    updated = middleware.modify_request(_request(
        middleware, {**runtime.state, "loaded_skill_tools": frozenset({"travel"})}))
    assert [item.name for item in updated.tools] == ["kernel_tool", "travel"]


def test_parallel_load_updates_have_union_reducer():
    body = b"---\nname: skill\ndescription: Help.\n---\n\nRULES"
    backend = SimpleNamespace(download_files=lambda _paths: [
        SimpleNamespace(error=None, content=body)])
    middleware = SmallModelSkillsMiddleware(
        backend=backend, sources=["/skills/"], bundled_sources=["/skills/"],
        tool_definitions=(kernel_tool, travel))
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
        backend=SimpleNamespace(), sources=["/skills/"],
        bundled_sources=["/skills/"])
    state = {
        "messages": [{"role": "user", "content": "I loaded travel."}],
        "skills_metadata": [_skill()],
        "loaded_skill_tools": frozenset(),
    }

    updated = middleware.modify_request(_request(middleware, state))
    with patch.object(middleware, "_acatalog_snapshot", new=AsyncMock(
            return_value=([_skill()], [], "current"))):
        reset = asyncio.run(middleware.abefore_agent(
            {**state, "skills_catalog_fingerprint": "current",
             "loaded_skill_tools": frozenset({"travel"})},
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
    assert len(update["messages"]) == 2
    assert update["messages"][0].name == "travel"
    assert update["messages"][0].status == "error"
    assert update["messages"][1].name == "kernel_tool"
    assert update["messages"][1].status == "error"
    assert update["jump_to"] == "model"


class _RecordingModel(FakeMessagesListChatModel):
    _bound_tools: list[list[str]] = PrivateAttr(default_factory=list)

    @property
    def bound_tools(self):
        return self._bound_tools

    @bound_tools.setter
    def bound_tools(self, value):
        self._bound_tools = value

    def bind_tools(self, tools, **_kwargs):
        self.bound_tools.append([
            (item.name if hasattr(item, "name") else
             item.get("name") or item.get("function", {}).get("name", ""))
            for item in tools
        ])
        return self


def test_compiled_graph_retains_skill_tool_for_the_next_invocation():
    from assist.agent import create_agent
    from assist.spec import AgentSpec
    from langgraph.checkpoint.memory import InMemorySaver

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
        checkpointer = InMemorySaver()
        first_agent = create_agent(
            model, wd, checkpointer=checkpointer,
            spec=AgentSpec(async_subagent_tools=()))
        config = {"configurable": {"thread_id": "skill-retention"}}
        first = first_agent.invoke(
            {"messages": [{"role": "user", "content": "route"}]}, config)
        before_second = len(model.bound_tools)
        rebuilt_agent = create_agent(
            model, wd, checkpointer=checkpointer,
            spec=AgentSpec(async_subagent_tools=()))
        second = rebuilt_agent.invoke(
            {"messages": [{"role": "user", "content": "again"}]}, config)

    assert "travel" not in model.bound_tools[0]
    assert "travel" in model.bound_tools[before_second]
    assert any(message.name == "travel" and message.status == "success"
               for message in first["messages"] if isinstance(message, ToolMessage))
    assert len([message for message in second["messages"]
                if isinstance(message, ToolMessage)
                and message.tool_call_id == "load-1"]) == 1
    continued = [message for message in second["messages"]
                 if isinstance(message, ToolMessage)
                 and message.tool_call_id == "travel-2"]
    assert len(continued) == 1 and continued[0].status == "success"
    assert second["messages"][-1].content == "second done"


def test_compiled_graph_discloses_domain_skill_tool_after_load():
    from assist.agent import create_agent
    from assist.spec import AgentSpec

    @tool("travel")
    def fake_travel(origin: str, destination: str) -> str:
        """Return a deterministic route."""
        return f"route:{origin}:{destination}"

    model = _RecordingModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "load_skill", "args": {"name": "domain-route"},
            "id": "load-domain"}]),
        AIMessage(content="", tool_calls=[{
            "name": "travel", "args": {"origin": "a", "destination": "b"},
            "id": "travel-domain"}]),
        AIMessage(content="done"),
    ])
    with tempfile.TemporaryDirectory() as wd:
        skill_dir = Path(wd) / ".claude" / "skills" / "domain-route"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: domain-route\ndescription: Route help.\n"
            "allowed-tools: travel\n---\n\nUse travel.\n",
            encoding="utf-8",
        )
        with patch("assist.agent.travel", fake_travel):
            agent = create_agent(
                model, wd, spec=AgentSpec(async_subagent_tools=()))
            result = agent.invoke(
                {"messages": [{"role": "user", "content": "route"}]},
                {"configurable": {"thread_id": "domain-disclosure"}},
            )

    assert "travel" not in model.bound_tools[0]
    assert any(message.name == "travel" and message.status == "success"
               for message in result["messages"] if isinstance(message, ToolMessage))


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


def test_compiled_graph_discloses_embedder_tool_after_load():
    from deepagents.backends import FilesystemBackend
    from assist.agent import create_agent
    from assist.spec import AgentSpec

    @tool("custom_lookup")
    def custom_lookup(value: str) -> str:
        """Return a deterministic custom lookup."""
        return f"lookup:{value}"

    model = _RecordingModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "load_skill", "args": {"name": "embedder-route"},
            "id": "load-embedder"}]),
        AIMessage(content="", tool_calls=[{
            "name": "custom_lookup", "args": {"value": "x"},
            "id": "lookup-embedder"}]),
        AIMessage(content="done"),
    ])
    with tempfile.TemporaryDirectory() as skill_root, tempfile.TemporaryDirectory() as wd:
        skill_dir = Path(skill_root) / "embedder-route"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: embedder-route\ndescription: Lookup help.\n"
            "allowed-tools: custom_lookup\n---\n\nUse custom_lookup.\n",
            encoding="utf-8",
        )
        agent = create_agent(
            model, wd,
            spec=AgentSpec(
                tools=(custom_lookup,), async_subagent_tools=(),
                skill_sources={
                    "/embedder-skills/": FilesystemBackend(
                        root_dir=skill_root, virtual_mode=True)}
            ),
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "lookup"}]},
            {"configurable": {"thread_id": "embedder-disclosure"}},
        )

    assert "custom_lookup" not in model.bound_tools[0]
    assert any(message.name == "custom_lookup" and message.status == "success"
               for message in result["messages"] if isinstance(message, ToolMessage))
