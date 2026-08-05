"""Deterministic gates for final prompt and capability census."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import edd.prompt_census as prompt_census

from edd.prompt_census import (
    DEFAULT_OUTPUT_ROOT,
    MAX_RUN_BYTES,
    REQUIRED_PATHS,
    _assert_hygiene,
    _prompt_provenance,
    _system_prompt,
    artifact_bytes,
    capture_census,
    observer_probe,
    write_capture,
)

pytestmark = pytest.mark.timeout(60)


@pytest.fixture(scope="module")
def census():
    return capture_census()


def _call(census, scenario, index=0):
    return next(call for call in census["calls"]
                if call["scenario"] == scenario and call["call_index"] == index)


def _resign(artifact):
    artifact.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = prompt_census._sha(artifact)


def _unsigned_copy(artifact):
    unsigned = copy.deepcopy(artifact)
    unsigned.pop("artifact_sha256")
    return unsigned


def _named_text_block(name):
    document = Path("docs/2026-07-26-agent-prompt-architecture.org").read_text(
        encoding="utf-8")
    marker = f"#+name: {name}\n#+begin_src text\n"
    return document.split(marker, 1)[1].split("\n#+end_src", 1)[0]


def test_reaches_every_required_real_path(census):
    assert {call["path"] for call in census["calls"]} == REQUIRED_PATHS

    for scenario in (
            "nested-research-worker", "nested-fact-check",
            "nested-report-critique"):
        assert [call["path"] for call in census["calls"]
                if call["scenario"] == scenario] == [
                    "research-lead", scenario, "research-lead"]


def test_final_provider_tools_are_the_visible_tools(census):
    for call in census["calls"]:
        schemas = call["provider_payload"].get("tools", [])
        names = [schema["function"]["name"] for schema in schemas]
        assert names == call["visible_tools"]
        assert len(names) == len(set(names))

    main = _call(census, "web-main-core")
    assert "general-purpose" not in _system_prompt(main).split(
        "Available subagent types:")[-1]
    assert "task" not in main["visible_tools"]
    assert "start_async_task" in main["visible_tools"]
    assert main["provider_payload"]["temperature"] == 0.1
    assert main["provider_payload"]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}}
    capture = _call(census, "capture")
    assert "extra_body" not in capture["provider_payload"]


def test_bundled_schema_disclosure_and_load_evidence_are_observed(census):
    initial = _call(census, "web-main-full", 0)
    disclosed = _call(census, "web-main-full", 1)

    assert "send_email" not in initial["visible_tools"]
    assert "send_email" in disclosed["visible_tools"]
    assert "travel" not in initial["visible_tools"]
    assert "notify" in initial["visible_tools"]
    assert "Tools available for this response: " \
        + ", ".join(initial["visible_tools"]) + "." in _system_prompt(initial)

    load = census["observations"]["web-main-full"]["tool_messages"][0]
    assert load["name"] == "load_skill"
    assert set(load["artifact"]) == {
        "schema", "requested_name", "winner_fingerprint", "result_sha256"}
    assert load["artifact"]["requested_name"] == "send-email"
    assert load["artifact"]["result_sha256"] == hashlib.sha256(
        load["content"].encode("utf-8")).hexdigest()
    assert "/render-skill/" not in json.dumps(load["artifact"])


def test_registered_visible_and_classification_stay_separate(census):
    for surface in census["capabilities"].values():
        registered = [tool["name"] for tool in surface["registered_tools"]]
        visible = surface["model_visible_tools"]
        assert not surface["ambiguous_tool_node_matches"]
        assert all(name in registered for name in visible)
        assert visible == [name for name in registered if name in visible]
        hidden = [tool for tool in surface["registered_tools"]
                  if tool["name"] not in visible]
        assert all(tool["name"] == "execute" or tool["possible_owners"]
                   for tool in hidden)

    surface = census["capabilities"]["web-main-core:0"]
    registered = [tool["name"] for tool in surface["registered_tools"]]
    assert set(surface["model_visible_tools"]) < set(registered)
    assert {tool["classification"] for tool in surface["registered_tools"]} == {
        "framework-kernel candidate", "skill-scoped candidate"}
    assert next(tool for tool in surface["registered_tools"]
                if tool["name"] == "write_todos")["origin"] \
        == "langchain.agents.middleware.todo"
    assert next(tool for tool in surface["registered_tools"]
                if tool["name"] == "create_schedule")["origin"] \
        == "assist.schedule.tools"
    assert next(tool for tool in surface["registered_tools"]
                if tool["name"] == "travel")["possible_owners"] == ["travel"]
    assert next(tool for tool in surface["registered_tools"]
                if tool["name"] == "map_data")["possible_owners"] == ["render"]
    assert "travel" not in surface["model_visible_tools"]
    assert "map_data" not in surface["model_visible_tools"]

    delegate = census["capabilities"]["web-delegate:0"]
    assert next(tool for tool in delegate["registered_tools"]
                if tool["name"] == "map_data")["possible_owners"] == []
    assert {
        (finding["kind"], finding["surface"], finding["detail"])
        for finding in census["findings"]
    } >= {("unowned-tool", "web-delegate:0", "map_data")}


def test_classified_kernel_matches_each_final_skills_enabled_request(census):
    filesystem = [
        "write_todos", "ls", "read_file", "write_file", "edit_file",
        "glob", "grep"]
    executable = [*filesystem, "execute"]
    async_lifecycle = [
        "start_async_task", "check_async_task", "update_async_task",
        "cancel_async_task", "list_async_tasks"]
    expected = {
        "web-main-core": [*executable, "load_skill", *async_lifecycle],
        "web-main-full": [*executable, "load_skill", *async_lifecycle],
        "web-delegate": [*executable, "task", "load_skill"],
        "legacy-main": [*filesystem, "task", "load_skill"],
        "skill-precedence-built-in": [
            *filesystem, "load_skill", *async_lifecycle],
        "skill-precedence-embedder": [
            *filesystem, "load_skill", *async_lifecycle],
    }
    for scenario, names in expected.items():
        call = _call(census, scenario)
        surface = census["capabilities"][f"{scenario}:{call['call_index']}"]
        actual = [
            tool["name"] for tool in surface["registered_tools"]
            if tool["classification"] == "framework-kernel candidate"
            and tool["name"] in call["visible_tools"]
        ]
        assert actual == names


def _recomputed_capabilities(census, source_manifest=None, tool_nodes=None):
    trace = prompt_census.CensusTrace(
        calls=census["calls"], tool_nodes=tool_nodes or census["tool_nodes"])
    sources = source_manifest or census["source_manifest"]
    return prompt_census._capabilities(
        trace, census["observations"], sources)


@pytest.mark.parametrize(("declared_name", "diagnostic"), [
    ("travle", "unknown declared tool `travle`"),
    ("execute(git:*)", "unknown declared tool `execute\\(git:\\*\\)`"),
    ("execute", "non-bundled or kernel tool `execute` must not be declared"),
])
def test_invalid_bundled_declarations_fail_with_path(
        census, declared_name, diagnostic):
    sources = copy.deepcopy(census["source_manifest"])
    travel = next(source for source in sources
                  if source.get("kind") == "skill"
                  and source.get("scenario") == "web-main-full"
                  and source.get("name") == "travel")
    travel["allowed_tools"].append(declared_name)
    capabilities = _recomputed_capabilities(census, sources)

    with pytest.raises(
            AssertionError,
            match=rf"{re.escape(travel['path'])}.*{diagnostic}"):
        prompt_census._validate_skill_tool_declarations(capabilities, sources)


def test_bad_winning_shadow_declaration_names_winning_path(census):
    sources = copy.deepcopy(census["source_manifest"])
    winner = next(source for source in sources
                  if source.get("kind") == "skill"
                  and source.get("scenario") == "skill-precedence-built-in"
                  and source.get("name") == "dev")
    winner["allowed_tools"].append("missing_embedder_tool")
    capabilities = _recomputed_capabilities(census, sources)

    with pytest.raises(AssertionError, match=(
            rf"{re.escape(winner['path'])}.*missing_embedder_tool")):
        prompt_census._validate_skill_tool_declarations(capabilities, sources)


def test_multiple_winning_owners_fail(census):
    sources = copy.deepcopy(census["source_manifest"])
    travel = next(source for source in sources
                  if source.get("kind") == "skill"
                  and source.get("scenario") == "web-main-full"
                  and source.get("name") == "travel")
    travel["allowed_tools"].append("map_data")
    capabilities = _recomputed_capabilities(census, sources)

    with pytest.raises(AssertionError, match=(
            r"non-kernel tool `map_data` has multiple owners: render, travel")):
        prompt_census._validate_skill_tool_declarations(capabilities, sources)


def test_external_declarations_are_recorded_but_not_enforced(census):
    external = next(source for source in census["source_manifest"]
                    if source.get("kind") == "skill"
                    and source.get("source") == "/.claude/skills/")
    assert external["allowed_tools"] == ["synthetic_external_tool"]
    capabilities = _recomputed_capabilities(census)

    prompt_census._validate_skill_tool_declarations(
        capabilities, census["source_manifest"])
    map_data = next(
        tool for tool in capabilities["web-main-full:0"]["registered_tools"]
        if tool["name"] == "map_data")
    assert map_data["possible_owners"] == ["render"]


def test_overridden_bundled_route_is_not_packaged_by_name_alone(census):
    source = copy.deepcopy(next(
        source for source in census["source_manifest"]
        if source.get("kind") == "skill"
        and source.get("source") == "/skills/"))
    source["content_sha256"] = "0" * 64

    assert prompt_census._packaged_skill_path(source) is None
    assert prompt_census._skill_tool_owners([source]) == {}


def test_external_registered_tools_do_not_require_bundled_owners(census):
    nodes = copy.deepcopy(census["tool_nodes"])
    node = next(node for node in nodes if node["scenario"] == "web-main-core")
    node["candidates"].append({
        "name": "embedder_tool", "origin": "external_embedder.tools"})
    capabilities = _recomputed_capabilities(census, tool_nodes=nodes)

    prompt_census._validate_skill_tool_declarations(
        capabilities, census["source_manifest"])


def test_application_kernel_name_collision_fails(census):
    nodes = copy.deepcopy(census["tool_nodes"])
    node = next(node for node in nodes if node["scenario"] == "web-main-core")
    node["candidates"].append({"name": "write_todos", "origin": "assist.tools"})

    with pytest.raises(
            AssertionError, match=(
                r"web-main-core:0: duplicate tool candidate `write_todos`.*"
                r"langchain\.agents\.middleware\.todo, assist\.tools")):
        _recomputed_capabilities(census, tool_nodes=nodes)


def test_same_class_tool_name_collision_fails(census):
    nodes = copy.deepcopy(census["tool_nodes"])
    node = next(node for node in nodes if node["scenario"] == "web-main-core")
    node["candidates"].append({
        "name": "travel", "origin": "assist.embedder_tools"})

    with pytest.raises(
            AssertionError, match=(
                r"web-main-core:0: duplicate tool candidate `travel`.*"
                r"assist\.tools, assist\.embedder_tools")):
        _recomputed_capabilities(census, tool_nodes=nodes)


def test_unmatched_tool_node_name_collision_fails(census):
    bad = copy.deepcopy(census)
    node = next(node for node in bad["tool_nodes"]
                if node["scenario"] == "capture")
    node["candidates"].append(copy.deepcopy(node["candidates"][0]))

    with pytest.raises(
            AssertionError, match=(
                rf"capture:tool-node:{node['index']}: duplicate tool candidate "
                rf"`{re.escape(node['candidates'][0]['name'])}`")):
        prompt_census._assert_semantic_views(bad)


def test_every_prompt_transition_is_observed_and_attributed(census):
    source_ids = {source["id"] for source in census["source_manifest"]}
    for call in census["calls"]:
        provenance = call["provenance"]
        spans = provenance["initial_spans"]
        if spans:
            assert spans[0]["start"] == 0
            assert spans[-1]["end"] == provenance["initial_characters"]
            assert all(left["end"] == right["start"]
                       for left, right in zip(spans, spans[1:]))
        events = call["prompt_events"]
        assert all(left["after_sha256"] == right["before_sha256"]
                   for left, right in zip(events, events[1:]))
        if events:
            assert events[-1]["after_sha256"] == provenance["final_sha256"]
        else:
            assert provenance["initial_sha256"] == provenance["final_sha256"]
        final_spans = provenance["final_spans"]
        assert final_spans[0]["start"] == 0
        assert final_spans[-1]["end"] == len(_system_prompt(call))
        assert all(left["end"] == right["start"]
                   for left, right in zip(final_spans, final_spans[1:]))
        for transition in provenance["transitions"]:
            assert set(transition["source_ids"]) <= source_ids
            segments = transition["content_segments"]
            if transition["exact_change"]:
                assert transition["source_ids"]
                assert segments[0]["start"] == 0
                assert segments[-1]["end"] == len(transition["exact_change"])
                assert all(left["end"] == right["start"]
                           for left, right in zip(segments, segments[1:]))
                assert all(set(segment["source_ids"]) <= source_ids
                           for segment in segments)
            else:
                assert transition["source_ids"] == []
                assert segments == []
            if transition["operation"] == "append":
                assert transition["exact_change"]
                assert transition["characters_after"] == (
                    transition["characters_before"]
                    + len(transition["exact_change"]))
            elif transition["operation"] == "replace":
                assert len(transition["exact_change"]) \
                    == transition["characters_after"]
        for span in provenance["initial_spans"]:
            assert span["source_id"] in source_ids
            source = next(source for source in census["source_manifest"]
                          if source["id"] == span["source_id"])
            if source["kind"] == "template":
                assert span["rendered_sha256"] == source["rendered_sha256"]
        for span in final_spans:
            references = ([span["source_id"]] if "source_id" in span
                          else span["source_ids"])
            assert references
            assert set(references) <= source_ids

    full = _call(census, "web-main-full")
    rider = next(transition for transition in full["provenance"]["transitions"]
                 if transition["owner"] == "assist.ContextRiderMiddleware")
    projected_owners = {segment.get("owner")
                        for segment in rider["content_segments"]
                        if segment.get("owner")}
    assert "assist.agent template" in projected_owners
    assert "deepagents.MemoryMiddleware" in projected_owners


def test_ambiguous_constructor_prompt_ownership_fails():
    call = {
        "scenario": "synthetic-ambiguity",
        "call_index": 0,
        "prompt_events": [],
        "system_blocks": [{"type": "text", "text": "SAME PROMPT"}],
    }
    constructors = [
        {"owner": "owner-a", "text": "SAME PROMPT", "sha256": "a",
         "source_id": "source-a"},
        {"owner": "owner-b", "text": "SAME PROMPT", "sha256": "b",
         "source_id": "source-b"},
    ]
    with pytest.raises(AssertionError, match="ambiguous constructor"):
        _prompt_provenance(call, constructors, [], {})


def test_observer_does_not_change_final_request():
    assert observer_probe(instrumented=True) == observer_probe(instrumented=False)


def test_observer_rejects_malformed_child_payload(monkeypatch):
    monkeypatch.setattr(prompt_census, "_run_isolated_json", lambda *_args: {})
    with pytest.raises(AssertionError, match="provider settings drifted"):
        observer_probe(instrumented=True)


def test_observer_probe_rejects_recorded_faults(monkeypatch):
    original = prompt_census._instrument

    @prompt_census.contextmanager
    def faulted(trace):
        with original(trace):
            yield
        trace.faults.append(AssertionError("synthetic observer fault"))

    monkeypatch.setattr(prompt_census, "_instrument", faulted)
    with pytest.raises(AssertionError, match="synthetic observer fault"):
        prompt_census._observer_probe(instrumented=True)


def test_observer_probe_rejects_extra_calls(monkeypatch):
    original = prompt_census.RecordingChatModel._generate

    def duplicated(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        self._trace.calls.append(copy.deepcopy(self._trace.calls[-1]))
        return result

    monkeypatch.setattr(prompt_census.RecordingChatModel, "_generate", duplicated)
    with pytest.raises(AssertionError, match="recorded 2 provider calls"):
        prompt_census._observer_probe(instrumented=False)


def test_enforcement_is_exercised_not_inferred(census):
    core = census["observations"]["web-main-core"]
    assert core["sandbox_commands"] == ["printf synthetic-ok"]
    assert [message["status"] for message in core["tool_messages"]] == [
        "success", "error"]
    assert "direct git push is not allowed" in core["tool_messages"][1]["content"]

    full = census["observations"]["web-main-full"]
    assert full["interrupted"] is True
    assert full["email_delivery_attempted"] is False
    assert full["sandbox_commands"] == []

    context = census["observations"]["context-read-only"]
    assert [message["status"] for message in context["tool_messages"]] \
        == ["error", "error"]
    assert all("read-only" in message["content"]
               for message in context["tool_messages"])

    leaf = census["observations"]["research-leaf-provenance"]
    assert leaf["tool_messages"][0]["status"] == "error"
    assert "URL was a guess" in leaf["tool_messages"][0]["content"]


def test_fixed_role_delegate_and_capture_input_boundaries(census):
    receptionist = _call(census, "receptionist")
    assert receptionist["visible_tools"] == [
        "list_threads", "open_thread", "new_thread"]
    assert not ({"ls", "execute", "task"} & set(receptionist["visible_tools"]))
    assert [claim["tool"] for claim in
            census["capabilities"]["receptionist:0"]["claims"]] == [
                "list_threads", "open_thread", "new_thread"]

    delegate = _call(census, "web-delegate")
    assert "task" in delegate["visible_tools"]
    assert not ({"start_async_task", "check_async_task"}
                & set(delegate["visible_tools"]))

    capture = _call(census, "capture")
    capture_task = capture["provider_payload"]["messages"][1]["content"]
    assert "You are an expert test case generator" in capture_task
    assert "SYNTHETIC CAPTURE REASON" in capture_task
    assert "SYNTHETIC CAPTURE USER REQUEST" in capture_task


def test_skill_source_precedence_matches_listing_and_load(census):
    built_in = census["observations"]["skill-precedence-built-in"]
    embedder = census["observations"]["skill-precedence-embedder"]
    assert "# Software Development Workflow" in built_in["loaded_skill_bodies"][0]
    assert "SYNTHETIC_DOMAIN_DEV_BODY" not in built_in["loaded_skill_bodies"][0]
    assert "SYNTHETIC_EMBEDDER_DEV_BODY" in embedder["loaded_skill_bodies"][0]

    built_prompt = _system_prompt(_call(census, "skill-precedence-built-in"))
    embedder_prompt = _system_prompt(_call(census, "skill-precedence-embedder"))
    assert "Software development work in a code project" in built_prompt
    assert "SYNTHETIC DOMAIN DEV DESCRIPTION" not in built_prompt
    assert "SYNTHETIC EMBEDDER DEV DESCRIPTION" in embedder_prompt


def test_expected_audit_findings_are_reported_not_hidden(census):
    assert [(finding["kind"], finding["surface"])
            for finding in census["findings"]] == [
        ("unowned-tool", "web-main-core:0"),
        ("unowned-tool", "web-main-full:0"),
        ("unowned-tool", "web-delegate:0"),
        ("unowned-tool", "legacy-main:0"),
        ("unowned-tool", "skill-precedence-built-in:0"),
        ("unowned-tool", "skill-precedence-embedder:0"),
        ("unavailable-capability-claim", "skill-precedence-built-in:1"),
        ("contradictory-capability-claims", "context-read-only:0"),
        ("contradictory-capability-claims", "context-read-only:0"),
        ("denied-capability-claim", "context-read-only:0"),
        ("denied-capability-claim", "context-read-only:0"),
        ("contradictory-capability-claims", "context-read-only:1"),
        ("contradictory-capability-claims", "context-read-only:1"),
        ("denied-capability-claim", "context-read-only:1"),
        ("denied-capability-claim", "context-read-only:1"),
        ("contradictory-capability-claims", "context-read-only:2"),
        ("contradictory-capability-claims", "context-read-only:2"),
        ("denied-capability-claim", "context-read-only:2"),
        ("denied-capability-claim", "context-read-only:2"),
        ("argument-contract-mismatch", "capture:0"),
        ("argument-contract-mismatch", "capture:0"),
        ("argument-contract-mismatch", "capture:0"),
        ("prompt-flattening", "web-main-full:0"),
        ("return-shape-claim-mismatch", "web-main-core:0"),
        ("return-shape-claim-mismatch", "web-main-full:0"),
    ]
    assert not any(
        finding["kind"] == "unavailable-capability-claim"
        and "SkillsMiddleware" in finding["detail"]
        and "`task`" in finding["detail"]
        for finding in census["findings"])
    for surface_key, surface in census["capabilities"].items():
        for claim in surface["claims"]:
            if claim["polarity"] == "positive" \
                    and claim["tool"] not in surface["model_visible_tools"]:
                assert any(
                    finding["kind"] == "unavailable-capability-claim"
                    and finding["surface"] == surface_key
                    and f"`{claim['tool']}`" in finding["detail"]
                    for finding in census["findings"])
    main_claims = census["capabilities"]["web-main-core:0"]["claims"]
    assert any(claim["polarity"] == "conditional" and claim["tool"] == "notify"
               for claim in main_claims)
    assert any(claim["polarity"] == "ordered" and claim["tool"] == "task"
               for claim in main_claims)
    context = census["capabilities"]["context-read-only:0"]
    assert context["effective_actions"]["write_file"] == "observed-denied"
    assert context["effective_actions"]["edit_file"] == "observed-denied"
    assert census["capabilities"]["research-leaf-provenance:0"][
        "effective_actions"]["read_url"] == "observed-denied-by-provenance"
    assert "read_url" not in census["capabilities"]["web-main-core:0"][
        "effective_actions"]
    assert "read_url" not in census["capabilities"]["web-delegate:0"][
        "effective_actions"]
    checked = json.loads(
        census["observations"]["async-task-return-contract"]["checked"])
    assert checked["agent_name"] == "research-agent"
    assert checked["result"] == "SYNTHETIC DIRECT RESEARCH FINDINGS"


def test_artifact_is_bounded_and_hygiene_checked(census, tmp_path):
    main = _call(census, "web-main-core")
    assert len(artifact_bytes(census)) \
        + len(_system_prompt(main).encode()) < MAX_RUN_BYTES
    bad = {"dynamic": str(Path.home() / "private-thread")}
    with pytest.raises(AssertionError):
        _assert_hygiene(bad, tmp_path)
    for leaked in (
            "REAL USER THREAD CONTENT WITHOUT PATHS",
            "PRIVATE USER FACT plus synthetic",
            "/var/lib/assist/threads/real-thread",
            "AKIAIOSFODNN7EXAMPLE",
            "+1 415-555-0123",
            "Bearer ordinary-unrecognized-secret"):
        bad = copy.deepcopy(census)
        user_message = next(
            message for message in bad["calls"][0]["provider_payload"]["messages"]
            if message["role"] == "user")
        user_message["content"] = leaked
        with pytest.raises(AssertionError):
            _assert_hygiene(bad, tmp_path)

    bad = copy.deepcopy(census)
    bad["private_metadata"] = "PRIVATE REAL USER DATA"
    with pytest.raises(AssertionError, match="undeclared fields"):
        _assert_hygiene(bad, tmp_path)

    for collection, key in (
            ("source_manifest", 0),
            ("tool_nodes", 0),
            ("findings", 0)):
        bad = copy.deepcopy(census)
        bad[collection][key]["private_metadata"] = "PRIVATE REAL USER DATA"
        with pytest.raises(AssertionError, match="undeclared fields"):
            _assert_hygiene(bad, tmp_path)

    bad = copy.deepcopy(census)
    bad["capabilities"]["web-main-core:0"]["private_metadata"] = \
        "PRIVATE REAL USER DATA"
    with pytest.raises(AssertionError, match="undeclared fields"):
        _assert_hygiene(bad, tmp_path)

    bad = copy.deepcopy(census)
    bad["calls"][0]["provider_payload"]["private_metadata"] = \
        "PRIVATE REAL USER DATA"
    with pytest.raises(AssertionError, match="undeclared fields"):
        _assert_hygiene(bad, tmp_path)

    for role in ("system", "user", "assistant"):
        bad = copy.deepcopy(census)
        message = next(
            message for call in bad["calls"]
            for message in call["provider_payload"]["messages"]
            if message["role"] == role)
        message["private_metadata"] = "PRIVATE REAL USER DATA"
        with pytest.raises(AssertionError, match="undeclared fields"):
            _assert_hygiene(bad, tmp_path)

    bad = copy.deepcopy(census)
    system = next(
        message for call in bad["calls"]
        for message in call["provider_payload"]["messages"]
        if message["role"] == "system" and isinstance(message["content"], list))
    system["content"][0]["private_metadata"] = "PRIVATE REAL USER DATA"
    _resign(bad)
    with pytest.raises(AssertionError, match="undeclared fields"):
        _assert_hygiene(bad, tmp_path)

    bad = copy.deepcopy(census)
    tool_message = next(
        message for call in bad["calls"]
        for message in call["provider_payload"]["messages"]
        if message["role"] == "tool")
    tool_message["content"] = "PRIVATE LIVE TOOL OUTPUT belonging to a real user"
    with pytest.raises(AssertionError, match="provider history"):
        _assert_hygiene(bad, tmp_path)

    bad = copy.deepcopy(census)
    tool_call = next(
        tool_call
        for call in bad["calls"]
        for message in call["provider_payload"]["messages"]
        for tool_call in message.get("tool_calls", []))
    tool_call["function"]["name"] = "read_file"
    with pytest.raises(AssertionError, match="provider history"):
        _assert_hygiene(bad, tmp_path)

    bad = copy.deepcopy(census)
    tool_call = next(
        tool_call
        for call in bad["calls"]
        for message in call["provider_payload"]["messages"]
        for tool_call in message.get("tool_calls", []))
    tool_call["private_metadata"] = "PRIVATE REAL USER DATA"
    with pytest.raises(AssertionError, match="undeclared fields"):
        _assert_hygiene(bad, tmp_path)

    bad = copy.deepcopy(census)
    tool_schema = next(
        schema for call in bad["calls"]
        for schema in call["provider_payload"].get("tools", []))
    tool_schema["function"]["parameters"]["private_metadata"] = True
    with pytest.raises(AssertionError, match="tool schema"):
        _assert_hygiene(bad, tmp_path)

    bad = _unsigned_copy(census)
    tool_schema = next(
        schema for call in bad["calls"]
        for schema in call["provider_payload"].get("tools", []))
    tool_schema["function"]["description"] = "PRIVATE REAL USER DATA"
    with pytest.raises(AssertionError, match="schema is not declared"):
        _assert_hygiene(bad, tmp_path)

    bad = _unsigned_copy(census)
    bad["tool_nodes"][0]["candidates"][0]["origin"] = "private.fake.module"
    with pytest.raises(AssertionError, match="ToolNode history"):
        _assert_hygiene(bad, tmp_path)

    bad = _unsigned_copy(census)
    bad["capabilities"]["web-main-core:0"]["effective_actions"][
        "load_skill"] = "observed-permitted"
    with pytest.raises(AssertionError, match="capability surfaces drifted"):
        _assert_hygiene(bad, tmp_path)

    bad = _unsigned_copy(census)
    call = _call(bad, "web-main-core")
    call["provider_payload"]["tools"] = [
        schema for schema in call["provider_payload"]["tools"]
        if schema["function"]["name"] != "load_skill"]
    with pytest.raises(AssertionError, match="schema surface drifted"):
        _assert_hygiene(bad, tmp_path)

    bad = _unsigned_copy(census)
    source = next(source for source in bad["source_manifest"]
                  if source["id"] ==
                  "python:assist.middleware.context_rider_middleware")
    source["source_sha256"] = "f" * 64
    with pytest.raises(AssertionError, match="fixed prompt source manifest"):
        _assert_hygiene(bad, tmp_path)

    bad = _unsigned_copy(census)
    source = next(source for source in bad["source_manifest"]
                  if source.get("kind") == "skill"
                  and source.get("source") == "/skills/")
    source["allowed_tools"] = ["private_fake_tool"]
    with pytest.raises(AssertionError, match="packaged skill metadata drifted"):
        _assert_hygiene(bad, tmp_path)

    bad = _unsigned_copy(census)
    value = "REAL USER THREAD CONTENT WITHOUT PATHS"
    bad["source_manifest"].append({
        "id": f"fixture:web-main-core:forged:{prompt_census._sha(value)[:16]}",
        "kind": "synthetic-fixture",
        "scenario": "web-main-core",
        "name": "forged",
        "value": value,
        "value_sha256": prompt_census._sha(value),
    })
    with pytest.raises(AssertionError, match="exactly reachable"):
        _assert_hygiene(bad, tmp_path)

    bad = copy.deepcopy(census)
    bad["observations"]["web-main-full"]["interrupted"] = False
    with pytest.raises(AssertionError, match="interrupt state"):
        _assert_hygiene(bad, tmp_path)

    bad = copy.deepcopy(census)
    bad["observations"]["web-main-core"]["tool_messages"][0]["extra"] = True
    with pytest.raises(AssertionError, match="tool-result shape"):
        _assert_hygiene(bad, tmp_path)

    bad = copy.deepcopy(census)
    bad["observations"]["thread-description"]["description"] = "PRIVATE RESULT"
    with pytest.raises(AssertionError, match="thread description"):
        _assert_hygiene(bad, tmp_path)

    bad = copy.deepcopy(census)
    bad["observations"]["async-task-return-contract"]["launch"] += \
        " PRIVATE REAL USER DATA"
    _resign(bad)
    with pytest.raises(AssertionError, match="async task return contract"):
        _assert_hygiene(bad, tmp_path)

    bad = copy.deepcopy(census)
    full = next(call for call in bad["calls"]
                if call["scenario"] == "web-main-full")
    system = next(message for message in full["provider_payload"]["messages"]
                  if message["role"] == "system")
    system["content"] = system["content"].replace(
        "SYNTHETIC PLACE", "PIERRE ACTUAL HOUSE")
    with pytest.raises(AssertionError):
        _assert_hygiene(bad, tmp_path)

    bad = copy.deepcopy(census)
    full = next(call for call in bad["calls"]
                if call["scenario"] == "web-main-full")
    system = next(message for message in full["provider_payload"]["messages"]
                  if message["role"] == "system")
    system["content"] = system["content"].replace(
        "SYNTHETIC_REPOSITORY_MEMORY", "SYNTHETIC_REPOSITORY_MEMORY EXTRA")
    with pytest.raises(AssertionError, match="agent_memory"):
        _assert_hygiene(bad, tmp_path)

    bad = copy.deepcopy(census)
    skill_segment = next(
        segment
        for call in bad["calls"]
        for transition in call["provenance"]["transitions"]
        for segment in transition["content_segments"]
        if any(source_id.startswith("skill:")
               for source_id in segment["source_ids"]))
    skill_segment["source_ids"] = ["python:deepagents.middleware.skills"]
    with pytest.raises(AssertionError, match="prompt byte ownership"):
        _assert_hygiene(bad, tmp_path)

    bad = copy.deepcopy(census)
    built_in_call = next(call for call in bad["calls"]
                         if call["scenario"] == "skill-precedence-built-in")
    transition = next(
        transition for transition in built_in_call["provenance"]["transitions"]
        if transition["owner"] == "deepagents.SkillsMiddleware")
    built_in_source = next(
        source["id"] for source in bad["source_manifest"]
        if source.get("kind") == "skill"
        and source.get("scenario") == "skill-precedence-built-in"
        and source.get("name") == "dev")
    embedder_source = next(
        source["id"] for source in bad["source_manifest"]
        if source.get("kind") == "skill"
        and source.get("scenario") == "skill-precedence-embedder"
        and source.get("name") == "dev")
    segment = next(segment for segment in transition["content_segments"]
                   if segment["source_ids"] == [built_in_source])
    segment["source_ids"] = [embedder_source]
    transition["source_ids"] = list(dict.fromkeys(
        source_id
        for item in transition["content_segments"]
        for source_id in item["source_ids"]))
    with pytest.raises(AssertionError, match="prompt byte ownership"):
        _assert_hygiene(bad, tmp_path)

    bad = copy.deepcopy(census)
    final_span = next(
        span for call in bad["calls"]
        for span in call["provenance"]["final_spans"]
        if "source_id" in span)
    final_span["source_id"] = "python:assist.middleware.context_rider_middleware"
    with pytest.raises(AssertionError, match="final prompt attribution"):
        _assert_hygiene(bad, tmp_path)

    bad = _unsigned_copy(census)
    transition = next(
        transition for call in bad["calls"]
        for transition in call["provenance"]["transitions"]
        if transition["owner"] == "deepagents.SkillsMiddleware")
    transition["exact_change"] = transition["exact_change"].replace(
        "## Skills", "## Skillz", 1)
    with pytest.raises(AssertionError, match="transition output"):
        _assert_hygiene(bad, tmp_path)

    bad = _unsigned_copy(census)
    transition = bad["calls"][0]["provenance"]["transitions"][0]
    transition["before_sha256"] = "0" * 64
    with pytest.raises(AssertionError, match="event/transition"):
        _assert_hygiene(bad, tmp_path)

    bad = copy.deepcopy(census)
    full = _call(bad, "web-main-full")
    rider_event = next(
        event for event in full["prompt_events"]
        if event["owner"] == "assist.ContextRiderMiddleware")
    rider_event["stringified_prior_blocks"] = False
    bad["findings"] = [
        finding for finding in bad["findings"]
        if finding["kind"] != "prompt-flattening"]
    _resign(bad)
    with pytest.raises(AssertionError, match="prompt flattening evidence"):
        _assert_hygiene(bad, tmp_path)

    bad = _unsigned_copy(census)
    bad["calls"][0]["prompt_events"][0]["owner"] = "PRIVATE OWNER"
    with pytest.raises(AssertionError, match="event/transition"):
        _assert_hygiene(bad, tmp_path)

    bad = _unsigned_copy(census)
    call = next(call for call in bad["calls"]
                if any(span["owner"] == "framework prompt composer"
                       for span in call["provenance"]["initial_spans"]))
    composer = next(span for span in call["provenance"]["initial_spans"]
                    if span["owner"] == "framework prompt composer")
    text = call["provenance"]["initial_text"]
    call["provenance"]["initial_text"] = (
        text[:composer["start"]] + "X" + text[composer["start"] + 1:])
    with pytest.raises(AssertionError, match="initial prompt text metadata"):
        _assert_hygiene(bad, tmp_path)

    bad = _unsigned_copy(census)
    call = next(call for call in bad["calls"]
                if any(span["owner"] == "deepagents.graph.BASE_AGENT_PROMPT"
                       for span in call["provenance"]["initial_spans"]))
    initial_span = next(
        span for span in call["provenance"]["initial_spans"]
        if span["owner"] == "deepagents.graph.BASE_AGENT_PROMPT")
    original_source = initial_span["source_id"]
    initial_span["source_id"] = "python:assist.middleware.context_rider_middleware"
    final_span = next(
        span for span in call["provenance"]["final_spans"]
        if span.get("source_id") == original_source
        and span["start"] == initial_span["start"])
    final_span["source_id"] = initial_span["source_id"]
    with pytest.raises(AssertionError, match="prompt block provenance|base prompt source"):
        _assert_hygiene(bad, tmp_path)

    bad = _unsigned_copy(census)
    user = next(message for message in bad["calls"][0]["provider_payload"]["messages"]
                if message["role"] == "user")
    user["content"] = "SYNTHETIC USER web-main-full"
    with pytest.raises(AssertionError, match="provider history"):
        _assert_hygiene(bad, tmp_path)

    bad = _unsigned_copy(census)
    delegate = _call(bad, "web-delegate")
    legacy = _call(bad, "legacy-main")
    delegate_task = next(schema for schema in delegate["provider_payload"]["tools"]
                         if schema["function"]["name"] == "task")
    legacy_task = next(schema for schema in legacy["provider_payload"]["tools"]
                       if schema["function"]["name"] == "task")
    delegate_task.clear()
    delegate_task.update(copy.deepcopy(legacy_task))
    with pytest.raises(AssertionError, match="schema is not declared"):
        _assert_hygiene(bad, tmp_path)

    bad = _unsigned_copy(census)
    for call in bad["calls"]:
        if call["scenario"] == "context-read-only":
            call["path"] = "research-leaf"
        elif call["scenario"] == "research-leaf-provenance":
            call["path"] = "context"
    for key, surface in bad["capabilities"].items():
        if key.startswith("context-read-only:"):
            surface["path"] = "research-leaf"
        elif key.startswith("research-leaf-provenance:"):
            surface["path"] = "context"
    with pytest.raises(AssertionError, match="scenario-to-path"):
        _assert_hygiene(bad, tmp_path)

    bad = _unsigned_copy(census)
    bad["tool_nodes"].append({
        "index": len(bad["tool_nodes"]),
        "scenario": "context-read-only",
        "candidates": [{"name": "send_email", "origin": "assist.events.email"}],
        "winners": ["send_email"],
    })
    with pytest.raises(AssertionError, match="ToolNode history"):
        _assert_hygiene(bad, tmp_path)

    bad = _unsigned_copy(census)
    full = _call(bad, "web-main-core")
    system = full["provider_payload"]["messages"][0]
    blocks = system["content"]
    system["content"] = [
        {"type": "text", "text": blocks[0]["text"] + blocks[1]["text"]},
        *blocks[2:],
    ]
    final_sha = prompt_census._sha(system["content"])
    full["provenance"]["final_sha256"] = final_sha
    full["provenance"]["final_block_layout"] = prompt_census._block_layout(
        system["content"], full["provenance"]["final_spans"])
    full["prompt_events"][-1]["after_sha256"] = final_sha
    full["provenance"]["transitions"][-1]["after_sha256"] = final_sha
    with pytest.raises(AssertionError, match="block transition chain"):
        _assert_hygiene(bad, tmp_path)


def test_recorder_integrity_failures_escape_agent_middleware(monkeypatch):
    original = prompt_census._prompt_provenance

    def fail_one_call(call, *args, **kwargs):
        if call["scenario"] == "skill-precedence-built-in" \
                and call["call_index"] == 1:
            raise AssertionError("synthetic integrity failure")
        return original(call, *args, **kwargs)

    monkeypatch.setattr(prompt_census, "_prompt_provenance", fail_one_call)
    with pytest.raises(AssertionError, match="synthetic integrity failure"):
        prompt_census._capture_census()


def test_undeclared_provider_calls_escape_agent_middleware(monkeypatch):
    original = prompt_census._path_for_call

    def fail_one_call(scenario, index):
        if scenario == "skill-precedence-built-in" and index == 1:
            raise IndexError("synthetic extra provider call")
        return original(scenario, index)

    monkeypatch.setattr(prompt_census, "_path_for_call", fail_one_call)
    with pytest.raises(
            AssertionError,
            match="undeclared provider call skill-precedence-built-in:1"):
        prompt_census._capture_census()


def test_malformed_candidate_skill_blocks_capture(monkeypatch):
    original = prompt_census._instrument

    @prompt_census.contextmanager
    def malformed(trace):
        import deepagents.middleware.skills as skills_mod

        real_parse = skills_mod._parse_skill_metadata

        def parse(content, skill_path, directory_name):
            if "/calculate/" in skill_path:
                return None
            return real_parse(content, skill_path, directory_name)

        with monkeypatch.context() as patcher:
            patcher.setattr(skills_mod, "_parse_skill_metadata", parse)
            with original(trace):
                yield

    monkeypatch.setattr(prompt_census, "_instrument", malformed)
    with pytest.raises(AssertionError, match="candidate skill metadata"):
        prompt_census._capture_census()


def test_repeated_isolated_capture_does_not_leak_process_global_state():
    first = capture_census()
    second = capture_census()
    assert first["artifact_sha256"] == second["artifact_sha256"]


def test_isolated_child_failure_surfaces_diagnostic_and_cleans_scratch(
        monkeypatch):
    scratch = []
    monkeypatch.setenv("PROMPT_CENSUS_SYNTHETIC_SECRET", "must-not-reach-child")

    def timeout(*_args, **kwargs):
        repo = Path(prompt_census.__file__).resolve().parents[1]
        assert kwargs["cwd"] == repo
        assert kwargs["env"]["PYTHONPATH"] == str(repo)
        assert "PROMPT_CENSUS_SYNTHETIC_SECRET" not in kwargs["env"]
        path = Path(kwargs["env"]["TMPDIR"])
        scratch.append(path)
        (path / "child-leftover").write_text("partial")
        raise subprocess.TimeoutExpired(
            "synthetic", 120, stderr=b"SYNTHETIC CHILD DIAGNOSTIC")

    monkeypatch.setattr(prompt_census.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="SYNTHETIC CHILD DIAGNOSTIC"):
        prompt_census._run_isolated_json("--stdout-census")
    assert scratch and not scratch[0].exists()


def test_isolated_child_exit_and_malformed_output_are_distinct(monkeypatch):
    def failed(*_args, **_kwargs):
        raise subprocess.CalledProcessError(
            7, "synthetic", stderr=b"SYNTHETIC EXIT DIAGNOSTIC")

    monkeypatch.setattr(prompt_census.subprocess, "run", failed)
    with pytest.raises(RuntimeError, match="exited with status 7.*EXIT DIAGNOSTIC"):
        prompt_census._run_isolated_json("--stdout-census")

    completed = subprocess.CompletedProcess("synthetic", 0, stdout=b"not-json")
    monkeypatch.setattr(
        prompt_census.subprocess, "run", lambda *_args, **_kwargs: completed)
    with pytest.raises(RuntimeError, match="returned malformed JSON"):
        prompt_census._run_isolated_json("--stdout-census")

    completed = subprocess.CompletedProcess("synthetic", 0, stdout=b"\xff")
    with pytest.raises(RuntimeError, match="returned malformed JSON"):
        prompt_census._run_isolated_json("--stdout-census")

    completed = subprocess.CompletedProcess("synthetic", 0, stdout=b"[]")
    with pytest.raises(RuntimeError, match="non-object JSON value"):
        prompt_census._run_isolated_json("--stdout-census")


def test_parent_revalidates_isolated_artifact(census, monkeypatch):
    tampered = copy.deepcopy(census)
    tampered["calls"][0]["provider_payload"]["messages"][1]["content"] = \
        "REAL USER THREAD CONTENT WITHOUT PATHS"
    _resign(tampered)
    monkeypatch.setattr(prompt_census, "_run_isolated_json", lambda *_args: tampered)
    with pytest.raises(AssertionError, match="provider history drifted"):
        capture_census()


def test_persistent_capture_is_home_backed_no_replace_and_atomic(
        census, tmp_path, monkeypatch):
    monkeypatch.setattr(prompt_census, "capture_census", lambda: census)
    assert DEFAULT_OUTPUT_ROOT.is_relative_to(Path.home())
    destination = write_capture("p0-test", output_root=tmp_path)
    assert (destination / "census.json").is_file()
    assert (destination / "web-main-bootstrap.txt").is_file()
    with pytest.raises(FileExistsError):
        write_capture("p0-test", output_root=tmp_path)
    assert not list(tmp_path.glob(".p0-test-*"))

    reserved = tmp_path / "reserved"
    reserved.mkdir()
    with pytest.raises(FileExistsError):
        write_capture("reserved", output_root=tmp_path)
    assert reserved.is_dir()
    assert not list(reserved.iterdir())


def test_two_fresh_process_captures_are_byte_stable(tmp_path):
    roots = [tmp_path / "first", tmp_path / "second"]
    for root in roots:
        root.mkdir()
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; from edd.prompt_census import "
                "write_capture; write_capture('p0', output_root=Path(__import__("
                "'sys').argv[1]))",
                str(root),
            ],
            check=True, capture_output=True, text=True, timeout=60,
        )
    first = (roots[0] / "p0" / "census.json").read_bytes()
    second = (roots[1] / "p0" / "census.json").read_bytes()
    assert first == second
    assert json.loads(first)["artifact_sha256"]


def test_publication_preflights_and_reports_directory_durability(
        census, tmp_path, monkeypatch):
    monkeypatch.setattr(prompt_census, "capture_census", lambda: census)

    def unsupported(_path):
        raise OSError("directory fsync unsupported")

    monkeypatch.setattr(prompt_census, "_fsync", unsupported)
    with pytest.raises(OSError, match="unsupported"):
        write_capture("preflight", output_root=tmp_path)
    assert not (tmp_path / "preflight").exists()

    calls = []

    def fail_after_publish(path):
        calls.append(path)
        if path == tmp_path and calls.count(tmp_path) == 2:
            raise OSError("post-publish fsync failed")

    monkeypatch.setattr(prompt_census, "_fsync", fail_after_publish)
    with pytest.raises(RuntimeError, match="published.*durability is uncertain"):
        write_capture("uncertain", output_root=tmp_path)
    assert (tmp_path / "uncertain" / "census.json").is_file()


def test_publication_reports_staging_cleanup_failure(tmp_path, monkeypatch):
    def primary_failure():
        raise RuntimeError("synthetic capture failure")

    original_rmtree = prompt_census.shutil.rmtree
    monkeypatch.setattr(prompt_census, "capture_census", primary_failure)
    monkeypatch.setattr(
        prompt_census.shutil, "rmtree",
        lambda _path: (_ for _ in ()).throw(OSError("synthetic cleanup failure")))
    with pytest.raises(ExceptionGroup, match="staging cleanup failed") as caught:
        write_capture("cleanup", output_root=tmp_path)
    assert len(caught.value.exceptions) == 2
    staging = next(tmp_path.glob(".cleanup-*"))
    monkeypatch.setattr(prompt_census.shutil, "rmtree", original_rmtree)
    original_rmtree(staging)


def test_keyboard_interrupt_cleans_publication_staging(tmp_path, monkeypatch):
    monkeypatch.setattr(
        prompt_census, "capture_census",
        lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        write_capture("interrupted", output_root=tmp_path)
    assert not list(tmp_path.glob(".interrupted-*"))


def test_p0_through_p2b2_history_matches_the_current_capture(census):
    document = Path("docs/2026-07-26-agent-prompt-architecture.org").read_text(
        encoding="utf-8")
    p2_document = Path("docs/2026-08-04-prompt-architecture-p2.org").read_text(
        encoding="utf-8")
    p2b1_document = Path(
        "docs/2026-08-05-prompt-architecture-p2b1.org").read_text(
        encoding="utf-8")
    p2b2_document = Path(
        "docs/2026-08-05-prompt-architecture-p2b2.org").read_text(
        encoding="utf-8")
    current = _call(census, "web-main-core")
    historical_p0_prompt = _named_text_block("p0-current-web-main-bootstrap")
    current_prompt = _system_prompt(current)

    schemas = json.dumps(
        current["provider_payload"]["tools"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert len(historical_p0_prompt) == 31_279
    assert len(current_prompt) == 29_527
    assert len(schemas) == 17_984
    assert len(census["calls"]) == 29
    assert len(census["tool_nodes"]) == 38
    assert len(census["findings"]) == 25
    assert len(artifact_bytes(census)) == 2_992_991
    assert census["artifact_sha256"] == \
        "a16e88ba2b5f0187a67423591138a856c81ed7c60aa1a547e4a5768d76df2490"
    assert "2,920,942 bytes (2.8 MiB)" in document
    assert "p0-100aa885-final-v2" in document
    expected_rows = [
        "| Assist role instructions | P0 baseline custom main template: role, routing, trust, research, artifact, and lifecycle procedure | 13,068 | 3,267 |",
        "| Prompt composer | Separator between caller and framework prompt | 2 | 1 |",
        "| Deep Agents base | Generic core behavior, objectivity, task execution, clarification, and progress guidance | 2,257 | 565 |",
        "| LangChain TODO middleware | Planning mechanics and TODO lifecycle | 1,076 | 269 |",
        "| Deep Agents filesystem middleware | Filesystem/execute conventions and large-result handling | 1,449 | 363 |",
        "| Assist skills middleware | Skill discovery protocol plus current name/description catalog | 9,631 | 2,408 |",
        "| Assist memory middleware | Empty repository and thread memory plus current persistence guidance | 3,796 | 949 |",
        "| Context rider | Inactive in the canonical comparison scenario | 0 | 0 |",
        "| Provider-bound tool schemas | 30 schemas: framework file/TODO tools, skill loader, async lifecycle, web tools, travel, and URL navigation | 28,037 | 7,010 |",
    ]
    assert all(row in document for row in expected_rows)
    assert "| *System-message total* | | *31,279* | *7,820* |" in document
    assert "| *Bootstrap request + schemas* | Excludes the synthetic user message | *59,316* | *14,829* |" in document
    assert "## Delegating whole tasks" not in current_prompt
    assert "your first call must be `load_skill" not in current_prompt
    assert "TODO bookkeeping is advisory" not in current_prompt
    assert "explicit and self-contained" not in current_prompt
    assert "| Assist role instructions | 13,068 | 11,389 |" in p2_document
    assert "| Complete system message | 31,279 | 29,600 |" in p2_document
    assert "| Provider-bound tool schemas | 28,037 | 28,037 |" in p2_document
    assert "| Bootstrap request plus schemas | 59,316 | 57,637 |" in p2_document
    assert "byte-identical to merged P2 at 29,600" in p2b1_document
    assert "30 provider-bound schemas remain byte-identical at 28,037" \
        in p2b1_document
    assert "2,856,251 bytes with 25 retained" in p2b1_document
    assert "29,527 characters" in p2b2_document
    assert "17,984 characters" in p2b2_document
    assert "2,992,991 bytes" in p2b2_document

    kernel = _named_text_block("proposed-main-bootstrap-kernel")
    headings = [
        "## Boundaries",
        "## Grounding",
        "## Choosing help",
        "## Durable work",
        "## Planning and completion",
    ]
    starts = [kernel.index(heading) for heading in headings]
    section_sizes = [
        starts[0],
        *(right - left for left, right in zip(starts, starts[1:])),
        len(kernel) - starts[-1],
    ]
    assert section_sizes == [123, 621, 551, 492, 615, 466]
    assert len(kernel) == 2_868
