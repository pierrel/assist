"""Deterministic census of Assist's final model requests and capabilities.

The census is deliberately test-side.  The recorder keeps ChatOpenAI's real
provider profile and tool binding, but replaces network inference with scripted
synthetic responses.  No production constructor accepts an observer or census
flag.
"""
from __future__ import annotations

import argparse
import ast
import contextvars
import ctypes
import errno
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from string import Formatter
from typing import Any, Callable, Iterator
from unittest.mock import patch

from fastapi import FastAPI, Request
from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI


SCHEMA_VERSION = 2
FIXED_NOW = "2026-07-30 12:00 UTC"
MAX_RUN_BYTES = 16 * 1024 * 1024
DEFAULT_OUTPUT_ROOT = Path.home() / "deploy" / "assist" / "prompt-census"
REQUIRED_PATHS = {
    "main",
    "delegate",
    "legacy-main",
    "context",
    "research-lead",
    "research-leaf",
    "nested-research-worker",
    "nested-fact-check",
    "nested-report-critique",
    "receptionist",
    "thread-description",
    "capture",
}
EXPECTED_PATHS_BY_SCENARIO = {
    "web-main-core": ("main", "main", "main", "main"),
    "web-main-full": ("main", "main"),
    "web-delegate": ("delegate",),
    "legacy-main": ("legacy-main",),
    "skill-precedence-built-in": ("main", "main"),
    "skill-precedence-embedder": ("main", "main"),
    "context-read-only": ("context", "context", "context"),
    "research-lead": ("research-lead",),
    "research-leaf-provenance": ("research-leaf", "research-leaf"),
    "nested-research-worker": (
        "research-lead", "nested-research-worker", "research-lead"),
    "nested-fact-check": (
        "research-lead", "nested-fact-check", "research-lead"),
    "nested-report-critique": (
        "research-lead", "nested-report-critique", "research-lead"),
    "receptionist": ("receptionist",),
    "thread-description": ("thread-description",),
    "capture": ("capture",),
}
EXPECTED_CALL_COUNTS = {
    scenario: len(paths)
    for scenario, paths in EXPECTED_PATHS_BY_SCENARIO.items()
}

_CURRENT_SCENARIO: contextvars.ContextVar[str] = contextvars.ContextVar(
    "prompt_census_scenario", default="unscoped")
_PROMPT_EVENTS = threading.local()


def _sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _blocks(message) -> list[dict[str, Any]]:
    if message is None:
        return []
    return [dict(block) for block in message.content_blocks]


def _flatten_blocks(blocks: list[dict[str, Any]]) -> str:
    return "".join(str(block.get("text", "")) for block in blocks)


def _block_layout(
    blocks: list[dict[str, Any]], spans: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    layout = []
    cursor = 0
    for index, block in enumerate(blocks):
        text = block["text"]
        end = cursor + len(text)
        source_ids = []
        for span in spans:
            if span["start"] < end and span["end"] > cursor:
                for source_id in (
                        [span["source_id"]] if "source_id" in span
                        else span["source_ids"]):
                    if source_id not in source_ids:
                        source_ids.append(source_id)
        layout.append({
            "index": index,
            "type": block["type"],
            "start": cursor,
            "end": end,
            "text_sha256": _sha(text),
            "source_ids": source_ids,
        })
        cursor = end
    return layout


def _assert_block_layout(
    layout: list[dict[str, Any]], text: str, spans: list[dict[str, Any]]
) -> None:
    if [block["index"] for block in layout] != list(range(len(layout))):
        raise AssertionError("prompt block indices drifted")
    cursor = 0
    reconstructed = []
    for block in layout:
        if block["type"] != "text" or block["start"] != cursor \
                or not cursor <= block["end"] <= len(text):
            raise AssertionError("prompt block boundary drifted")
        value = text[cursor:block["end"]]
        reconstructed.append({"type": "text", "text": value})
        cursor = block["end"]
    if cursor != len(text) or layout != _block_layout(reconstructed, spans):
        raise AssertionError("prompt block provenance drifted")


def _event_list() -> list[dict[str, Any]]:
    events = getattr(_PROMPT_EVENTS, "events", None)
    if events is None:
        events = []
        _PROMPT_EVENTS.events = events
    return events


def _record_transition(owner: str, before, after) -> None:
    before_blocks = _blocks(before)
    after_blocks = _blocks(after)
    _event_list().append({
        "owner": owner,
        "before_sha256": _sha(before_blocks),
        "after_sha256": _sha(after_blocks),
        "before": before_blocks,
        "after": after_blocks,
    })


def _take_events() -> list[dict[str, Any]]:
    events = list(_event_list())
    _PROMPT_EVENTS.events = []
    return events


def _message(message: BaseMessage) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": message.type,
        "content": message.content,
    }
    if getattr(message, "name", None):
        result["name"] = message.name
    if getattr(message, "tool_calls", None):
        result["tool_calls"] = message.tool_calls
    if isinstance(message, ToolMessage):
        result["tool_call_id"] = message.tool_call_id
        result["status"] = message.status
        if message.artifact is not None:
            result["artifact"] = message.artifact
    return result


def _system_prompt(call: dict[str, Any]) -> str:
    messages = [message for message in call["provider_payload"]["messages"]
                if message.get("role") == "system"]
    if len(messages) != 1:
        raise AssertionError(
            f"{call.get('scenario', 'unknown')} has {len(messages)} system messages")
    content = messages[0].get("content")
    if isinstance(content, list):
        return _flatten_blocks(content)
    if not isinstance(content, str):
        raise AssertionError("provider system message is not text")
    return content


def _tool_name(schema: dict[str, Any]) -> str:
    return schema.get("function", {}).get("name", schema.get("name", ""))


def _tool_origin(tool: Any) -> str:
    func = getattr(tool, "func", None)
    module = getattr(func, "__module__", None)
    if module:
        return module
    module = getattr(tool, "__module__", None)
    if module:
        return module
    return type(tool).__module__


def _tool_candidate_name(tool: Any) -> str:
    return getattr(tool, "name", None) or getattr(tool, "__name__", "")


def _is_ordered_subset(values: list[str], candidates: list[str]) -> bool:
    iterator = iter(candidates)
    return all(any(candidate == value for candidate in iterator)
               for value in values)


def _matching_tool_nodes(
    nodes: list[dict[str, Any]], scenario: str, visible: list[str]
) -> list[dict[str, Any]]:
    matching = [
        node for node in nodes
        if node["scenario"] == scenario
        and _is_ordered_subset(visible, node["winners"])
    ]
    if not matching:
        return []
    hidden = min(len(node["winners"]) - len(visible) for node in matching)
    return [node for node in matching
            if len(node["winners"]) - len(visible) == hidden]


@dataclass
class CensusTrace:
    """Mutable test-run sink shared by the recorder and construction probes."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    tool_nodes: list[dict[str, Any]] = field(default_factory=list)
    constructor_prompts: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    faults: list[Exception] = field(default_factory=list)

    def add_template_source(self, package: str, template: str, rendered: str) -> str:
        rendered_sha = _sha(rendered)
        source_id = f"template:{package}:{template}:{rendered_sha[:16]}"
        repo = Path(__file__).resolve().parents[1]
        path = repo / package / "templates" / template
        self.sources[source_id] = {
            "id": source_id,
            "kind": "template",
            "locator": f"{package}/templates/{template}",
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rendered_sha256": rendered_sha,
        }
        return source_id

    def add_skill_source(self, scenario: str, skill: dict[str, Any],
                         source: str, content: str) -> None:
        source_id = f"skill:{scenario}:{skill['name']}:{skill['path']}"
        self.sources[source_id] = {
            "id": source_id,
            "kind": "skill",
            "scenario": scenario,
            "name": skill["name"],
            "source": source,
            "path": skill["path"],
            "description": skill["description"],
            "description_sha256": _sha(skill["description"]),
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "allowed_tools": list(skill.get("allowed_tools", [])),
        }

    def add_fixture_source(self, scenario: str, name: str, value: str) -> str:
        source_id = f"fixture:{scenario}:{name}:{_sha(value)[:16]}"
        self.sources[source_id] = {
            "id": source_id,
            "kind": "synthetic-fixture",
            "scenario": scenario,
            "name": name,
            "value": value,
            "value_sha256": _sha(value),
        }
        return source_id

    def add_constructor_prompt(self, prompt: Any, owner: str) -> None:
        if prompt is None:
            return
        if hasattr(prompt, "text"):
            text = prompt.text
        else:
            text = str(prompt)
        scenario = _CURRENT_SCENARIO.get()
        entries = self.constructor_prompts.setdefault(scenario, [])
        rendered_sha = _sha(text)
        source_id = next((source_id for source_id, source in self.sources.items()
                          if source.get("rendered_sha256") == rendered_sha), None)
        candidate = {"owner": owner, "text": text, "sha256": rendered_sha,
                     "source_id": source_id}
        if candidate not in entries:
            entries.append(candidate)

    def next_call(self, scenario: str) -> int:
        return sum(call["scenario"] == scenario for call in self.calls)


class RecordingChatModel(ChatOpenAI):
    """Offline ChatOpenAI boundary preserving OpenAI harness/tool conversion."""

    _trace: CensusTrace
    _attribute: bool

    def __init__(
        self,
        trace: CensusTrace,
        *,
        attribute: bool = True,
        enable_thinking: bool | None = False,
    ) -> None:
        kwargs = dict(
            model="synthetic-qwen-census",
            api_key="synthetic-census-key",
            base_url="http://127.0.0.1:1/v1",
            temperature=0.1,
            max_retries=0,
        )
        if enable_thinking is False:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False},
            }
        super().__init__(**kwargs)
        object.__setattr__(self, "_trace", trace)
        object.__setattr__(self, "_attribute", attribute)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        scenario = _CURRENT_SCENARIO.get()
        index = self._trace.next_call(scenario)
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        events = _take_events()
        for event in events:
            text = _flatten_blocks(event["after"])
            if event["owner"] == "deepagents.MemoryMiddleware":
                for tag in ("agent_memory", "thread_memory"):
                    match = re.search(
                        rf"<{tag}>\n(.*?)\n</{tag}>", text, re.DOTALL)
                    if match and match.group(1) not in {
                            "(No memory loaded)",
                            "(No thread memory loaded)"}:
                        self._trace.add_fixture_source(
                            scenario, tag, match.group(1))
            if event["owner"] == "assist.ContextRiderMiddleware" \
                    and "SYNTHETIC PLACE" in text:
                self._trace.add_fixture_source(
                    scenario, "context-place", "SYNTHETIC PLACE")
        visible = [_tool_name(schema) for schema in payload.get("tools", [])]
        matching = _matching_tool_nodes(self._trace.tool_nodes, scenario, visible)
        matching_nodes = [node["index"] for node in matching]
        system_blocks = (_blocks(messages[0])
                         if messages and messages[0].type == "system" else [])
        try:
            path = _path_for_call(scenario, index)
        except (KeyError, IndexError) as exc:
            # A graph middleware may turn a model-boundary exception into a
            # response. Preserve an undeclared call as a run-level fault so an
            # apparently complete expected matrix can never hide it.
            fault = AssertionError(
                f"undeclared provider call {scenario}:{index}")
            fault.__cause__ = exc
            self._trace.faults.append(fault)
            path = "undeclared"
        call = {
            "scenario": scenario,
            "call_index": index,
            "path": path,
            "provider_payload": payload,
            "system_blocks": system_blocks,
            "prompt_events": events,
            "visible_tools": visible,
            "matching_tool_nodes": matching_nodes,
        }
        if self._attribute:
            try:
                call["provenance"] = _prompt_provenance(
                    call, self._trace.constructor_prompts.get(scenario, []),
                    [source_id for source_id, source in self._trace.sources.items()
                     if source.get("kind") == "skill"
                     and source.get("scenario") == scenario],
                    self._trace.sources)
            except Exception as exc:
                # Agent middleware may turn model-boundary failures into an AI
                # response.  Preserve the failure here and re-raise it outside
                # the graph after the matrix finishes.
                self._trace.faults.append(exc)
                call["provenance"] = {}
        else:
            call["provenance"] = {}
        call["prompt_events"] = [{
            "owner": event["owner"],
            "before_sha256": event["before_sha256"],
            "after_sha256": event["after_sha256"],
            "before_text_sha256": _sha(_flatten_blocks(event["before"])),
            "after_text_sha256": _sha(_flatten_blocks(event["after"])),
            "characters_before": len(_flatten_blocks(event["before"])),
            "characters_after": len(_flatten_blocks(event["after"])),
            "stringified_prior_blocks": (
                len(event["after"]) == 1
                and isinstance(event["after"][0].get("text"), str)
                and "{'type': 'text'" in event["after"][0]["text"]
            ),
        } for event in events]
        del call["system_blocks"]
        self._trace.calls.append(call)
        return ChatResult(generations=[ChatGeneration(
            message=_scripted_response(scenario, index))])


def _path_for_call(scenario: str, index: int) -> str:
    if scenario == "observer-probe":
        return scenario
    return EXPECTED_PATHS_BY_SCENARIO[scenario][index]


def _call(name: str, args: dict[str, Any], suffix: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{
        "name": name,
        "args": args,
        "id": f"synthetic-call-{suffix}",
    }])


def _scripted_response(scenario: str, index: int) -> AIMessage:
    if scenario == "web-main-core":
        if index == 0:
            return _call("load_skill", {"name": "grounding"}, "grounding-load")
        if index == 1:
            return _call("execute", {"command": "printf synthetic-ok"}, "safe-exec")
        if index == 2:
            return _call("execute", {"command": "git push origin main"}, "git-push")
    if scenario == "web-main-full":
        if index == 0:
            return _call(
                "load_skill", {"name": "send-email"}, "hitl-skill-load")
        return _call("send_email", {
            "to": "synthetic-recipient@example.invalid",
            "subject": "Synthetic approval probe",
            "body": "Synthetic body.",
        }, "hitl")
    if scenario == "context-read-only" and index == 0:
        return _call("write_file", {
            "file_path": "/synthetic-forbidden.txt",
            "content": "SYNTHETIC_FORBIDDEN_WRITE",
        }, "read-only")
    if scenario == "context-read-only" and index == 1:
        return _call("edit_file", {
            "file_path": "/synthetic-forbidden.txt",
            "old_string": "SYNTHETIC_OLD",
            "new_string": "SYNTHETIC_NEW",
        }, "read-only-edit")
    if scenario == "research-leaf-provenance" and index == 0:
        return _call("read_url", {
            "url": "https://unprovenanced.example.invalid/synthetic",
        }, "provenance")
    if scenario.startswith("skill-precedence") and index == 0:
        return _call("load_skill", {"name": "dev"}, f"{scenario}-load")
    targets = {
        "nested-research-worker": "research-agent",
        "nested-fact-check": "fact-check-agent",
        "nested-report-critique": "critique-agent",
    }
    if scenario in targets and index == 0:
        return _call("task", {
            "description": "SYNTHETIC CHILD BRIEF",
            "subagent_type": targets[scenario],
        }, scenario)
    return AIMessage(content=f"SYNTHETIC TERMINAL {scenario} {index}")


_OWNER_SOURCE_IDS = {
    "deepagents.FilesystemMiddleware": "python:deepagents.middleware.filesystem",
    "deepagents.SkillsMiddleware": "python:deepagents.middleware.skills",
    "deepagents.MemoryMiddleware": "python:deepagents.middleware.memory",
    "deepagents.SubAgentMiddleware": "python:deepagents.middleware.subagents",
    "deepagents.AsyncSubAgentMiddleware": "python:deepagents.middleware.async_subagents",
    "deepagents.SummarizationMiddleware": "python:deepagents.middleware.summarization",
    "langchain.TodoListMiddleware": "python:langchain.agents.middleware.todo",
    "assist.ContextRiderMiddleware": "python:assist.middleware.context_rider_middleware",
    "assist.PromptCompositionMiddleware": "python:assist.middleware.prompt_composition",
    "assist.SmallModelSkillsMiddleware": "python:assist.middleware.skills_middleware",
}

_PACKAGED_SKILL_ROOTS = {
    "/skills/": Path("assist/skills"),
    "/main-skills/": Path("assist/main_skills"),
    "/main-guidance-skills/": Path("assist/main_guidance_skills"),
    "/render-skill/": Path("assist/web_skills"),
}

_BUNDLED_OWNER_EXCEPTION = "notify"


def _append_owned(segments: list[dict[str, Any]], value: str,
                  source_ids: list[str]) -> None:
    if not value:
        return
    start = segments[-1]["end"] if segments else 0
    if segments and segments[-1]["source_ids"] == source_ids:
        segments[-1]["end"] += len(value)
    else:
        segments.append({"start": start, "end": start + len(value),
                         "source_ids": source_ids})


def _append_segment(segments: list[dict[str, Any]], start: int, end: int,
                    source_ids: list[str], owner: str | None = None) -> None:
    if start == end:
        return
    if segments and segments[-1]["end"] == start \
            and segments[-1]["source_ids"] == source_ids \
            and segments[-1].get("owner") == owner:
        segments[-1]["end"] = end
        return
    segment: dict[str, Any] = {
        "start": start,
        "end": end,
        "source_ids": source_ids,
    }
    if owner is not None:
        segment["owner"] = owner
    segments.append(segment)


def _repr_character_ranges(value: str) -> tuple[str, list[tuple[int, int]]]:
    rendered = repr(value)
    quote = rendered[0]
    if quote not in {"'", '"'} or rendered[-1] != quote:
        raise AssertionError("unsupported Python string representation")
    interior = rendered[1:-1]
    ranges: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(interior):
        start = cursor
        if interior[cursor] != "\\":
            cursor += 1
        else:
            cursor += 1
            marker = interior[cursor]
            cursor += {"x": 3, "u": 5, "U": 9}.get(marker, 1)
            if marker in "01234567":
                while cursor < min(start + 4, len(interior)) \
                        and interior[cursor] in "01234567":
                    cursor += 1
        ranges.append((start + 1, cursor + 1))
    if len(ranges) != len(value):
        raise AssertionError("Python string representation mapping drifted")
    return rendered, ranges


def _stringified_replacement_segments(
        text: str, before_text: str, prior_spans: list[dict[str, Any]],
        scenario: str, sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rider_id = "python:assist.middleware.context_rider_middleware"
    serialized, separator, _ = text.rpartition("\n\n[Message context:")
    if not separator:
        raise AssertionError(f"ContextRider serialization drifted in {scenario}")
    blocks = ast.literal_eval(serialized)
    if not isinstance(blocks, list) or _flatten_blocks(blocks) != before_text:
        raise AssertionError(f"ContextRider prior blocks drifted in {scenario}")
    segments: list[dict[str, Any]] = []
    cursor = 0
    character_offset = 0
    span_index = 0
    for block in blocks:
        value = str(block.get("text", ""))
        rendered_value, character_ranges = _repr_character_ranges(value)
        value_start = serialized.find(rendered_value, cursor)
        if value_start < 0:
            raise AssertionError(
                f"ContextRider block representation drifted in {scenario}")
        _append_segment(segments, cursor, value_start + 1, [rider_id])
        for local_index, (start, end) in enumerate(character_ranges):
            character_index = character_offset + local_index
            while character_index >= prior_spans[span_index]["end"]:
                span_index += 1
            span = prior_spans[span_index]
            source_ids = ([span["source_id"]] if "source_id" in span
                          else span["source_ids"])
            _append_segment(
                segments,
                value_start + start,
                value_start + end,
                source_ids,
                span["owner"],
            )
        cursor = value_start + len(rendered_value) - 1
        character_offset += len(value)
    suffix_start = cursor
    _append_segment(segments, suffix_start, len(text), [rider_id])
    fixture = next((source for source in sources.values()
                    if source.get("kind") == "synthetic-fixture"
                    and source.get("scenario") == scenario
                    and source.get("name") == "context-place"), None)
    if fixture:
        place_start = text.find(fixture["value"], len(serialized))
        if place_start < 0:
            raise AssertionError(f"context fixture is absent from {scenario}")
        # Split the final rider-owned segment around its synthetic place value.
        segments.pop()
        _append_segment(segments, suffix_start, place_start, [rider_id])
        _append_segment(segments, place_start, place_start + len(fixture["value"]),
                        [fixture["id"]])
        _append_segment(segments, place_start + len(fixture["value"]), len(text),
                        [rider_id])
    return segments


def _render_owned_template(
        template: str, template_source_id: str,
        fields: dict[str, dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    parts: list[str] = []
    segments: list[dict[str, Any]] = []
    for literal, field_name, format_spec, conversion in Formatter().parse(template):
        _append_owned(segments, literal, [template_source_id])
        parts.append(literal)
        if field_name is None:
            continue
        if format_spec or conversion:
            raise AssertionError("prompt census does not support formatted fields")
        field = fields[field_name]
        value = field["value"]
        offset = segments[-1]["end"] if segments else 0
        nested = field.get("segments")
        if nested is not None:
            if nested and (nested[0]["start"] != 0
                           or nested[-1]["end"] != len(value)):
                raise AssertionError(f"field {field_name} has incomplete ownership")
            for segment in nested:
                segments.append({
                    "start": offset + segment["start"],
                    "end": offset + segment["end"],
                    "source_ids": segment["source_ids"],
                })
        else:
            _append_owned(segments, value, field["source_ids"])
        parts.append(value)
    return "".join(parts), segments


def _fixture_or_literal(sources: dict[str, dict[str, Any]], scenario: str,
                        name: str, literal: str,
                        source_id: str) -> dict[str, Any]:
    matches = [source for source in sources.values()
               if source.get("kind") == "synthetic-fixture"
               and source.get("scenario") == scenario
               and source.get("name") == name]
    if not matches:
        return {"value": literal, "id": source_id}
    if len(matches) != 1:
        raise AssertionError(f"{scenario} has {len(matches)} {name} fixtures")
    return matches[0]


def _content_segments(owner: str, text: str, scenario: str,
                      skill_source_ids: list[str],
                      sources: dict[str, dict[str, Any]], *,
                      before_text: str = "",
                      prior_spans: list[dict[str, Any]] | None = None,
                      operation: str = "append") -> list[dict[str, Any]]:
    if not text:
        return []
    if owner == "assist.PromptCompositionMiddleware" and operation == "replace":
        if prior_spans is None:
            raise AssertionError("prompt composition replacement lacks prior provenance")
        base = next((span for span in prior_spans
                     if span.get("source_id") == "python:deepagents.graph.BASE_AGENT_PROMPT"),
                    None)
        core = next((span for span in prior_spans
                     if span.get("owner") == "assist.agent template"), None)
        if base is None or core is None:
            raise AssertionError("prompt composition static sources are absent")
        base_text = before_text[base["start"]:base["end"]]
        core_text = before_text[core["start"]:core["end"]]
        suffix = before_text[base["end"]:]
        if text != f"{base_text}\n\n{core_text}{suffix}":
            raise AssertionError(f"prompt composition replacement drifted in {scenario}")
        segments = [
            {"start": 0, "end": len(base_text),
             "source_ids": [base["source_id"]],
             "owner": "deepagents.graph.BASE_AGENT_PROMPT"},
            {"start": len(base_text), "end": len(base_text) + 2,
             "source_ids": ["python:deepagents.graph"],
             "owner": "framework prompt composer"},
            {"start": len(base_text) + 2, "end": len(text),
             "source_ids": [core["source_id"]],
             "owner": "assist.agent template"},
        ]
        cursor = len(base_text) + 2 + len(core_text)
        for span in prior_spans:
            if span["start"] < base["end"]:
                continue
            value = before_text[span["start"]:span["end"]]
            span_source_ids = ([span["source_id"]] if "source_id" in span
                               else span["source_ids"])
            segments.append({"start": cursor, "end": cursor + len(value),
                             "source_ids": span_source_ids,
                             **({"owner": span["owner"]} if "owner" in span else {})})
            cursor += len(value)
        # The core segment ends before later framework-owned blocks.
        segments[2]["end"] = len(base_text) + 2 + len(core_text)
        if cursor != len(text):
            raise AssertionError("prompt composition suffix attribution drifted")
        return segments
    if owner == "deepagents.SkillsMiddleware":
        from assist.middleware.skills_middleware import SMALL_MODEL_SKILLS_PROMPT
        skills_prompt = SMALL_MODEL_SKILLS_PROMPT
        prompt_source = (
            "python:assist.middleware.skills_middleware.SMALL_MODEL_SKILLS_PROMPT")
        formatter_id = (
            "python:assist.middleware.skills_middleware."
            "SmallModelSkillsMiddleware._format_skills_list")
        listing_parts: list[str] = []
        listing_segments: list[dict[str, Any]] = []
        for index, source_id in enumerate(skill_source_ids):
            skill = sources[source_id]
            if index:
                _append_owned(listing_segments, "\n", [formatter_id])
                listing_parts.append("\n")
            pieces = (
                ("- **", [formatter_id]),
                (skill["name"], [source_id]),
                ("**: ", [formatter_id]),
                (skill["description"], [source_id]),
            )
            for value, source_ids in pieces:
                _append_owned(listing_segments, value, source_ids)
                listing_parts.append(value)
        if not listing_parts:
            listing_parts.append("(No skills available.)")
            _append_owned(listing_segments, listing_parts[0], [formatter_id])
        listing = "".join(listing_parts)
        tools_match = re.search(
            r"Tools available for this response: ([^\n]+)\.", text)
        if tools_match is None:
            raise AssertionError(
                f"retained tool guidance is absent from {scenario}")
        rendered, segments = _render_owned_template(
            skills_prompt,
            prompt_source,
            {
                "skills_list": {"value": listing, "segments": listing_segments},
                "tools_available": {
                    "value": tools_match.group(1),
                    "source_ids": [
                        "python:assist.middleware.skills_middleware"],
                },
            },
        )
    elif owner == "deepagents.MemoryMiddleware":
        from assist.middleware.memory_middleware import (
            SMALL_MODEL_MEMORY_PROMPT,
            THREAD_MEMORY_PROMPT,
        )
        memory_prompt = SMALL_MODEL_MEMORY_PROMPT
        memory_source = (
            "python:assist.middleware.memory_middleware.SMALL_MODEL_MEMORY_PROMPT")
        thread_prompt = THREAD_MEMORY_PROMPT
        thread_source = (
            "python:assist.middleware.memory_middleware.THREAD_MEMORY_PROMPT")
        formatter_id = (
            "python:assist.middleware.memory_middleware."
            "SmallModelMemoryMiddleware._format_agent_memory")
        agent_memory = _fixture_or_literal(
            sources, scenario, "agent_memory", "(No memory loaded)", formatter_id)
        memory_path_match = re.search(r"(?:file|at) `([^`]+)`", text)
        if memory_path_match is None:
            raise AssertionError(f"memory path is absent from {scenario}")
        memory_path = memory_path_match.group(1)
        repo_prompt, repo_segments = _render_owned_template(
            memory_prompt,
            memory_source,
            {
                "agent_memory": {"value": agent_memory["value"],
                                 "source_ids": [agent_memory["id"]]},
                "memory_path": {"value": memory_path,
                                "source_ids": ["python:assist.agent.create_agent"]},
            },
        )
        if "<thread_memory>" not in text:
            rendered, segments = repo_prompt, repo_segments
        else:
            thread_memory = _fixture_or_literal(
                sources, scenario, "thread_memory",
                "(No thread memory loaded)", formatter_id)
            thread_path_match = re.search(
                r"\n`([^`]+)` belongs only to this thread", text)
            if thread_path_match is None:
                raise AssertionError(f"thread memory path is absent from {scenario}")
            rendered, segments = _render_owned_template(
                thread_prompt,
                thread_source,
                {
                    "repo_prompt": {"value": repo_prompt,
                                    "segments": repo_segments},
                    "thread_memory": {"value": thread_memory["value"],
                                      "source_ids": [thread_memory["id"]]},
                    "repo_memory_path": {"value": memory_path,
                                         "source_ids": ["python:assist.agent.create_agent"]},
                    "thread_memory_path": {"value": thread_path_match.group(1),
                                           "source_ids": ["python:assist.agent.create_agent"]},
                },
            )
    elif owner == "assist.ContextRiderMiddleware" and operation == "replace":
        if not prior_spans:
            raise AssertionError("ContextRider replacement lacks prior provenance")
        return _stringified_replacement_segments(
            text, before_text, prior_spans, scenario, sources)
    else:
        default_source = [_OWNER_SOURCE_IDS[owner]]
        ranges: list[tuple[int, int, list[str]]] = []
        if owner == "assist.ContextRiderMiddleware":
            fixture = next((source for source in sources.values()
                            if source.get("kind") == "synthetic-fixture"
                            and source.get("scenario") == scenario
                            and source.get("name") == "context-place"), None)
            if fixture:
                start = text.find(fixture["value"])
                if start < 0:
                    raise AssertionError(f"context fixture is absent from {scenario}")
                ranges.append((start, start + len(fixture["value"]), [fixture["id"]]))
        segments = []
        cursor = 0
        for start, end, source_ids in ranges:
            _append_owned(segments, text[cursor:start], default_source)
            _append_owned(segments, text[start:end], source_ids)
            cursor = end
        _append_owned(segments, text[cursor:], default_source)
        rendered = text
    if rendered != text and text.endswith(rendered):
        separator = text[:len(text) - len(rendered)]
        if separator and not separator.strip():
            segments = [{
                "start": 0,
                "end": len(separator),
                "source_ids": [_OWNER_SOURCE_IDS[owner]],
            }, *({
                "start": len(separator) + segment["start"],
                "end": len(separator) + segment["end"],
                "source_ids": segment["source_ids"],
            } for segment in segments)]
            rendered = separator + rendered
    if rendered != text:
        raise AssertionError(f"{owner} rendering drifted in {scenario}")
    return segments


def _prompt_provenance(call: dict[str, Any],
                       constructor_prompts: list[dict[str, Any]],
                       skill_source_ids: list[str],
                       sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Validate the observed middleware transition chain and source the initial text."""
    events = call["prompt_events"]
    blocks = call["system_blocks"]
    if events:
        for left, right in zip(events, events[1:]):
            if left["after_sha256"] != right["before_sha256"]:
                raise AssertionError(
                    f"prompt contribution chain broke in {call['scenario']} call "
                    f"{call['call_index']}: {left['owner']} -> {right['owner']}")
        if events[-1]["after_sha256"] != _sha(blocks):
            raise AssertionError(
                f"final prompt was not observed in {call['scenario']} call "
                f"{call['call_index']}")
        initial = events[0]["before"]
    else:
        initial = blocks

    initial_text = _flatten_blocks(initial)
    spans: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for item in sorted(constructor_prompts, key=lambda value: len(value["text"]), reverse=True):
        start = initial_text.find(item["text"])
        if start < 0:
            continue
        end = start + len(item["text"])
        overlapping = [span for span in spans
                       if start < span["end"] and end > span["start"]]
        if overlapping and any(
                span["owner"] != item["owner"]
                or span.get("source_id") != item["source_id"]
                for span in overlapping):
            owners = sorted({item["owner"], *(
                span["owner"] for span in overlapping)})
            raise AssertionError(
                f"ambiguous constructor prompt ownership in {call['scenario']}: "
                f"{', '.join(owners)}")
        if overlapping:
            continue
        spans.append({"start": start, "end": end, "owner": item["owner"],
                      "source_id": item["source_id"],
                      "rendered_sha256": item["sha256"]})
        occupied.append((start, end))

    # Deep Agents composes the caller prompt and its base prompt before middleware.
    from deepagents.graph import BASE_AGENT_PROMPT
    start = initial_text.find(BASE_AGENT_PROMPT)
    if start >= 0 and not any(
            start < prior_end and start + len(BASE_AGENT_PROMPT) > prior_start
            for prior_start, prior_end in occupied):
        spans.append({
            "start": start,
            "end": start + len(BASE_AGENT_PROMPT),
            "owner": "deepagents.graph.BASE_AGENT_PROMPT",
            "source_id": "python:deepagents.graph.BASE_AGENT_PROMPT",
        })
        occupied.append((start, start + len(BASE_AGENT_PROMPT)))

    spans.sort(key=lambda span: span["start"])
    cursor = 0
    covered: list[dict[str, Any]] = []
    for span in spans:
        if span["start"] > cursor:
            gap = initial_text[cursor:span["start"]]
            if gap.strip():
                raise AssertionError(
                    f"unattributed initial prompt text in {call['scenario']}: {gap[:80]!r}")
            covered.append({"start": cursor, "end": span["start"],
                            "owner": "framework prompt composer",
                            "source_id": "python:deepagents.graph"})
        covered.append(span)
        cursor = span["end"]
    if cursor < len(initial_text):
        gap = initial_text[cursor:]
        if gap.strip():
            raise AssertionError(
                f"unattributed trailing prompt text in {call['scenario']}: {gap[:80]!r}")
        covered.append({"start": cursor, "end": len(initial_text),
                        "owner": "framework prompt composer",
                        "source_id": "python:deepagents.graph"})

    final_spans = [dict(span) for span in covered]
    current_text = initial_text
    transitions = []
    for event in events:
        before_text = _flatten_blocks(event["before"])
        after_text = _flatten_blocks(event["after"])
        if before_text != current_text:
            raise AssertionError(
                f"prompt text chain broke in {call['scenario']} at {event['owner']}")
        if after_text.startswith(before_text):
            added_text = after_text[len(before_text):]
            operation = "append" if added_text else "unchanged"
            exact_change = added_text
        else:
            operation = "replace"
            exact_change = after_text
        content_segments = _content_segments(
            event["owner"], exact_change, call["scenario"],
            skill_source_ids, sources, before_text=before_text,
            prior_spans=final_spans, operation=operation)
        content_source_ids = list(dict.fromkeys(
            source_id
            for segment in content_segments
            for source_id in segment["source_ids"]))
        if operation == "append":
            final_spans = [*final_spans, *({
                "start": len(before_text) + segment["start"],
                "end": len(before_text) + segment["end"],
                "owner": event["owner"],
                "source_ids": segment["source_ids"],
            } for segment in content_segments)]
        elif operation == "replace":
            final_spans = [{
                "start": segment["start"],
                "end": segment["end"],
                "owner": segment.get("owner", event["owner"]),
                "source_ids": segment["source_ids"],
            } for segment in content_segments]
        transitions.append({
            "owner": event["owner"],
            "operation": operation,
            "before_sha256": event["before_sha256"],
            "after_sha256": event["after_sha256"],
            "before_text_sha256": _sha(before_text),
            "after_text_sha256": _sha(after_text),
            "characters_before": len(before_text),
            "characters_after": len(after_text),
            "exact_change": exact_change,
            "source_ids": content_source_ids,
            "content_segments": content_segments,
        })
        current_text = after_text

    final_text = _flatten_blocks(blocks)
    if current_text != final_text:
        raise AssertionError(f"final prompt text was not observed in {call['scenario']}")
    if final_spans:
        if final_spans[0]["start"] != 0 or final_spans[-1]["end"] != len(final_text):
            raise AssertionError(f"final prompt attribution is incomplete in {call['scenario']}")
        if any(left["end"] != right["start"]
               for left, right in zip(final_spans, final_spans[1:])):
            raise AssertionError(f"final prompt attribution has gaps in {call['scenario']}")
    elif final_text:
        raise AssertionError(f"final prompt attribution is empty in {call['scenario']}")

    return {
        "initial_sha256": _sha(initial),
        "initial_text_sha256": _sha(initial_text),
        "initial_characters": len(initial_text),
        "initial_text": initial_text,
        "initial_spans": covered,
        "initial_block_layout": _block_layout(initial, covered),
        "transitions": transitions,
        "final_spans": final_spans,
        "final_block_layout": _block_layout(blocks, final_spans),
        "final_sha256": _sha(blocks),
        "final_text_sha256": _sha(final_text),
    }


class _SyntheticContainer:
    id = "synthetic-container-000000000000"

    def exec_run(self, command, **_kwargs):
        return 0, b"SYNTHETIC_CONTAINER_OK"


class SyntheticSandbox(FilesystemBackend, SandboxBackendProtocol):
    """Filesystem-backed production sandbox shape; no Docker or shell execution."""

    work_dir = "/workspace"
    native_agent_dir = True

    def __init__(self, root_dir: str):
        super().__init__(root_dir=root_dir, virtual_mode=True)
        self.container = _SyntheticContainer()
        self.commands: list[str] = []

    @property
    def id(self) -> str:
        return "synthetic-sandbox"

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        self.commands.append(command)
        return ExecuteResponse(output="SYNTHETIC_EXECUTE_OK", exit_code=0)


def _wrap_append(owner: str, original: Callable) -> Callable:
    def traced(system_message, text):
        result = original(system_message, text)
        _record_transition(owner, system_message, result)
        return result
    return traced


def _wrap_model_prompt(owner: str, original: Callable) -> Callable:
    def traced(self, request, handler):
        def observe(inner_request):
            _record_transition(owner, request.system_message, inner_request.system_message)
            return handler(inner_request)
        return original(self, request, observe)
    return traced


@contextmanager
def _instrument(trace: CensusTrace) -> Iterator[None]:
    """Install test-process construction/provenance observers, then restore them."""
    import deepagents.middleware.async_subagents as async_subagents_mod
    import deepagents.middleware.filesystem as filesystem_mod
    import deepagents.middleware.memory as memory_mod
    import deepagents.middleware.skills as skills_mod
    import deepagents.middleware.subagents as subagents_mod
    import deepagents.middleware.summarization as summarization_mod
    import langchain.agents.factory as factory_mod
    import langchain.agents.middleware.todo as todo_mod
    import assist.agent as assist_agent_mod
    import assist.middleware.context_rider_middleware as rider_mod
    import assist.middleware.prompt_composition as composition_mod
    import assist.middleware.skills_middleware as assist_skills_mod
    import edd.agent as capture_agent_mod
    from deepagents import create_deep_agent as real_create_deep_agent

    def prompt_renderer(package: str, original: Callable) -> Callable:
        def wrapped(template: str, **kwargs):
            rendered = original(template, **kwargs)
            trace.add_template_source(package, template, rendered)
            return rendered
        return wrapped

    def constructor(owner: str):
        def wrapped(*args, **kwargs):
            trace.add_constructor_prompt(kwargs.get("system_prompt"), owner)
            for subagent in kwargs.get("subagents") or []:
                if isinstance(subagent, dict):
                    trace.add_constructor_prompt(
                        subagent.get("system_prompt"),
                        f"{owner}.subagent.{subagent.get('name', 'unknown')}")
            return real_create_deep_agent(*args, **kwargs)
        return wrapped

    original_tool_node = factory_mod.ToolNode
    original_skills_before = skills_mod.SkillsMiddleware.before_agent

    def traced_skills_before(self, state, runtime, config):
        backend = self._get_backend(state, runtime, config)
        for source in self.sources:
            listing = backend.ls(source)
            error = getattr(listing, "error", None)
            if error:
                raise AssertionError(
                    f"could not enumerate skill source {source}: {error}")
            entries = getattr(listing, "entries", listing) or []
            for entry in entries:
                if not entry.get("is_dir"):
                    continue
                skill_path = entry["path"].rstrip("/") + "/SKILL.md"
                read_result = backend.read(skill_path, limit=1_000_000)
                if read_result.error or read_result.file_data is None:
                    raise AssertionError(
                        f"could not read candidate skill {skill_path}: "
                        f"{read_result.error}")
                metadata = skills_mod._parse_skill_metadata(
                    read_result.file_data["content"], skill_path,
                    Path(entry["path"]).name)
                if metadata is None:
                    raise AssertionError(
                        f"candidate skill metadata did not parse: {skill_path}")
        update = original_skills_before(self, state, runtime, config)
        if update and update.get("skills_metadata"):
            for skill in update["skills_metadata"]:
                read_result = backend.read(skill["path"], limit=1_000_000)
                if read_result.error or read_result.file_data is None:
                    raise AssertionError(
                        f"could not fingerprint skill {skill['path']}: "
                        f"{read_result.error}")
                source = next((source for source in reversed(self.sources)
                               if skill["path"].startswith(source.rstrip("/") + "/")),
                              "unresolved")
                trace.add_skill_source(
                    _CURRENT_SCENARIO.get(), skill, source,
                    read_result.file_data["content"])
        return update

    def tracing_tool_node(tools, *args, **kwargs):
        node = original_tool_node(tools, *args, **kwargs)
        entry = {
            "index": len(trace.tool_nodes),
            "scenario": _CURRENT_SCENARIO.get(),
            "candidates": [{
                "name": _tool_candidate_name(tool),
                "origin": _tool_origin(tool),
            } for tool in tools],
            "winners": list(node.tools_by_name),
        }
        trace.tool_nodes.append(entry)
        return node

    modules = [
        (filesystem_mod, "deepagents.FilesystemMiddleware"),
        (skills_mod, "deepagents.SkillsMiddleware"),
        (memory_mod, "deepagents.MemoryMiddleware"),
        (subagents_mod, "deepagents.SubAgentMiddleware"),
        (async_subagents_mod, "deepagents.AsyncSubAgentMiddleware"),
        (summarization_mod, "deepagents.SummarizationMiddleware"),
    ]
    with ExitStack() as stack:
        for module, owner in modules:
            stack.enter_context(patch.object(
                module, "append_to_system_message",
                _wrap_append(owner, module.append_to_system_message)))
        stack.enter_context(patch.object(
            todo_mod.TodoListMiddleware, "wrap_model_call",
            _wrap_model_prompt("langchain.TodoListMiddleware",
                               todo_mod.TodoListMiddleware.wrap_model_call)))
        stack.enter_context(patch.object(
            rider_mod.ContextRiderMiddleware, "wrap_model_call",
            _wrap_model_prompt("assist.ContextRiderMiddleware",
                               rider_mod.ContextRiderMiddleware.wrap_model_call)))
        stack.enter_context(patch.object(
            composition_mod.PromptCompositionMiddleware, "wrap_model_call",
            _wrap_model_prompt("assist.PromptCompositionMiddleware",
                               composition_mod.PromptCompositionMiddleware.wrap_model_call)))
        stack.enter_context(patch.object(factory_mod, "ToolNode", tracing_tool_node))
        stack.enter_context(patch.object(
            skills_mod.SkillsMiddleware, "before_agent", traced_skills_before))
        stack.enter_context(patch.object(
            assist_agent_mod, "base_prompt_for",
            prompt_renderer("assist", assist_agent_mod.base_prompt_for)))
        stack.enter_context(patch.object(
            capture_agent_mod, "base_prompt_for",
            prompt_renderer("edd", capture_agent_mod.base_prompt_for)))
        stack.enter_context(patch.object(
            assist_agent_mod, "create_deep_agent", constructor("assist.agent template")))
        stack.enter_context(patch.object(
            capture_agent_mod, "create_deep_agent", constructor("edd.capture template")))
        yield


@contextmanager
def _scenario(name: str) -> Iterator[None]:
    token = _CURRENT_SCENARIO.set(name)
    _PROMPT_EVENTS.events = []
    try:
        yield
    finally:
        _take_events()
        _CURRENT_SCENARIO.reset(token)


def _write_skill(root: Path, name: str, description: str, marker: str,
                 *, allowed_tools: str | None = None) -> None:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    allowed_line = (f"allowed-tools: {allowed_tools}\n"
                    if allowed_tools else "")
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n"
        f"{allowed_line}---\n\n{marker}\n",
        encoding="utf-8",
    )


def _web_toolset(root: Path, *, full: bool) -> list:
    from assist.events.email import email_tools
    from assist.events.notify import notify_tools
    from assist.events.store import SubscriptionStore
    from assist.events.tools import subscription_tools
    from assist.schedule.store import ScheduleStore
    from assist.schedule.tools import schedule_tools

    root.mkdir(parents=True, exist_ok=True)
    tools = (
        schedule_tools(ScheduleStore(str(root)))
        + subscription_tools(SubscriptionStore(str(root)))
        + notify_tools(lambda _tid: None)
        + email_tools()
    )
    if full:
        from assist.egress.store import EgressStore
        from assist.egress.tools import egress_tools
        from assist.geo.catalog import Catalog
        from assist.geo.proposals import ProposalStore
        from assist.geo.registry import RegionRegistry
        from assist.geo.tools import geo_tools

        geo_root = root / "synthetic-geo"
        geo_root.mkdir()
        egress_root = root / "synthetic-egress"
        tools += geo_tools(
            RegionRegistry(str(geo_root)), Catalog(str(geo_root)),
            ProposalStore(str(geo_root)))
        tools += egress_tools(
            EgressStore(str(egress_root)), frozenset({"allowed.example.invalid"}))
    return tools


@contextmanager
def _web_config(tools: list) -> Iterator[None]:
    import assist.thread_manager as manager_mod
    from assist.events.email import EMAIL_INTERRUPT_ON

    old_tools = manager_mod._web_tools
    old_interrupt = manager_mod._web_interrupt_on
    manager_mod.set_web_tools(tools)
    manager_mod.set_web_interrupt_on(EMAIL_INTERRUPT_ON)
    try:
        yield
    finally:
        manager_mod.set_web_tools(old_tools)
        manager_mod.set_web_interrupt_on(old_interrupt)


def _invoke_agent(agent, scenario: str, config: dict[str, Any] | None = None):
    config = config or {"configurable": {"thread_id": f"synthetic-{scenario}"}}
    return agent.invoke(
        {"messages": [HumanMessage(content=f"SYNTHETIC USER {scenario}")]},
        config,
    )


def _tool_messages(result, *, name: str | None = None) -> list[dict[str, Any]]:
    return [
        _message(message)
        for message in result.get("messages", [])
        if isinstance(message, ToolMessage) and (name is None or message.name == name)
    ]


def _invoke_web(trace: CensusTrace, root: Path, *, full: bool,
                delegate: bool = False) -> dict[str, Any]:
    from assist.context_rider import CONTEXT_RIDER_KEY, ContextRider
    from assist.thread_manager import ThreadManager
    from langgraph.checkpoint.memory import InMemorySaver

    name = "web-delegate" if delegate else ("web-main-full" if full else "web-main-core")
    manager = ThreadManager(str(root / name / "threads"))
    manager._model = RecordingChatModel(trace)
    # ThreadManager.get is the production composition seam.  Its SQLite saver is
    # irrelevant to request composition and can self-deadlock under a synthetic
    # rapid multi-tool loop (the same chained-put shape production avoids with
    # sync durability).  Keep the real constructor, replace only persistence.
    manager.checkpointer = InMemorySaver()
    tid = f"synthetic-{name}"
    thread_dir = Path(manager.thread_dir(tid))
    working_dir = Path(manager.make_default_working_dir(str(thread_dir)))
    sandbox_root = root / name / "sandbox"
    sandbox_root.mkdir(parents=True)
    sandbox = SyntheticSandbox(str(sandbox_root))
    # Production delegates do not mount the parent thread's private /agent
    # directory; retaining it would suppress their synchronous specialists.
    sandbox.native_agent_dir = not delegate
    configurable = None
    if full and not delegate:
        (sandbox_root / "workspace").mkdir()
        (sandbox_root / "workspace" / "AGENTS.md").write_text(
            "SYNTHETIC_REPOSITORY_MEMORY\n", encoding="utf-8")
        (sandbox_root / "agent").mkdir()
        (sandbox_root / "agent" / "memory.md").write_text(
            "SYNTHETIC_THREAD_MEMORY\n", encoding="utf-8")
        _write_skill(
            sandbox_root / ".claude" / "skills",
            "synthetic-domain", "SYNTHETIC DOMAIN DESCRIPTION",
            "SYNTHETIC_DOMAIN_BODY",
            allowed_tools="synthetic_external_tool")
        configurable = {CONTEXT_RIDER_KEY: ContextRider(
            sent_at=datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc),
            tz="UTC", place_label="SYNTHETIC PLACE")}

    tools = _web_toolset(root / name / "stores", full=full)
    email_delivery_attempted = False

    def block_email_delivery():
        nonlocal email_delivery_attempted
        email_delivery_attempted = True
        raise AssertionError("HITL probe reached the email delivery body")

    try:
        with ExitStack() as stack:
            stack.enter_context(_web_config(tools))
            stack.enter_context(_scenario(name))
            if full:
                stack.enter_context(patch(
                    "assist.events.email._config", side_effect=block_email_delivery))
            thread = manager.get(
                tid, working_dir=str(working_dir), sandbox_backend=sandbox,
                configurable=configurable,
                assistant_id="delegate-agent" if delegate else "general-agent")
            result = _invoke_agent(thread.agent, name, thread.runconfig)
    finally:
        manager.close()
    return {
        "sandbox_commands": sandbox.commands,
        "tool_messages": _tool_messages(result),
        "interrupted": bool(isinstance(result, dict) and result.get("__interrupt__")),
        "email_delivery_attempted": email_delivery_attempted,
    }


def _invoke_legacy(trace: CensusTrace, root: Path) -> dict[str, Any]:
    from assist.agent import create_agent
    from assist.spec import AgentSpec
    with _scenario("legacy-main"):
        agent = create_agent(RecordingChatModel(trace), str(root / "legacy"),
                             spec=AgentSpec())
        _invoke_agent(agent, "legacy")
    return {}


def _invoke_skill_precedence(trace: CensusTrace, root: Path,
                             *, embedder: bool) -> dict[str, Any]:
    from assist.agent import create_agent
    from assist.async_subagents import async_task_tools
    from assist.spec import AgentSpec
    from deepagents.backends import FilesystemBackend

    name = "skill-precedence-embedder" if embedder else "skill-precedence-built-in"
    work = root / name / "workspace"
    _write_skill(work / ".claude" / "skills", "dev",
                 "SYNTHETIC DOMAIN DEV DESCRIPTION", "SYNTHETIC_DOMAIN_DEV_BODY")
    sources = {}
    if embedder:
        extra = root / name / "embedder-skills"
        _write_skill(extra, "dev", "SYNTHETIC EMBEDDER DEV DESCRIPTION",
                     "\x1b[31mSYNTHETIC_EMBEDDER_DEV_BODY\x1b[0m")
        sources["/synthetic-embedder-skills/"] = FilesystemBackend(
            root_dir=str(extra), virtual_mode=True)
    with _scenario(name):
        agent = create_agent(
            RecordingChatModel(trace), str(work),
            spec=AgentSpec(skill_sources=sources,
                           async_subagent_tools=async_task_tools))
        result = _invoke_agent(agent, name)
    messages = _tool_messages(result, name="load_skill")
    return {
        "loaded_skill_bodies": [str(message["content"]) for message in messages],
        "load_artifacts": [message["artifact"] for message in messages],
    }


def _invoke_context(trace: CensusTrace, root: Path) -> dict[str, Any]:
    from assist.agent import create_context_agent
    with _scenario("context-read-only"):
        agent = create_context_agent(RecordingChatModel(trace), str(root / "context"))
        result = _invoke_agent(agent, "context")
    return {"tool_messages": _tool_messages(result)}


def _invoke_research(trace: CensusTrace, root: Path, *, leaf: bool,
                     scenario: str) -> dict[str, Any]:
    from assist.agent import create_research_agent
    work = root / scenario
    work.mkdir()
    with _scenario(scenario):
        agent = create_research_agent(
            RecordingChatModel(trace), str(work), leaf=leaf)
        result = _invoke_agent(agent, scenario)
    return {"tool_messages": _tool_messages(result)}


def _invoke_receptionist(trace: CensusTrace) -> dict[str, Any]:
    from assist.receptionist import create_receptionist, receptionist_tools
    from assist.promptable import base_prompt_for

    class EmptyCatalog:
        def entries(self, limit=None):
            return []

    model = RecordingChatModel(trace)
    tools = receptionist_tools(EmptyCatalog(), ["SYNTHETIC PROJECT"],
                               lambda _tid: None, lambda _domain, _message: None)
    with _scenario("receptionist"), \
            patch("assist.model_manager.select_assistant_model", return_value=model):
        prompt = base_prompt_for("receptionist_system.md.j2",
                                 domains=["SYNTHETIC PROJECT"])
        trace.add_template_source("assist", "receptionist_system.md.j2", prompt)
        trace.add_constructor_prompt(prompt, "assist.receptionist template")
        agent = create_receptionist(tools, ["SYNTHETIC PROJECT"])
        _invoke_agent(agent, "receptionist")
    return {}


def _invoke_description(trace: CensusTrace) -> dict[str, Any]:
    from assist.promptable import base_prompt_for
    from assist.thread import Thread
    thread = object.__new__(Thread)
    thread.model = RecordingChatModel(trace)
    thread.get_messages = lambda: [
        {"role": "user", "content": "SYNTHETIC USER description"},
        {"role": "assistant", "content": "SYNTHETIC ASSISTANT description"},
    ]
    with _scenario("thread-description"):
        prompt = base_prompt_for("deepagents/describe_system.md.j2")
        trace.add_template_source("assist", "deepagents/describe_system.md.j2", prompt)
        trace.add_constructor_prompt(prompt, "assist.thread-description template")
        description = thread.description()
    return {"description": description}


def _invoke_capture(trace: CensusTrace, root: Path) -> dict[str, Any]:
    from edd.capture import capture_conversation

    class SyntheticThread:
        thread_id = "synthetic-capture-thread"

        def __init__(self, threads_root: Path):
            self.threads_root = str(threads_root)

        def get_messages(self):
            return [
                {"role": "user", "content": "SYNTHETIC CAPTURE USER REQUEST"},
                {"role": "assistant", "content": "SYNTHETIC CAPTURE RESPONSE"},
            ]

        def description(self):
            return "synthetic capture description"

    work = root / "capture"
    work.mkdir()
    threads_root = work / "thread"
    (threads_root / "domain").mkdir(parents=True)
    (threads_root / "domain" / "synthetic.txt").write_text(
        "SYNTHETIC CAPTURE DOMAIN\n", encoding="utf-8")
    with _scenario("capture"), \
            patch("edd.capture.select_chat_model",
                  return_value=RecordingChatModel(
                      trace, enable_thinking=None)), \
            patch("edd.capture.sanitize_dirname",
                  return_value="synthetic-capture-output"):
        capture_conversation(
            SyntheticThread(threads_root),
            "SYNTHETIC CAPTURE REASON",
            str(work / "improvements"),
        )
    return {}


def _invoke_async_return_contract() -> dict[str, str]:
    """Exercise Assist's real async launch/check tools against an in-process API."""
    from types import SimpleNamespace

    import assist.async_subagents as async_mod

    app = FastAPI()
    tasks: dict[str, dict[str, Any]] = {}

    @app.post("/threads")
    async def create_thread(request: Request):
        body = await request.json()
        task_id = body["thread_id"]
        tasks[task_id] = {
            "task_id": task_id,
            "agent_name": None,
            "description": None,
            "status": "pending",
            "run_id": None,
            "parent_thread_id": body["metadata"]["parent_thread_id"],
            "created_at": "2026-07-30T12:00:00Z",
            "updated_at": "2026-07-30T12:00:00Z",
        }
        return {"thread_id": task_id, "status": "idle", "values": {}}

    @app.post("/threads/{task_id}/runs")
    async def create_run(task_id: str, request: Request):
        body = await request.json()
        task = tasks[task_id]
        task.update({
            "agent_name": body["assistant_id"],
            "description": body["input"]["messages"][0]["content"],
            "run_id": "synthetic-async-run",
        })
        return {
            "run_id": task["run_id"],
            "thread_id": task_id,
            "assistant_id": task["agent_name"],
            "status": "pending",
        }

    @app.get("/threads/{task_id}")
    async def get_thread(task_id: str):
        return {
            "thread_id": task_id,
            "status": "idle",
            "values": {"async_task": tasks[task_id]},
        }

    tools = {tool.name: tool for tool in async_mod.async_task_tools}
    context = async_mod.AsyncTaskContext(
        "synthetic-parent", "synthetic-parent-run", "synthetic-parent-work")
    runtime = SimpleNamespace(tool_call_id="synthetic-async-start")
    with patch.object(async_mod, "_APP", app), async_mod.async_task_context(context):
        launch = tools["start_async_task"].func(
            "SYNTHETIC RESEARCH BRIEF", "research-agent", runtime)
        task_id = launch.split("task_id: ", 1)[1].split(".", 1)[0]
        tasks[task_id].update({
            "status": "success",
            "result": "SYNTHETIC DIRECT RESEARCH FINDINGS",
            "error": None,
        })
        checked = tools["check_async_task"].func(
            task_id, SimpleNamespace(tool_call_id="synthetic-async-check"))
    return {"launch": launch, "checked": checked}


def _expected_async_task_id() -> str:
    key = "synthetic-parent-work:synthetic-async-start"
    return "sub-" + hashlib.sha256(key.encode()).hexdigest()[:24]


def _expected_async_launch() -> str:
    task_id = _expected_async_task_id()
    return (f"Started subagent. task_id: {task_id}. In the user reply, call it a "
            "subagent or task, never background or async. Report this full ID and "
            "return now; the result will trigger a follow-up.")


def _source_manifest(trace: CensusTrace) -> list[dict[str, Any]]:
    import assist.agent as assist_agent_mod
    import assist.middleware.context_rider_middleware as rider_mod
    import assist.middleware.prompt_composition as composition_mod
    import assist.middleware.memory_middleware as assist_memory_mod
    import assist.middleware.skills_middleware as assist_skills_mod
    import deepagents.graph as graph_mod
    import deepagents.middleware.async_subagents as async_subagents_mod
    import deepagents.middleware.filesystem as filesystem_mod
    import deepagents.middleware.memory as memory_mod
    import deepagents.middleware.skills as skills_mod
    import deepagents.middleware.subagents as subagents_mod
    import deepagents.middleware.summarization as summarization_mod
    import langchain.agents.middleware.todo as todo_mod

    modules = {
        "python:deepagents.graph": graph_mod,
        "python:deepagents.middleware.async_subagents": async_subagents_mod,
        "python:deepagents.middleware.filesystem": filesystem_mod,
        "python:deepagents.middleware.memory": memory_mod,
        "python:deepagents.middleware.skills": skills_mod,
        "python:deepagents.middleware.subagents": subagents_mod,
        "python:deepagents.middleware.summarization": summarization_mod,
        "python:langchain.agents.middleware.todo": todo_mod,
        "python:assist.middleware.context_rider_middleware": rider_mod,
        "python:assist.middleware.prompt_composition": composition_mod,
        "python:assist.middleware.memory_middleware": assist_memory_mod,
        "python:assist.middleware.skills_middleware": assist_skills_mod,
    }
    referenced = set()
    for call in trace.calls:
        provenance = call.get("provenance", {})
        for span in [*provenance.get("initial_spans", []),
                     *provenance.get("final_spans", [])]:
            referenced.update(
                [span["source_id"]] if "source_id" in span
                else span.get("source_ids", []))
        for transition in provenance.get("transitions", []):
            referenced.update(transition["source_ids"])
    entries = [source for source_id, source in trace.sources.items()
               if source_id in referenced]
    for source_id, module in modules.items():
        if source_id not in referenced:
            continue
        path = inspect.getsourcefile(module)
        if path is None:
            raise AssertionError(f"cannot locate prompt source {source_id}")
        entries.append({
            "id": source_id,
            "kind": "python-module",
            "locator": module.__name__,
            "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        })
    if "python:deepagents.graph.BASE_AGENT_PROMPT" in referenced:
        entries.append({
            "id": "python:deepagents.graph.BASE_AGENT_PROMPT",
            "kind": "python-symbol",
            "locator": "deepagents.graph.BASE_AGENT_PROMPT",
            "source_sha256": _sha(graph_mod.BASE_AGENT_PROMPT),
        })
    for source_id, locator, value in (
            ("python:assist.middleware.skills_middleware.SMALL_MODEL_SKILLS_PROMPT",
             "assist.middleware.skills_middleware.SMALL_MODEL_SKILLS_PROMPT",
             assist_skills_mod.SMALL_MODEL_SKILLS_PROMPT),
            ("python:assist.middleware.memory_middleware.SMALL_MODEL_MEMORY_PROMPT",
             "assist.middleware.memory_middleware.SMALL_MODEL_MEMORY_PROMPT",
             assist_memory_mod.SMALL_MODEL_MEMORY_PROMPT),
            ("python:assist.middleware.memory_middleware.THREAD_MEMORY_PROMPT",
             "assist.middleware.memory_middleware.THREAD_MEMORY_PROMPT",
             assist_memory_mod.THREAD_MEMORY_PROMPT),
            ):
        if source_id in referenced:
            entries.append({
                "id": source_id,
                "kind": "python-symbol",
                "locator": locator,
                "source_sha256": _sha(value),
            })
    for source_id, locator, value in (
            ("python:assist.agent.create_agent",
             "assist.agent.create_agent",
             inspect.getsource(assist_agent_mod.create_agent)),
            ("python:assist.middleware.skills_middleware."
             "SmallModelSkillsMiddleware._format_skills_list",
             "assist.middleware.skills_middleware."
             "SmallModelSkillsMiddleware._format_skills_list",
             inspect.getsource(
                 assist_skills_mod.SmallModelSkillsMiddleware._format_skills_list)),
            ("python:assist.middleware.memory_middleware."
             "SmallModelMemoryMiddleware._format_agent_memory",
             "assist.middleware.memory_middleware."
             "SmallModelMemoryMiddleware._format_agent_memory",
             inspect.getsource(
                 assist_memory_mod.SmallModelMemoryMiddleware._format_agent_memory))):
        if source_id in referenced:
            entries.append({
                "id": source_id,
                "kind": "python-symbol",
                "locator": locator,
                "source_sha256": _sha(value),
            })
    return sorted(entries, key=lambda entry: entry["id"])


_TOOL_ORIGINS = {
    "cancel_async_task": "assist.async_subagents",
    "check_async_task": "assist.async_subagents",
    "create_schedule": "assist.schedule.tools",
    "create_subscription": "assist.events.tools",
    "delete_schedule": "assist.schedule.tools",
    "delete_subscription": "assist.events.tools",
    "directions": "assist.tools",
    "edit_file": "deepagents.middleware.filesystem",
    "execute": "deepagents.middleware.filesystem",
    "find_regions": "assist.geo.tools",
    "glob": "deepagents.middleware.filesystem",
    "grep": "deepagents.middleware.filesystem",
    "list_allowed_hosts": "assist.egress.tools",
    "list_async_tasks": "assist.async_subagents",
    "list_regions": "assist.geo.tools",
    "list_schedules": "assist.schedule.tools",
    "list_subscriptions": "assist.events.tools",
    "list_threads": "assist.receptionist",
    "load_skill": "assist.middleware.skills_middleware",
    "ls": "deepagents.middleware.filesystem",
    "map_data": "assist.tools",
    "modify_schedule": "assist.schedule.tools",
    "modify_subscription": "assist.events.tools",
    "new_thread": "assist.receptionist",
    "notify": "assist.events.notify",
    "open_thread": "assist.receptionist",
    "pause_schedule": "assist.schedule.tools",
    "propose_region_download": "assist.geo.tools",
    "read_file": "deepagents.middleware.filesystem",
    "read_url": "assist.tools",
    "remove_allowed_host": "assist.egress.tools",
    "request_egress": "assist.egress.tools",
    "resume_schedule": "assist.schedule.tools",
    "search_internet": "assist.tools",
    "send_email": "assist.events.email",
    "start_async_task": "assist.async_subagents",
    "task": "deepagents.middleware.subagents",
    "travel": "assist.tools",
    "update_async_task": "assist.async_subagents",
    "write_file": "deepagents.middleware.filesystem",
    "write_todos": "langchain.agents.middleware.todo",
}

_DECLARED_TEMPLATE_RENDER_HASHES = {
    "assist/templates/deepagents/assist_core.md.j2": {
        "d0bf608e2e7b8c0b5bad52e2dd87fe17b91f357d06bb449742406f08230eb32d",
        "cbc483e291b07adf34e6be71a409f486bee4de7a6f520f0c15994ede69dbf30c",
        "41d234fb460cd2a09f3ff18bef2b243b99f8f0faa5d444763f30c7ee314b0cfe",
        "c4ac60cad1b9ed89cd617682eefa1b300b29166725d873bb8e41af7272bda233",
        "9d89bf9b6d89b519fd4657742bae103836ac8eb2133ab11ffe1aa50ebaaea23b",
        "a2a6a821f96a1e9f3b78823fcd3d81ccb4fdd0437e8ded435984e2e3f28636cf",
        "37d8b35868296a9623a05a8f0d6c21dcf56b9e67bd061a7b94959a3c00207bf2",
        "18641e7a04979397d92c8fd26ce39abb1b03eb087c009d052553d633b7e311c3",
        "4ea439c3d0ff152c0c749a2d89ad733c4946f1f6a5ebb70f13a6bc219968d875",
    },
    "assist/templates/deepagents/context_agent.md.j2": {
        "29cca4088f56e56eee0a685e794de8a6ff0c63526231839f2f24e246adb9ec70",
        "e79b9a545fec4ed27d3b2917b674f6999a1869eb4bbea713416304f0c22bac74",
    },
    "assist/templates/deepagents/describe_system.md.j2": {
        "07d7647a62ac9a4f482a82d8a2e8d2e2bb2f8a41fb8f4c377e6944d63678bff7",
    },
    "assist/templates/deepagents/dev_critique.md.j2": {
        "a8b42659124c2f4e7531349dab272e56b01a481825d5d2b87db374d1a9a504a9",
        "b154c65f2d6659c9b2db5865cc49515c4488551c3bf98b7b2847b595cade6dfc",
    },
    "assist/templates/deepagents/fact_checker.md.j2": {
        "c8853dc566ad691ea9406fc2c42dcb01298ba24193fd0f81a3b8677205d0afef",
    },
    "assist/templates/deepagents/general_instructions.md.j2": {
        "102082e634233200a9b0fc4c0240ac93b6afc7e922fba97ffec254ea7f82eafd",
        "1b8e35f63bf4be8dd8676e477a084171d0ee50e01e812656c4ab594024ba3629",
        "1e9fb38a54b05e59da69f1199c231a8dacba1c6500a609c4603196cc9a51b5e7",
        "3c8878748692f95d9490f931e3c24ec2b0bb85134abb9d5067602df829966b5a",
        "24eb808031adfa5ec188311e03035d0ea8824915dbb4d3c8b75b6b151347ebf5",
        "5288bb0e5d96033aceeedab00b227264a409f2304531050d4a4b0d88ee468055",
        "7f0ccd414ba75057f468fca97520024c8c991b64adb0ac53039886dc7e76c2e3",
        "85099417c63007d135a6b72eb3a62a162248bdf522be8597e1f0a9ee14163ddd",
    },
    "assist/templates/deepagents/research_instructions.txt.j2": {
        "379e55c61addf130f00a29a6221f16de89964e23b5c1e98cf9746f38211fb071",
        "85dec6ea05cfaa2aed2cd943ff5144107a729b71be04042a78c2ed7458c0e1fe",
    },
    "assist/templates/deepagents/sub_critique.txt.j2": {
        "b3d00837be3dbfac3158e9b47d0fa64612b8255248eea5fbb19a095075d5b801",
    },
    "assist/templates/deepagents/sub_research.txt.j2": {
        "ea51549d2bae79ee031b6b5a8fdc0872708a6f7536303f5f1ba68512c0db3415",
    },
    "assist/templates/receptionist_system.md.j2": {
        "f2bfbc35fb540f64c385b74c7bf590517b80bc5ee9abd660eb3ad736bb3adb8e",
    },
    "edd/templates/capture_agent.md.j2": {
        "2cbb0976855311a174129a93d87801f4abe5404404422f46a14b2cb786f4ec7a",
    },
}

_REFERENCED_TEMPLATE_LOCATORS = {
    "assist/templates/deepagents/assist_core.md.j2",
    "assist/templates/deepagents/context_agent.md.j2",
    "assist/templates/deepagents/describe_system.md.j2",
    "assist/templates/deepagents/fact_checker.md.j2",
    "assist/templates/deepagents/general_instructions.md.j2",
    "assist/templates/deepagents/research_instructions.txt.j2",
    "assist/templates/deepagents/sub_critique.txt.j2",
    "assist/templates/deepagents/sub_research.txt.j2",
    "assist/templates/receptionist_system.md.j2",
    "edd/templates/capture_agent.md.j2",
}

_CLAIM_SPECS = (
    (None, "positive", "load_skill", "FIRST tool call MUST be `load_skill"),
    ("assist.agent template", "positive", "task",
     "The `task` tool is only for your synchronous"),
    ("assist.agent template", "negative", "task",
     "Delegation is unavailable in this restricted turn"),
    (None, "conditional", "notify", "If `notify` is available"),
    (None, "positive", "start_async_task", "Every subagent call uses `start_async_task`"),
    (None, "positive", "check_async_task", "call `check_async_task` for that task"),
    (None, "positive", "list_async_tasks", "call `list_async_tasks` or `check_async_task`"),
    (None, "positive", "update_async_task", "steer it with `update_async_task`"),
    (None, "positive", "cancel_async_task", "stop it with `cancel_async_task`"),
    (None, "positive", "task", "call BOTH in parallel via the `task` tool"),
    (None, "positive", "edit_file", "ADD new content using `edit_file`"),
    (None, "positive", "write_todos", "You have access to the `write_todos` tool"),
    (None, "positive", "ls", "## Filesystem Tools `ls`, `read_file`, `write_file`"),
    (None, "positive", "read_file", "## Filesystem Tools `ls`, `read_file`, `write_file`"),
    (None, "positive", "write_file", "## Filesystem Tools `ls`, `read_file`, `write_file`"),
    (None, "positive", "edit_file", "## Filesystem Tools `ls`, `read_file`, `write_file`"),
    (None, "positive", "glob", "## Filesystem Tools `ls`, `read_file`, `write_file`"),
    (None, "positive", "grep", "## Filesystem Tools `ls`, `read_file`, `write_file`"),
    (None, "positive", "execute", "You have access to an `execute` tool"),
    (None, "positive", "execute", "shell-execution tool (e.g. `execute`)"),
    (None, "conditional", "start_async_task",
     "If `start_async_task` is available"),
    (None, "conditional", "task", "Otherwise use `task`"),
    (None, "positive", "load_skill", "Call `load_skill(name="),
    ("deepagents.SkillsMiddleware", "ordered", "task",
     "Do not run `ls`, `read_file`,\n   `task`"),
    ("deepagents.MemoryMiddleware", "ordered", "task",
     "Before any work tool (`task`,"),
    (None, "positive", "write_file", "empty — use `write_file`"),
    (None, "positive", "edit_file", "use `edit_file` to append"),
    ("assist.agent template", "negative", "write_file", "You are strictly read-only"),
    ("assist.agent template", "negative", "edit_file", "You are strictly read-only"),
    (None, "positive", "ls", "You may `ls`, `glob`, `grep`, and read files"),
    (None, "positive", "glob", "You may `ls`, `glob`, `grep`, and read files"),
    (None, "positive", "grep", "You may `ls`, `glob`, `grep`, and read files"),
    (None, "positive", "read_file", "You may `ls`, `glob`, `grep`, and read files"),
    (None, "positive", "search_internet", "`search_internet` returns a list"),
    (None, "positive", "read_url", "pass that to `read_url`"),
    (None, "positive", "read_url", "Use the `read_url` tool to fetch"),
    (None, "positive", "read_file", "do you `read_file` that exact filename"),
    (None, "positive", "write_todos", "list them with `write_todos`"),
    (None, "positive", "task", "3 `task` calls"),
    ("deepagents.SubAgentMiddleware", "positive", "task",
     "You have access to a `task` tool"),
    ("assist.receptionist template", "positive", "list_threads",
     "- list_threads —"),
    ("assist.receptionist template", "positive", "open_thread",
     "- open_thread —"),
    ("assist.receptionist template", "positive", "new_thread",
     "- new_thread —"),
    ("edd.capture template", "positive", "read_file", "- `read_file(path)`"),
    ("edd.capture template", "positive", "write_file", "- `write_file(path, content)`"),
    ("edd.capture template", "positive", "ls", "- `ls(path)`"),
    ("edd.capture template", "positive", "glob", "- `glob(pattern)`"),
    ("edd.capture template", "argument-relative", "read_file",
     "All paths are **relative to your working directory**"),
    ("edd.capture template", "argument-relative", "write_file",
     "All paths are **relative to your working directory**"),
    ("edd.capture template", "argument-relative", "ls",
     "All paths are **relative to your working directory**"),
)


def _instruction_blocks(call: dict[str, Any]) -> list[tuple[str, str]]:
    provenance = call["provenance"]
    initial_text = provenance["initial_text"]
    blocks = [
        (span["owner"], initial_text[span["start"]:span["end"]])
        for span in provenance["initial_spans"]
    ]
    blocks.extend((transition["owner"], transition["exact_change"])
                  for transition in provenance["transitions"]
                  if transition["operation"] != "replace")
    tool_names: dict[str, str] = {}
    for message in call["provider_payload"]["messages"]:
        for tool_call in message.get("tool_calls", []):
            tool_names[tool_call["id"]] = tool_call["function"]["name"]
        if message.get("role") == "tool" \
                and tool_names.get(message.get("tool_call_id")) == "load_skill":
            content = message.get("content")
            if not isinstance(content, str):
                raise AssertionError("loaded skill result is not text")
            match = re.search(r"^---\nname:\s*([^\n]+)", content)
            name = match.group(1).strip() if match else "unknown"
            blocks.append((f"loaded skill:{name}", content))
    return blocks


def _capability_claims(call: dict[str, Any]) -> list[dict[str, str]]:
    claims = []
    seen = set()
    for owner, text in _instruction_blocks(call):
        for expected_owner, polarity, tool, marker in _CLAIM_SPECS:
            key = (owner, polarity, tool, marker)
            if (expected_owner is None or expected_owner == owner) \
                    and marker in text and key not in seen:
                claims.append({
                    "owner": owner,
                    "polarity": polarity,
                    "tool": tool,
                    "marker": marker,
                })
                seen.add(key)
    return claims

_DECLARED_TOOL_CALLS = {
    "synthetic-call-grounding-load": (
        "load_skill", {"name": "grounding"}),
    "synthetic-call-safe-exec": (
        "execute", {"command": "printf synthetic-ok"}),
    "synthetic-call-git-push": (
        "execute", {"command": "git push origin main"}),
    "synthetic-call-hitl": (
        "send_email", {
            "to": "synthetic-recipient@example.invalid",
            "subject": "Synthetic approval probe",
            "body": "Synthetic body.",
        }),
    "synthetic-call-hitl-skill-load": (
        "load_skill", {"name": "send-email"}),
    "synthetic-call-read-only": (
        "write_file", {
            "file_path": "/synthetic-forbidden.txt",
            "content": "SYNTHETIC_FORBIDDEN_WRITE",
        }),
    "synthetic-call-read-only-edit": (
        "edit_file", {
            "file_path": "/synthetic-forbidden.txt",
            "old_string": "SYNTHETIC_OLD",
            "new_string": "SYNTHETIC_NEW",
        }),
    "synthetic-call-provenance": (
        "read_url", {"url": "https://unprovenanced.example.invalid/synthetic"}),
    "synthetic-call-skill-precedence-built-in-load": (
        "load_skill", {"name": "dev"}),
    "synthetic-call-skill-precedence-embedder-load": (
        "load_skill", {"name": "dev"}),
    "synthetic-call-nested-research-worker": (
        "task", {"description": "SYNTHETIC CHILD BRIEF",
                 "subagent_type": "research-agent"}),
    "synthetic-call-nested-fact-check": (
        "task", {"description": "SYNTHETIC CHILD BRIEF",
                 "subagent_type": "fact-check-agent"}),
    "synthetic-call-nested-report-critique": (
        "task", {"description": "SYNTHETIC CHILD BRIEF",
                 "subagent_type": "critique-agent"}),
}

_DECLARED_TOOL_SCHEMA_HASHES = {
    "cancel_async_task": {"72f0b2b9a7b781b4e3e60facf9899f59faa94743f32f2b43d1fa16e0bc1af01b"},
    "check_async_task": {"3b54d1abd04592123802b504c50e47f855320db630605c75ac0e50dad03ea445"},
    "create_schedule": {"17ae4733ba78d32789761332c0b782600873027412beecd2a7d2d2b548392f34"},
    "create_subscription": {"72476dc729012cbe0f9d65af14dc1541a7fb2a9e88e5f29730f9b68969fc7974"},
    "delete_schedule": {"ac6a06a26f9df248183a12c5aab9ecaafe637804ac61c2f4d9edca588d15359f"},
    "delete_subscription": {"e185f8c258a141810057c9ed5fd00765db905264eb79747aba47bc2e9dcaaff3"},
    "directions": {"63a3aaf8cf56c193bfb59e5de457215a46ed48f1844a614b16ef230c1f7cecef"},
    "edit_file": {"a5ee5273f41f2b2e8f5db44efdf2c1b3bb533f4f4891e2becaa216803382177d"},
    "execute": {"056e1ff2b9fb1851f0c7c40a2844932002d957dc53522bb4813fba9a02689dbb"},
    "find_regions": {"0b83025d7331f37308c54fe8001c669b1467ee6209846fc40495ea06ac8227fe"},
    "glob": {"cfdb0c748d1c08e1dbad1424a7a0ebb67b2a0b0b9006cb69cbf9fdc7e5272679"},
    "grep": {"387cc9b610ca8e760fab5ba9e2fdb5b598db6ff10cbee60eb31f4f5027f7534b"},
    "list_allowed_hosts": {"75f484c4031eb56e305320f9d599e1b81f7cf4402b55cf971d296bc3575dfd36"},
    "list_async_tasks": {"4e23d3e6bc9be65808d2d08749b75e36204dd46d3c4853c71e8f9dc3e071de6a"},
    "list_regions": {"10fb83255f542acbe460605fd06fa01fcb6061e138df0c5194c38177e1a6a079"},
    "list_schedules": {"e0cd57bcb7cd8a6c27e84a665534daf714ef7867e8c95795252c3981caf7217e"},
    "list_subscriptions": {"dcb6ac3c9c61a0dab7c30d0ea6311d20fe5e3502cbc748145ba8032ee1cd931f"},
    "list_threads": {"6ac6da21f65a92eea33c6f825b98322478f3640c944969bf1a47024ed6d7c78b"},
    "load_skill": {
        "9870c9427a98117cbe8f4947a569f1b490250611ca3b0b305929ddf1365a629b",
        "abc2bc372ed164d625f4f03f3b24602af5e79b99fa43e1c7c60bc5ef2c8233cb",
    },
    "ls": {"295a42f624bb3d6fd5e8f13e3d3172bf5856df6bc32d5c700c2b4d6b559850de"},
    "map_data": {"0285e871e6f8ebe3d7bdd34068af9fd7dce5841918c76d2866d64e2620f5851f"},
    "modify_schedule": {"2b82e293c48bbdfcaaafe797533693b7d15653b7f3dd2e996c6c06f27a0c16ca"},
    "modify_subscription": {"eebbeda47d86c17fda5ae6d6332ab74c5663be9068fe470189e9f45ef0adf769"},
    "new_thread": {"110cfb0e827f07c9d800a314d4bc119aa3f107422983a7b2bca9bea2d01f29cf"},
    "notify": {"3c2aef11b380cda5d273b7303e544960478f2834d1757b8d7ee1dd87189015aa"},
    "open_thread": {"d9d2c013e8759960c87b9114d1cbac06e9f606b18cc956d7b9b25acfd49c9722"},
    "pause_schedule": {"f4650ba6d1e6b58948b8a09821a1cf1dbfcfb8820a3ae5e63b64dc5841eb2265"},
    "propose_region_download": {"638cb587cde24979bcd6ee36a7169d8b61bec09afd2ab7adf2afecb3f1695d58"},
    "read_file": {"2e139ac315b65f5b6614591642b26defae959ab6878fff990f26f0b997feb5af"},
    "read_url": {"a4e5eae871d9956ee0227f48f8e86eb1c96b1740cb73a7e1a1e57a8cfae18716"},
    "remove_allowed_host": {"4bc5319afb9de4381fdd059f5bb35034caccacee5eb52b7443a6af2c27ed0498"},
    "request_egress": {"84fdeccfd7466d726247ecc6c482c1bded92bf33eb54a7ce137cd42c233c35b4"},
    "resume_schedule": {"884f79fb96c7e374a2e204ed9f00203243607a71a1440a954936a0516a612183"},
    "search_internet": {"a240ffc08228c91577dde743290e775c51b5aa1c3b059cd147f490fa999def13"},
    "send_email": {"13a0df1decb69fe0f4839784c892bd4d5b68b341beef6c5985e799f8d86ab9a7"},
    "start_async_task": {"eddaad4cd7c4daac2dcb056eda682895987de8d4c2ba869fc00f2a2aa8157793"},
    "travel": {"00329efd850f0564a51fc91cda1cef386988a8538eb4dbf9c58a3103f99f9384"},
    "update_async_task": {"f01d4f360963e9d7c9f00b36e0518dc09e6a26a4cab50f304e893c613655a5b4"},
    "write_file": {"9bfc37771fdc668c20dd0a508eef41a79a36743e3a368efb4d48024e6fcd9a38"},
    "write_todos": {"48687d0ff93c2b303781aaed33f8230ba2eeaf3d158ad19b27c968aad39c1601"},
}

_TASK_SCHEMA_HASH_BY_PATH = {
    "delegate": "6946a1514ff0b9a60887aa17ed4d13ce8317482e860f1f691b7b3282613499e3",
    "legacy-main": "59fd529189b7eaef0e07f1c4b709896d5304dfde283d3121f405ebbbff3b347d",
    "research-lead": "a7df416ab3090318aa4619d804e3b04593ea612756de3f01e500605cfb5fe9cc",
}


def _expected_skill_file(tool_call_id: str) -> str:
    repo = Path(__file__).resolve().parents[1]
    files = {
        "synthetic-call-grounding-load":
            (repo / "assist/main_guidance_skills/grounding/SKILL.md").read_text(
                encoding="utf-8"),
        "synthetic-call-hitl-skill-load":
            (repo / "assist/web_skills/send-email/SKILL.md").read_text(
                encoding="utf-8"),
        "synthetic-call-skill-precedence-built-in-load":
            (repo / "assist/skills/dev/SKILL.md").read_text(encoding="utf-8"),
        "synthetic-call-skill-precedence-embedder-load":
            "---\nname: dev\ndescription: SYNTHETIC EMBEDDER DEV DESCRIPTION\n"
            "---\n\n\x1b[31mSYNTHETIC_EMBEDDER_DEV_BODY\x1b[0m\n",
    }
    return files[tool_call_id]


def _expected_tool_results() -> dict[str, str]:
    return {
        "synthetic-call-grounding-load":
            _expected_skill_file("synthetic-call-grounding-load")
            + "\n\nNo additional tools became available.",
        "synthetic-call-safe-exec":
            "SYNTHETIC_EXECUTE_OK\n[Command succeeded with exit code 0]",
        "synthetic-call-git-push":
            "Error: direct git push is not allowed.  The user controls pushes from "
            "the web UI.  To publish your work to origin, ask the user to click "
            "'Push to origin' in their browser.",
        "synthetic-call-hitl-skill-load":
            _expected_skill_file("synthetic-call-hitl-skill-load")
            + "\n\nNewly available tools: send_email.",
        "synthetic-call-skill-precedence-built-in-load":
            _expected_skill_file("synthetic-call-skill-precedence-built-in-load")
            + "\n\nNo additional tools became available.",
        "synthetic-call-skill-precedence-embedder-load":
            "---\nname: dev\ndescription: SYNTHETIC EMBEDDER DEV DESCRIPTION\n"
            "---\n\nSYNTHETIC_EMBEDDER_DEV_BODY\n"
            + "\n\nNo additional tools became available.",
        "synthetic-call-read-only":
            "Error: this agent is read-only and cannot call 'write_file'. If the "
            "user asked you to write or modify something, do not attempt the write "
            "— instead surface the relevant file path, current contents, and any "
            "format conventions the caller will need, and explicitly note that you "
            "did not perform the write.",
        "synthetic-call-read-only-edit":
            "Error: this agent is read-only and cannot call 'edit_file'. If the "
            "user asked you to write or modify something, do not attempt the write "
            "— instead surface the relevant file path, current contents, and any "
            "format conventions the caller will need, and explicitly note that you "
            "did not perform the write.",
        "synthetic-call-provenance":
            "That URL was a guess: it appears in no search result, the question, or "
            "any trusted source in your context, so it would be a dead link. You have "
            "no sourced URLs to read. Do NOT type URLs from memory or keep trying "
            "different guesses. If you can run search_internet, do that first to find "
            "real URLs; if search has already come back empty or unavailable, report "
            "that you could not find reliable sources for this and stop.",
        "synthetic-call-nested-research-worker":
            "SYNTHETIC TERMINAL nested-research-worker 1",
        "synthetic-call-nested-fact-check":
            "SYNTHETIC TERMINAL nested-fact-check 1",
        "synthetic-call-nested-report-critique":
            "SYNTHETIC TERMINAL nested-report-critique 1",
    }


def _expected_load_artifact(tool_call_id: str) -> dict[str, str]:
    from deepagents.middleware.skills import _parse_skill_metadata

    requested_name = _DECLARED_TOOL_CALLS[tool_call_id][1]["name"]
    skill_file = _expected_skill_file(tool_call_id)
    content = _expected_tool_results()[tool_call_id]
    metadata = _parse_skill_metadata(
        skill_file, f"/{requested_name}/SKILL.md", requested_name)
    if metadata is None:
        raise AssertionError("declared load result metadata did not parse")
    fingerprint_payload = {
        "allowed_tools": list(metadata["allowed_tools"]),
        "skill_file_sha256": hashlib.sha256(
            skill_file.encode("utf-8")).hexdigest(),
        "description": metadata["description"],
        "name": metadata["name"],
    }
    return {
        "schema": "assist.skill-load.v1",
        "requested_name": requested_name,
        "winner_fingerprint": hashlib.sha256(json.dumps(
            fingerprint_payload, ensure_ascii=False, separators=(",", ":"),
            sort_keys=True).encode("utf-8")).hexdigest(),
        "result_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


_TOOL_RESULT_METADATA = {
    "synthetic-call-grounding-load": ("load_skill", "success"),
    "synthetic-call-safe-exec": ("execute", "success"),
    "synthetic-call-git-push": ("execute", "error"),
    "synthetic-call-hitl-skill-load": ("load_skill", "success"),
    "synthetic-call-skill-precedence-built-in-load": ("load_skill", "success"),
    "synthetic-call-skill-precedence-embedder-load": ("load_skill", "success"),
    "synthetic-call-read-only": ("write_file", "error"),
    "synthetic-call-read-only-edit": ("edit_file", "error"),
    "synthetic-call-provenance": ("read_url", "error"),
    "synthetic-call-nested-research-worker": ("task", "success"),
    "synthetic-call-nested-fact-check": ("task", "success"),
    "synthetic-call-nested-report-critique": ("task", "success"),
}

_DECLARED_CAPTURE_TASK_SHA256 = \
    "dc6155e1863f23b35006efd98f0a5e2ea0c1b392650567596cd7c86a6c6af402"
_DECLARED_TOOL_NODE_HISTORY_SHA256 = \
    "47543010e202c1c99aa62e953eafad99b81d55a345e75100ef58556f1f61ad04"
_DECLARED_PROMPT_BLOCK_CHAIN_SHA256 = \
    "0bfba686ddb0f3b91d7931b6468aeaa00bfe32e9ee01bcd77636182ac696edd9"


def _provider_tool_pair(tool_call_id: str) -> list[dict[str, Any]]:
    name, arguments = _DECLARED_TOOL_CALLS[tool_call_id]
    return [
        {
            "content": None,
            "role": "assistant",
            "tool_calls": [{
                "type": "function",
                "id": tool_call_id,
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }],
        },
        {
            "content": _expected_tool_results()[tool_call_id],
            "role": "tool",
            "tool_call_id": tool_call_id,
        },
    ]


def _expected_message_history(scenario: str, index: int) -> list[dict[str, Any]]:
    if scenario == "thread-description":
        return [
            {"content": "SYNTHETIC USER description", "role": "user"},
            {"content": "SYNTHETIC ASSISTANT description", "role": "assistant"},
            {"content": "Describe the conversation up until now", "role": "user"},
        ]
    if scenario.startswith("nested-") and index == 1:
        return [{"content": "SYNTHETIC CHILD BRIEF", "role": "user"}]
    user_suffix = {
        "legacy-main": "legacy",
        "context-read-only": "context",
        "thread-description": "description",
    }.get(scenario, scenario)
    messages = [{"content": f"SYNTHETIC USER {user_suffix}", "role": "user"}]
    ids: list[str] = []
    if scenario == "web-main-core":
        ids = ["synthetic-call-grounding-load", "synthetic-call-safe-exec",
               "synthetic-call-git-push"][:index]
    elif scenario == "web-main-full" and index == 1:
        ids = ["synthetic-call-hitl-skill-load"]
    elif scenario.startswith("skill-precedence-") and index == 1:
        ids = [f"synthetic-call-{scenario}-load"]
    elif scenario == "context-read-only":
        ids = ["synthetic-call-read-only",
               "synthetic-call-read-only-edit"][:index]
    elif scenario == "research-leaf-provenance" and index == 1:
        ids = ["synthetic-call-provenance"]
    elif scenario.startswith("nested-") and index == 2:
        ids = [f"synthetic-call-{scenario}"]
    for tool_call_id in ids:
        messages.extend(_provider_tool_pair(tool_call_id))
    return messages


def _assert_tool_result(message: dict[str, Any]) -> None:
    tool_call_id = message.get("tool_call_id", "")
    if not tool_call_id.startswith("synthetic-call-"):
        raise AssertionError("tool result ID is not a declared fixture")
    expected = _expected_tool_results().get(tool_call_id)
    if expected is None or message.get("content") != expected:
        raise AssertionError(f"undeclared tool result for {tool_call_id}")
    if message.get("role") == "tool":
        if set(message) != {"role", "content", "tool_call_id"}:
            raise AssertionError(f"undeclared provider tool-result shape: {tool_call_id}")
    else:
        name, status = _TOOL_RESULT_METADATA[tool_call_id]
        expected_message = {
            "type": "tool",
            "content": expected,
            "name": name,
            "tool_call_id": tool_call_id,
            "status": status,
        }
        if name == "load_skill":
            expected_message["artifact"] = _expected_load_artifact(tool_call_id)
        if message != expected_message:
            raise AssertionError(f"undeclared observed tool-result shape: {tool_call_id}")


def _assert_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise AssertionError(f"{label} has undeclared fields")


_JSON_SCHEMA_KEYSETS = {
    frozenset({"type"}),
    frozenset({"description", "type"}),
    frozenset({"enum", "type"}),
    frozenset({"default", "type"}),
    frozenset({"anyOf", "default"}),
    frozenset({"anyOf", "default", "description"}),
    frozenset({"default", "description", "type"}),
    frozenset({"default", "description", "enum", "type"}),
    frozenset({"description", "maxLength", "minLength", "type"}),
    frozenset({"items", "type"}),
    frozenset({"description", "properties", "required", "type"}),
    frozenset({"properties", "type"}),
    frozenset({"properties", "required", "type"}),
}


def _assert_json_schema(schema: dict[str, Any]) -> None:
    if frozenset(schema) not in _JSON_SCHEMA_KEYSETS:
        raise AssertionError("provider tool schema has undeclared fields")
    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, dict):
            raise AssertionError("provider tool properties are not an object")
        for field_schema in properties.values():
            if not isinstance(field_schema, dict):
                raise AssertionError("provider tool field schema is not an object")
            _assert_json_schema(field_schema)
    items = schema.get("items")
    if items is not None:
        if not isinstance(items, dict):
            raise AssertionError("provider tool items schema is not an object")
        _assert_json_schema(items)
    alternatives = schema.get("anyOf")
    if alternatives is not None:
        if not isinstance(alternatives, list):
            raise AssertionError("provider tool alternatives are not a list")
        for alternative in alternatives:
            if not isinstance(alternative, dict):
                raise AssertionError("provider tool alternative is not an object")
            _assert_json_schema(alternative)


def _assert_closed_artifact_shapes(artifact: dict[str, Any]) -> None:
    root_keys = {"schema_version", "fixed_clock", "source_manifest", "calls",
                 "tool_nodes", "capabilities", "observations", "findings"}
    if "artifact_sha256" in artifact:
        root_keys.add("artifact_sha256")
    _assert_keys(artifact, root_keys, "census artifact")

    source_keys = {
        "package": {"id", "kind", "version"},
        "python-module": {"id", "kind", "locator", "source_sha256"},
        "python-symbol": {"id", "kind", "locator", "source_sha256"},
        "template": {"id", "kind", "locator", "source_sha256",
                     "rendered_sha256"},
        "skill": {"id", "kind", "scenario", "name", "description",
                  "description_sha256", "source", "path", "allowed_tools",
                  "content_sha256"},
        "synthetic-fixture": {"id", "kind", "scenario", "name", "value",
                              "value_sha256"},
    }
    for source in artifact["source_manifest"]:
        expected = source_keys.get(source.get("kind"))
        if expected is None:
            raise AssertionError("source manifest has an undeclared kind")
        _assert_keys(source, expected, "source manifest entry")

    call_keys = {"scenario", "call_index", "path", "provider_payload",
                 "prompt_events", "visible_tools",
                 "matching_tool_nodes", "provenance"}
    prompt_event_keys = {"owner", "before_sha256", "after_sha256",
                         "before_text_sha256", "after_text_sha256",
                         "characters_before", "characters_after",
                         "stringified_prior_blocks"}
    provenance_keys = {"initial_sha256", "initial_text_sha256",
                       "initial_characters", "initial_text", "initial_spans",
                       "initial_block_layout", "transitions", "final_spans",
                       "final_block_layout", "final_sha256", "final_text_sha256"}
    transition_keys = {"owner", "operation", "before_sha256", "after_sha256",
                       "before_text_sha256", "after_text_sha256",
                       "characters_before", "characters_after", "exact_change",
                       "source_ids", "content_segments"}
    segment_keysets = {
        frozenset({"start", "end", "source_ids"}),
        frozenset({"start", "end", "source_ids", "owner"}),
    }
    span_keysets = {
        frozenset({"start", "end", "owner", "source_id"}),
        frozenset({"start", "end", "owner", "source_id", "rendered_sha256"}),
        frozenset({"start", "end", "owner", "source_ids"}),
    }
    for call in artifact["calls"]:
        _assert_keys(call, call_keys, "provider call")
        payload = call["provider_payload"]
        payload_keys = {"messages", "model", "stream", "temperature"}
        expected_extra_body = None
        if call["path"] != "capture":
            payload_keys.add("extra_body")
            expected_extra_body = {
                "chat_template_kwargs": {"enable_thinking": False}}
        if call["visible_tools"]:
            payload_keys.add("tools")
        _assert_keys(payload, payload_keys, "provider payload")
        if payload["model"] != "synthetic-qwen-census" \
                or payload["stream"] is not False \
                or payload["temperature"] != 0.1 \
                or payload.get("extra_body") != expected_extra_body:
            raise AssertionError("provider payload has undeclared model settings")
        for message in payload["messages"]:
            role = message.get("role")
            expected = {"role", "content"}
            if role == "assistant" and message.get("tool_calls") is not None:
                expected.add("tool_calls")
            elif role == "tool":
                expected.add("tool_call_id")
            _assert_keys(message, expected, f"provider {role} message")
            content = message.get("content")
            if isinstance(content, list):
                if role != "system" or not content:
                    raise AssertionError(
                        f"provider {role} message has undeclared content blocks")
                for block in content:
                    _assert_keys(block, {"type", "text"},
                                 "provider content block")
                    if block["type"] != "text" \
                            or not isinstance(block["text"], str):
                        raise AssertionError(
                            "provider content block is not exact text")
            elif content is not None and not isinstance(content, str):
                raise AssertionError(
                    f"provider {role} message content is not declared text")
            for tool_call in message.get("tool_calls", []):
                _assert_keys(tool_call, {"type", "id", "function"},
                             "provider tool call")
                if tool_call["type"] != "function":
                    raise AssertionError("provider tool call has an undeclared type")
                _assert_keys(tool_call["function"], {"name", "arguments"},
                             "provider tool-call function")
        for tool_schema in payload.get("tools", []):
            _assert_keys(tool_schema, {"type", "function"},
                         "provider tool schema")
            if tool_schema["type"] != "function":
                raise AssertionError("provider tool schema has an undeclared type")
            function = tool_schema["function"]
            _assert_keys(function, {"name", "description", "parameters"},
                         "provider tool function")
            _assert_json_schema(function["parameters"])
            schema_hash = _sha(tool_schema)
            if function["name"] == "task":
                declared_hashes = {_TASK_SCHEMA_HASH_BY_PATH.get(call["path"])}
            else:
                declared_hashes = _DECLARED_TOOL_SCHEMA_HASHES.get(
                    function["name"], set())
            if schema_hash not in declared_hashes:
                raise AssertionError(
                    f"provider tool schema is not declared: {function['name']}")
        schema_names = [tool_schema["function"]["name"]
                        for tool_schema in payload.get("tools", [])]
        if schema_names != call["visible_tools"] \
                or len(schema_names) != len(set(schema_names)):
            raise AssertionError(
                f"{call['scenario']} provider schema surface drifted")
        for event in call["prompt_events"]:
            _assert_keys(event, prompt_event_keys, "prompt event")
        provenance = call["provenance"]
        _assert_keys(provenance, provenance_keys, "prompt provenance")
        for span in [*provenance["initial_spans"],
                     *provenance["final_spans"]]:
            if frozenset(span) not in span_keysets:
                raise AssertionError("prompt span has undeclared fields")
        for transition in provenance["transitions"]:
            _assert_keys(transition, transition_keys, "prompt transition")
            for segment in transition["content_segments"]:
                if frozenset(segment) not in segment_keysets:
                    raise AssertionError(
                        "prompt content segment has undeclared fields")
        for layout_name in ("initial_block_layout", "final_block_layout"):
            for block in provenance[layout_name]:
                _assert_keys(
                    block,
                    {"index", "type", "start", "end", "text_sha256",
                     "source_ids"},
                    "prompt block layout",
                )

    for node in artifact["tool_nodes"]:
        _assert_keys(node, {"index", "scenario", "candidates", "winners"},
                     "tool node")
        for candidate in node["candidates"]:
            _assert_keys(candidate, {"name", "origin"}, "tool-node candidate")

    call_keys_by_surface = {
        f"{call['scenario']}:{call['call_index']}" for call in artifact["calls"]}
    if set(artifact["capabilities"]) != call_keys_by_surface:
        raise AssertionError("capability surfaces do not match provider calls")
    allowed_action_states = {
        "unexercised",
        "observed-permitted",
        "observed-denied",
        "observed-denied-by-provenance",
        "observed-requires-approval",
        "observed-command-policy",
    }
    for surface in artifact["capabilities"].values():
        _assert_keys(surface, {"path", "registered_tools", "model_visible_tools",
                               "ambiguous_tool_node_matches", "claims",
                               "effective_actions"}, "capability surface")
        for tool in surface["registered_tools"]:
            _assert_keys(tool, {"name", "origin", "classification",
                                "possible_owners"}, "registered tool")
        for claim in surface["claims"]:
            _assert_keys(claim, {"owner", "polarity", "tool", "marker"},
                         "capability claim")
        if set(surface["effective_actions"]) != set(surface["model_visible_tools"]):
            raise AssertionError("effective actions do not match visible tools")
        if not set(surface["effective_actions"].values()) <= allowed_action_states:
            raise AssertionError("effective actions contain an undeclared state")
    for finding in artifact["findings"]:
        _assert_keys(finding, {"kind", "surface", "detail"}, "audit finding")


def _assert_declared_inputs(artifact: dict[str, Any]) -> None:
    _assert_closed_artifact_shapes(artifact)
    if artifact["schema_version"] != SCHEMA_VERSION \
            or artifact["fixed_clock"] != FIXED_NOW:
        raise AssertionError("census artifact has undeclared fixed metadata")
    actual_calls = {
        scenario: sorted(call["call_index"] for call in artifact["calls"]
                         if call["scenario"] == scenario)
        for scenario in EXPECTED_CALL_COUNTS
    }
    expected_calls = {
        scenario: list(range(count))
        for scenario, count in EXPECTED_CALL_COUNTS.items()
    }
    if actual_calls != expected_calls or len(artifact["calls"]) != sum(
            EXPECTED_CALL_COUNTS.values()):
        raise AssertionError("census provider call matrix is incomplete")
    actual_paths = {
        scenario: tuple(
            call["path"] for call in sorted(
                (item for item in artifact["calls"]
                 if item["scenario"] == scenario),
                key=lambda item: item["call_index"],
            )
        )
        for scenario in EXPECTED_PATHS_BY_SCENARIO
    }
    if actual_paths != EXPECTED_PATHS_BY_SCENARIO:
        raise AssertionError("census scenario-to-path matrix drifted")
    for call in artifact["calls"]:
        messages = call["provider_payload"].get("messages", [])
        if call["scenario"] == "capture":
            if len(messages) != 2 or messages[1].get("role") != "user" \
                    or _sha(messages[1].get("content")) != \
                    _DECLARED_CAPTURE_TASK_SHA256:
                raise AssertionError("capture application task drifted")
        elif messages[1:] != _expected_message_history(
                    call["scenario"], call["call_index"]):
            raise AssertionError(
                f"{call['scenario']}:{call['call_index']} provider history drifted")

        prompt_texts = [("system", _system_prompt(call))]
        prompt_texts.extend(
            (transition["owner"], transition["exact_change"])
            for transition in call["provenance"].get("transitions", []))
        for prompt_owner, text in prompt_texts:
            for tag in ("agent_memory", "thread_memory"):
                for match in re.finditer(
                        rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL):
                    memory = match.group(1).strip()
                    expected_memory = (
                        "SYNTHETIC_REPOSITORY_MEMORY"
                        if call["scenario"] == "web-main-full" and tag == "agent_memory"
                        else "SYNTHETIC_THREAD_MEMORY"
                        if call["scenario"] == "web-main-full" and tag == "thread_memory"
                        else "(No thread memory loaded)"
                        if tag == "thread_memory"
                        else "(No memory loaded)")
                    expected_rendering = (
                        rf"\n{expected_memory}\n\n"
                        if prompt_owner == "assist.ContextRiderMiddleware"
                        or (prompt_owner == "system"
                            and call["scenario"] == "web-main-full")
                        else expected_memory)
                    if memory != expected_rendering:
                        raise AssertionError(
                            f"{call['scenario']} {tag} contains undeclared "
                            f"fixture content: {memory!r}")
        if call["scenario"] == "web-main-full":
            from assist.context_rider import ContextRider
            expected_rider = ContextRider(
                sent_at=datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc),
                tz="UTC", place_label="SYNTHETIC PLACE").prose_line()
            if not _system_prompt(call).endswith(f"\n\n{expected_rider}"):
                raise AssertionError("web-main-full context is not the declared fixture")
        if call["scenario"] == "receptionist":
            from assist.promptable import base_prompt_for
            expected = base_prompt_for(
                "receptionist_system.md.j2", domains=["SYNTHETIC PROJECT"])
            if _system_prompt(call) != expected:
                raise AssertionError("receptionist domain is not the declared fixture")

    expected_observation_keys = {
            "web-main-core": {"sandbox_commands", "tool_messages", "interrupted",
                              "email_delivery_attempted"},
            "web-main-full": {"sandbox_commands", "tool_messages", "interrupted",
                              "email_delivery_attempted"},
            "web-delegate": {"sandbox_commands", "tool_messages", "interrupted",
                             "email_delivery_attempted"},
            "legacy-main": set(),
            "skill-precedence-built-in": {
                "loaded_skill_bodies", "load_artifacts"},
            "skill-precedence-embedder": {
                "loaded_skill_bodies", "load_artifacts"},
            "context-read-only": {"tool_messages"},
            "research-lead": {"tool_messages"},
            "research-leaf-provenance": {"tool_messages"},
            "nested-research-worker": {"tool_messages"},
            "nested-fact-check": {"tool_messages"},
            "nested-report-critique": {"tool_messages"},
            "receptionist": set(),
            "thread-description": {"description"},
            "capture": set(),
            "async-task-return-contract": {"launch", "checked"},
    }
    if set(artifact["observations"]) != set(expected_observation_keys):
        raise AssertionError("census artifact has undeclared observation scenarios")
    for scenario, observation in artifact["observations"].items():
        if set(observation) != expected_observation_keys[scenario]:
            raise AssertionError(f"{scenario} has an undeclared observation field")
        for message in observation.get("tool_messages", []):
            _assert_tool_result(message)
        if "loaded_skill_bodies" in observation:
            call_id = ("synthetic-call-skill-precedence-embedder-load"
                       if scenario.endswith("embedder")
                       else "synthetic-call-skill-precedence-built-in-load")
            if observation["loaded_skill_bodies"] != [
                    _expected_tool_results()[call_id]]:
                raise AssertionError(f"{scenario} has undeclared loaded skills")
            if observation["load_artifacts"] != [
                    _expected_load_artifact(call_id)]:
                raise AssertionError(f"{scenario} has undeclared load evidence")
        if "description" in observation \
                and observation["description"] != \
                "SYNTHETIC TERMINAL thread-description 0":
            raise AssertionError("thread description is not the declared fixture")
        if scenario == "async-task-return-contract":
            launch = observation["launch"]
            checked = json.loads(observation["checked"])
            if launch != _expected_async_launch() \
                    or checked != {
                        "task_id": _expected_async_task_id(),
                        "agent_name": "research-agent",
                        "description": "SYNTHETIC RESEARCH BRIEF",
                        "status": "success",
                        "created_at": "2026-07-30T12:00:00Z",
                        "updated_at": "2026-07-30T12:00:00Z",
                        "result": "SYNTHETIC DIRECT RESEARCH FINDINGS",
                        "error": None,
                    }:
                raise AssertionError("async task return contract drifted")

        expected_commands = {
            "web-main-core": ["printf synthetic-ok"],
            "web-main-full": [],
            "web-delegate": [],
        }
        if "sandbox_commands" in observation \
                and observation["sandbox_commands"] != expected_commands[scenario]:
            raise AssertionError(f"{scenario} has undeclared sandbox commands")
        expected_tool_ids = {
            "web-main-core": ["synthetic-call-grounding-load",
                              "synthetic-call-safe-exec",
                              "synthetic-call-git-push"],
            "web-main-full": ["synthetic-call-hitl-skill-load"],
            "web-delegate": [],
            "context-read-only": ["synthetic-call-read-only",
                                  "synthetic-call-read-only-edit"],
            "research-lead": [],
            "research-leaf-provenance": ["synthetic-call-provenance"],
            "nested-research-worker": ["synthetic-call-nested-research-worker"],
            "nested-fact-check": ["synthetic-call-nested-fact-check"],
            "nested-report-critique": ["synthetic-call-nested-report-critique"],
        }
        if "tool_messages" in observation and [
                message["tool_call_id"] for message in observation["tool_messages"]
        ] != expected_tool_ids[scenario]:
            raise AssertionError(f"{scenario} has undeclared tool observations")
        if "interrupted" in observation \
                and observation["interrupted"] is not (scenario == "web-main-full"):
            raise AssertionError(f"{scenario} has an undeclared interrupt state")
        if "email_delivery_attempted" in observation \
                and observation["email_delivery_attempted"] is not False:
            raise AssertionError(f"{scenario} reached the email body")

    expected_external_skills = {
        ("web-main-full", "synthetic-domain", "/.claude/skills/"): {
            "path": "/.claude/skills/synthetic-domain/SKILL.md",
            "description": "SYNTHETIC DOMAIN DESCRIPTION",
            "content": "---\nname: synthetic-domain\ndescription: SYNTHETIC DOMAIN "
                       "DESCRIPTION\nallowed-tools: synthetic_external_tool\n"
                       "---\n\nSYNTHETIC_DOMAIN_BODY\n",
            "allowed_tools": ["synthetic_external_tool"],
        },
        ("skill-precedence-embedder", "dev", "/synthetic-embedder-skills/"): {
            "path": "/synthetic-embedder-skills/dev/SKILL.md",
            "description": "SYNTHETIC EMBEDDER DEV DESCRIPTION",
            "content": "---\nname: dev\ndescription: SYNTHETIC EMBEDDER DEV "
                       "DESCRIPTION\n---\n\n\x1b[31mSYNTHETIC_EMBEDDER_DEV_BODY"
                       "\x1b[0m\n",
            "allowed_tools": [],
        },
    }
    for source in artifact["source_manifest"]:
        if source.get("kind") != "skill":
            continue
        local_path = _packaged_skill_path(source)
        if local_path is not None:
            from deepagents.middleware.skills import _parse_skill_metadata
            content = local_path.read_text(encoding="utf-8")
            metadata = _parse_skill_metadata(
                content, source["path"], local_path.parent.name)
            if metadata is None \
                    or source["name"] != metadata["name"] \
                    or source["description"] != metadata["description"] \
                    or source["allowed_tools"] != metadata["allowed_tools"]:
                raise AssertionError("packaged skill metadata drifted")
        else:
            expected = expected_external_skills.get((
                source["scenario"], source["name"], source["source"]))
            if expected is None \
                    or source["path"] != expected["path"] \
                    or source["description"] != expected["description"] \
                    or source["allowed_tools"] != expected["allowed_tools"] \
                    or source["content_sha256"] != hashlib.sha256(
                        expected["content"].encode()).hexdigest():
                raise AssertionError("non-packaged skill is not a declared fixture")


def _assert_source_links(artifact: dict[str, Any]) -> None:
    source_ids = [source["id"] for source in artifact["source_manifest"]]
    if len(source_ids) != len(set(source_ids)):
        raise AssertionError("source manifest contains duplicate IDs")
    known = set(source_ids)
    sources = {source["id"]: source for source in artifact["source_manifest"]}
    fixed_kinds = {"package", "python-module", "python-symbol"}
    actual_fixed = sorted(
        (source for source in sources.values()
         if source["kind"] in fixed_kinds),
        key=lambda source: source["id"],
    )
    expected_fixed = [
        source for source in _source_manifest(CensusTrace(calls=artifact["calls"]))
        if source["kind"] in fixed_kinds
    ]
    if actual_fixed != expected_fixed:
        raise AssertionError("fixed prompt source manifest drifted")
    template_locators = {source["locator"] for source in sources.values()
                         if source["kind"] == "template"}
    if template_locators != _REFERENCED_TEMPLATE_LOCATORS:
        raise AssertionError("template source set drifted")
    block_chain = [
        (call["scenario"], call["call_index"],
         call["provenance"]["initial_sha256"],
         call["provenance"]["final_sha256"])
        for call in artifact["calls"]
    ]
    block_chain_sha = _sha(block_chain)
    if block_chain_sha != _DECLARED_PROMPT_BLOCK_CHAIN_SHA256:
        raise AssertionError(
            f"prompt block transition chain drifted: {block_chain_sha}")
    for source in sources.values():
        if source["kind"] == "template":
            expected_id = (
                f"template:{source['locator'].split('/templates/', 1)[0]}:"
                f"{source['locator'].split('/templates/', 1)[1]}:"
                f"{source['rendered_sha256'][:16]}")
            if source["id"] != expected_id:
                raise AssertionError("template source ID drifted")
            path = Path(__file__).resolve().parents[1] / source["locator"]
            if not path.is_file() or source["source_sha256"] != hashlib.sha256(
                    path.read_bytes()).hexdigest() \
                    or source["rendered_sha256"] not in \
                    _DECLARED_TEMPLATE_RENDER_HASHES[source["locator"]]:
                raise AssertionError(
                    f"template source hash drifted: {source['locator']}")
        if source["kind"] == "synthetic-fixture" \
                and (source["scenario"] not in EXPECTED_CALL_COUNTS
                     or source["value_sha256"] != _sha(source["value"])
                     or source["id"] != (
                         f"fixture:{source['scenario']}:{source['name']}:"
                         f"{source['value_sha256'][:16]}")):
            raise AssertionError(f"fixture source identity drifted: {source['id']}")
        if source["kind"] == "skill" \
                and (source["scenario"] not in EXPECTED_CALL_COUNTS
                     or source["description_sha256"] != _sha(source["description"])
                     or source["id"] != (
                         f"skill:{source['scenario']}:{source['name']}:"
                         f"{source['path']}")):
            raise AssertionError(f"skill source identity drifted: {source['id']}")
    referenced_all: set[str] = set()
    for call in artifact["calls"]:
        provenance = call["provenance"]
        initial_text = provenance["initial_text"]
        if provenance["initial_characters"] != len(initial_text) \
                or provenance["initial_text_sha256"] != _sha(initial_text):
            raise AssertionError(
                f"{call['scenario']} initial prompt text metadata drifted")
        final_prompt = _system_prompt(call)
        if provenance["final_text_sha256"] != _sha(final_prompt):
            raise AssertionError(
                f"{call['scenario']} final prompt text does not match provenance")
        system_messages = [message for message in call["provider_payload"]["messages"]
                           if message.get("role") == "system"]
        provider_content = system_messages[0].get("content") if system_messages else None
        provider_blocks = (provider_content if isinstance(provider_content, list)
                           else [{"type": "text", "text": provider_content}])
        if len(system_messages) != 1 \
                or _flatten_blocks(provider_blocks) != final_prompt \
                or _sha(provider_blocks) != provenance["final_sha256"]:
            raise AssertionError(
                f"{call['scenario']} provider system message does not match capture")
        references: list[str] = []
        expected_final_spans = [dict(span) for span in provenance["initial_spans"]]
        initial_spans = provenance["initial_spans"]
        if not initial_spans or initial_spans[0]["start"] != 0 \
                or initial_spans[-1]["end"] != len(initial_text) \
                or any(left["end"] != right["start"]
                       for left, right in zip(initial_spans, initial_spans[1:])):
            raise AssertionError(
                f"{call['scenario']} initial prompt attribution has gaps")
        _assert_block_layout(
            provenance["initial_block_layout"], initial_text, initial_spans)
        for span in provenance["initial_spans"]:
            references.append(span.get("source_id"))
            source = sources.get(span.get("source_id"))
            text = initial_text[span["start"]:span["end"]]
            if span["owner"] == "framework prompt composer":
                if span["source_id"] != "python:deepagents.graph" or text.strip():
                    raise AssertionError(
                        f"framework prompt composer source drifted in "
                        f"{call['scenario']}")
            elif span["owner"] == "deepagents.graph.BASE_AGENT_PROMPT":
                from deepagents.graph import BASE_AGENT_PROMPT
                if span["source_id"] != "python:deepagents.graph.BASE_AGENT_PROMPT" \
                        or text != BASE_AGENT_PROMPT:
                    raise AssertionError(
                        f"Deep Agents base prompt source drifted in "
                        f"{call['scenario']}")
            else:
                package = "edd" if span["owner"] == "edd.capture template" else "assist"
                if source is None or source["kind"] != "template" \
                        or not source["locator"].startswith(f"{package}/templates/") \
                        or _sha(text) != source["rendered_sha256"] \
                        or span.get("rendered_sha256") != source["rendered_sha256"]:
                    raise AssertionError(
                        f"template rendering link drifted in {call['scenario']}")
        for span in provenance["final_spans"]:
            if "source_id" in span:
                references.append(span["source_id"])
            else:
                references.extend(span.get("source_ids", []))
        events = call["prompt_events"]
        transitions = provenance["transitions"]
        if len(events) != len(transitions):
            raise AssertionError(f"{call['scenario']} prompt event count drifted")
        current_text = initial_text
        previous_after_sha = provenance["initial_sha256"]
        for event, transition in zip(events, transitions, strict=True):
            aligned_fields = {
                "owner", "before_sha256", "after_sha256",
                "before_text_sha256", "after_text_sha256",
                "characters_before", "characters_after",
            }
            if any(event[field] != transition[field] for field in aligned_fields):
                raise AssertionError(
                    f"{call['scenario']} prompt event/transition drifted")
            stringified = (
                transition["owner"] == "assist.ContextRiderMiddleware"
                and transition["operation"] == "replace"
                and "{'type': 'text'" in transition["exact_change"]
            )
            if event["stringified_prior_blocks"] is not stringified:
                raise AssertionError(
                    f"{call['scenario']} prompt flattening evidence drifted")
            if transition["before_sha256"] != previous_after_sha \
                    or transition["before_text_sha256"] != _sha(current_text) \
                    or transition["characters_before"] != len(current_text):
                raise AssertionError(
                    f"{call['scenario']} prompt transition input drifted")
            if transition["operation"] == "append":
                if not transition["exact_change"]:
                    raise AssertionError("empty prompt append")
                after_text = current_text + transition["exact_change"]
            elif transition["operation"] == "replace":
                after_text = transition["exact_change"]
                if after_text.startswith(current_text):
                    raise AssertionError("prompt replacement is actually an append")
            elif transition["operation"] == "unchanged":
                if transition["exact_change"]:
                    raise AssertionError("unchanged prompt has added text")
                after_text = current_text
            else:
                raise AssertionError("prompt transition has an undeclared operation")
            if transition["after_text_sha256"] != _sha(after_text) \
                    or transition["characters_after"] != len(after_text):
                raise AssertionError(
                    f"{call['scenario']} prompt transition output drifted")
            skill_sources = [source["id"] for source in sources.values()
                             if source.get("kind") == "skill"
                             and source.get("scenario") == call["scenario"]]
            if transition["owner"] == "deepagents.SkillsMiddleware":
                skill_sources.sort(key=lambda source_id: transition["exact_change"].find(
                    f"- **{sources[source_id]['name']}**: "
                    f"{sources[source_id]['description']}"))
                if skill_sources and transition["exact_change"].find(
                        f"- **{sources[skill_sources[0]]['name']}**: "
                        f"{sources[skill_sources[0]]['description']}") < 0:
                    raise AssertionError(
                        f"{call['scenario']} skill source is absent from its prompt")
            expected_segments = _content_segments(
                transition["owner"], transition["exact_change"], call["scenario"],
                skill_sources, sources, before_text=current_text,
                prior_spans=expected_final_spans,
                operation=transition["operation"])
            if transition["content_segments"] != expected_segments:
                raise AssertionError(
                    f"{call['scenario']} prompt byte ownership drifted")
            references.extend(transition["source_ids"])
            segment_source_ids = list(dict.fromkeys(
                source_id
                for segment in transition["content_segments"]
                for source_id in segment["source_ids"]))
            if transition["source_ids"] != segment_source_ids:
                raise AssertionError(
                    f"{call['scenario']} transition source summary drifted")
            for segment in transition["content_segments"]:
                references.extend(segment["source_ids"])
                if any(source_id not in known for source_id in segment["source_ids"]):
                    raise AssertionError(
                        f"{call['scenario']} has an unknown prompt source")
                segment_sources = [sources[source_id]
                                   for source_id in segment["source_ids"]]
                value = transition["exact_change"][segment["start"]:segment["end"]]
                typed = [source for source in segment_sources
                         if source["kind"] in {"skill", "synthetic-fixture"}]
                if typed and "owner" not in segment:
                    if len(segment_sources) != 1:
                        raise AssertionError(
                            f"dynamic prompt bytes are ambiguously sourced in "
                            f"{call['scenario']}")
                    source = typed[0]
                    if source["scenario"] != call["scenario"]:
                        raise AssertionError(
                            f"dynamic prompt source crosses scenarios in "
                            f"{call['scenario']}")
                    allowed = ({source["name"], source["description"]}
                               if source["kind"] == "skill"
                               else {source["value"]})
                    if value not in allowed:
                        raise AssertionError(
                            f"dynamic prompt bytes do not match {source['id']}")
            projected = [{
                "start": (transition["characters_before"] + segment["start"]
                          if transition["operation"] == "append"
                          else segment["start"]),
                "end": (transition["characters_before"] + segment["end"]
                        if transition["operation"] == "append"
                        else segment["end"]),
                "owner": segment.get("owner", transition["owner"]),
                "source_ids": segment["source_ids"],
            } for segment in transition["content_segments"]]
            if transition["operation"] == "append":
                expected_final_spans.extend(projected)
            elif transition["operation"] == "replace":
                expected_final_spans = projected
            current_text = after_text
            previous_after_sha = transition["after_sha256"]
        if current_text != final_prompt:
            raise AssertionError(f"{call['scenario']} prompt replay drifted")
        if transitions:
            if previous_after_sha != provenance["final_sha256"]:
                raise AssertionError(f"{call['scenario']} final prompt hash drifted")
        elif provenance["initial_sha256"] != provenance["final_sha256"]:
            raise AssertionError(f"{call['scenario']} static prompt hash drifted")
        if provenance["final_spans"] != expected_final_spans:
            raise AssertionError(
                f"{call['scenario']} final prompt attribution drifted")
        if provenance["final_block_layout"] != _block_layout(
                provider_blocks, expected_final_spans):
            raise AssertionError(
                f"{call['scenario']} final block provenance drifted")
        if not references or any(reference not in known for reference in references):
            raise AssertionError(
                f"{call['scenario']} has an unlinked prompt provenance source")
        referenced_all.update(references)
    if referenced_all != known:
        raise AssertionError("source manifest is not exactly reachable")


def _observed_actions(call: dict[str, Any],
                      observations: dict[str, Any]) -> dict[str, str]:
    states = {name: "unexercised" for name in call["visible_tools"]}
    observation = observations[call["scenario"]]
    for result in observation.get("tool_messages", []):
        name = result["name"]
        if name not in states:
            continue
        if result["status"] == "success":
            states[name] = "observed-permitted"
        elif name == "read_url":
            states[name] = "observed-denied-by-provenance"
        else:
            states[name] = "observed-denied"
    if "execute" in states and observation.get("sandbox_commands"):
        states["execute"] = (
            "observed-command-policy"
            if any(result["name"] == "execute" and result["status"] == "error"
                   for result in observation.get("tool_messages", []))
            else "observed-permitted")
    if "send_email" in states and observation.get("interrupted") \
            and not observation.get("email_delivery_attempted"):
        states["send_email"] = "observed-requires-approval"
    return states


def _packaged_skill_path(source: dict[str, Any]) -> Path | None:
    """Return the matching bundled file only when the recorded bytes prove it."""
    route = source.get("source")
    if source.get("kind") != "skill" or route not in _PACKAGED_SKILL_ROOTS \
            or not source.get("path", "").startswith(route):
        return None
    relative = source["path"][len(route):]
    local_root = (Path(__file__).resolve().parents[1]
                  / _PACKAGED_SKILL_ROOTS[route]).resolve()
    local_path = (local_root / relative).resolve()
    if not local_path.is_relative_to(local_root) or not local_path.is_file():
        return None
    if hashlib.sha256(local_path.read_bytes()).hexdigest() \
            != source.get("content_sha256"):
        return None
    return local_path


def _skill_tool_owners(source_manifest: list[dict[str, Any]], *,
                       scenario: str | None = None) \
        -> dict[str, list[str]]:
    """Return bundled owners whose declarations win the selected compositions."""
    owners: dict[str, list[str]] = {}
    for source in source_manifest:
        if scenario is not None and source.get("scenario") != scenario:
            continue
        if _packaged_skill_path(source) is None:
            continue
        for tool_name in source["allowed_tools"]:
            owners.setdefault(tool_name, []).append(source["name"])
    return {name: sorted(set(skill_names))
            for name, skill_names in owners.items()}


def _tool_classification(path: str, name: str, origin: str) -> str:
    fixed = {"context", "research-lead", "research-leaf",
             "nested-research-worker", "nested-fact-check",
             "nested-report-critique", "receptionist", "thread-description",
             "capture"}
    if path in fixed:
        return "fixed-role tool"
    if origin.startswith("deepagents") or name == "load_skill" \
            or origin.startswith("langchain.agents.middleware") \
            or origin == "assist.async_subagents":
        return "framework-kernel candidate"
    return "skill-scoped candidate"


def _assert_unique_tool_candidates(
        surface: str, candidates: list[dict[str, str]]) -> None:
    candidates_by_name: dict[str, list[dict[str, str]]] = {}
    for item in candidates:
        candidates_by_name.setdefault(item["name"], []).append(item)
    for name, duplicates in candidates_by_name.items():
        if len(duplicates) > 1:
            origins = ", ".join(item["origin"] for item in duplicates)
            raise AssertionError(
                f"{surface}: duplicate tool candidate `{name}` from {origins}")


def _capabilities(trace: CensusTrace, observations: dict[str, Any],
                  source_manifest: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for call in trace.calls:
        owners = _skill_tool_owners(
            source_manifest, scenario=call["scenario"])
        key = f"{call['scenario']}:{call['call_index']}"
        matches = [trace.tool_nodes[index] for index in call["matching_tool_nodes"]]
        if call["visible_tools"] and len(matches) != 1:
            raise AssertionError(
                f"{key} has {len(matches)} construction-time ToolNode matches")
        registered = matches[0]["candidates"] if matches else []
        _assert_unique_tool_candidates(key, registered)
        tools = []
        for item in registered:
            name = item["name"]
            origin = item["origin"]
            classification = _tool_classification(call["path"], name, origin)
            tools.append({
                **item,
                "classification": classification,
                "possible_owners": owners.get(name, []),
            })
        result[key] = {
            "path": call["path"],
            "registered_tools": tools,
            "model_visible_tools": call["visible_tools"],
            "ambiguous_tool_node_matches": len(matches) > 1,
            "claims": _capability_claims(call),
            "effective_actions": _observed_actions(call, observations),
        }
    return result


def _validate_skill_tool_declarations(
        capabilities: dict[str, Any], source_manifest: list[dict[str, Any]]) -> None:
    """Validate winning bundled declarations against recorded compositions."""
    global_owners = _skill_tool_owners(source_manifest)
    skill_sources = [
        source for source in source_manifest
        if _packaged_skill_path(source) is not None
    ]
    for source in skill_sources:
        winning_scenarios = {
            candidate["scenario"] for candidate in skill_sources
            if candidate["name"] == source["name"]
            and candidate["path"] == source["path"]
        }
        surfaces = [surface for key, surface in capabilities.items()
                    if key.split(":", 1)[0] in winning_scenarios]
        for name in source["allowed_tools"]:
            matches = [tool for surface in surfaces
                       for tool in surface["registered_tools"]
                       if tool["name"] == name]
            if not matches:
                raise AssertionError(
                    f"{source['path']}: unknown declared tool `{name}`")
            if all(tool["classification"] != "skill-scoped candidate"
                   or not tool["origin"].startswith("assist.")
                   for tool in matches):
                raise AssertionError(
                    f"{source['path']}: non-bundled or kernel tool `{name}` "
                    "must not be declared")

    for key, surface in capabilities.items():
        for tool in surface["registered_tools"]:
            if tool["classification"] != "skill-scoped candidate":
                continue
            if not tool["origin"].startswith("assist.") \
                    or tool["name"] == _BUNDLED_OWNER_EXCEPTION:
                continue
            owners = global_owners.get(tool["name"], [])
            if not owners:
                raise AssertionError(
                    f"{key}: non-kernel tool `{tool['name']}` has no winning owner")
            if len(owners) > 1:
                raise AssertionError(
                    f"{key}: non-kernel tool `{tool['name']}` has multiple owners: "
                    f"{', '.join(owners)}")


def _findings(trace: CensusTrace, capabilities: dict[str, Any],
              observations: dict[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for key, surface in capabilities.items():
        if not key.endswith(":0"):
            continue
        for tool in surface["registered_tools"]:
            if tool["classification"] == "skill-scoped candidate" \
                    and not tool["possible_owners"]:
                findings.append({
                    "kind": "unowned-tool",
                    "surface": key,
                    "detail": tool["name"],
                })
            elif tool["classification"] == "skill-scoped candidate" \
                    and len(tool["possible_owners"]) > 1:
                findings.append({
                    "kind": "multiply-owned-tool",
                    "surface": key,
                    "detail": (
                        f"{tool['name']}: "
                        f"{', '.join(tool['possible_owners'])}"),
                })
    for call in trace.calls:
        surface_key = f"{call['scenario']}:{call['call_index']}"
        claims = capabilities[surface_key]["claims"]
        actions = capabilities[surface_key]["effective_actions"]
        claims_by_tool: dict[str, list[dict[str, str]]] = {}
        for claim in claims:
            claims_by_tool.setdefault(claim["tool"], []).append(claim)
        for tool, tool_claims in claims_by_tool.items():
            polarities = {claim["polarity"] for claim in tool_claims}
            if {"positive", "negative"} <= polarities:
                owners = sorted({claim["owner"] for claim in tool_claims
                                 if claim["polarity"] in {"positive", "negative"}})
                findings.append({
                    "kind": "contradictory-capability-claims",
                    "surface": surface_key,
                    "detail": (f"`{tool}` is described as both available and "
                               f"unavailable by {', '.join(owners)}."),
                })
        for claim in claims:
            if claim["polarity"] == "argument-relative":
                schema = next(
                    (item for item in call["provider_payload"].get("tools", [])
                     if item["function"]["name"] == claim["tool"]), None)
                if schema and "absolute, not relative" in json.dumps(
                        schema["function"]["parameters"]).lower():
                    findings.append({
                        "kind": "argument-contract-mismatch",
                        "surface": surface_key,
                        "detail": (
                            f"{claim['owner']} requires relative paths for "
                            f"`{claim['tool']}`, but its provider schema requires "
                            "an absolute path."),
                    })
            if claim["polarity"] == "positive" \
                    and claim["tool"] not in call["visible_tools"]:
                findings.append({
                    "kind": "unavailable-capability-claim",
                    "surface": surface_key,
                    "detail": (f"{claim['owner']} positively requires "
                               f"`{claim['tool']}`, but it is not visible."),
                })
            elif claim["polarity"] == "positive" \
                    and actions.get(claim["tool"]) == "observed-denied":
                findings.append({
                    "kind": "denied-capability-claim",
                    "surface": surface_key,
                    "detail": (f"{claim['owner']} positively requires "
                               f"`{claim['tool']}`, but enforcement denies it."),
                })
            if claim["polarity"] == "negative" and claim["tool"] in actions \
                    and actions[claim["tool"]] == "observed-permitted":
                findings.append({
                    "kind": "contradictory-negative-claim",
                    "surface": surface_key,
                    "detail": (f"{claim['owner']} says `{claim['tool']}` is unavailable, "
                               "but it is visible."),
                })
    full_call = next(call for call in trace.calls
                     if call["scenario"] == "web-main-full" and call["call_index"] == 0)
    if any(
            transition["owner"] == "assist.ContextRiderMiddleware"
            and transition["operation"] == "replace"
            and "{'type': 'text'" in transition["exact_change"]
            for transition in full_call["provenance"]["transitions"]):
        findings.append({
            "kind": "prompt-flattening",
            "surface": "web-main-full:0",
            "detail": "ContextRiderMiddleware stringifies prior content blocks.",
        })
    async_result = json.loads(
        observations["async-task-return-contract"]["checked"])["result"]
    for scenario in ("web-main-core", "web-main-full"):
        call = next(call for call in trace.calls
                    if call["scenario"] == scenario
                    and call["call_index"] == 0)
        marker = "saved to a file; the path is returned"
        if marker in _system_prompt(call) \
                and async_result == "SYNTHETIC DIRECT RESEARCH FINDINGS":
            findings.append({
                "kind": "return-shape-claim-mismatch",
                "surface": f"{scenario}:0",
                "detail": (
                    "The main prompt guarantees a saved-report path, but the "
                    "observed async task contract returns the child result directly."),
            })
    return findings


def _assert_semantic_views(artifact: dict[str, Any]) -> None:
    nodes = artifact["tool_nodes"]
    for node in nodes:
        _assert_unique_tool_candidates(
            f"{node['scenario']}:tool-node:{node['index']}",
            node["candidates"],
        )
    if _sha(nodes) != _DECLARED_TOOL_NODE_HISTORY_SHA256:
        raise AssertionError("construction-time ToolNode history drifted")
    if [node["index"] for node in nodes] != list(range(len(nodes))):
        raise AssertionError("tool node indices drifted")
    for node in nodes:
        if node["scenario"] not in EXPECTED_CALL_COUNTS:
            raise AssertionError("tool node has an undeclared scenario")
        if node["winners"] != list(dict.fromkeys(
                candidate["name"] for candidate in node["candidates"])):
            raise AssertionError("tool node winners drifted from candidates")
        for candidate in node["candidates"]:
            if _TOOL_ORIGINS.get(candidate["name"]) != candidate["origin"]:
                raise AssertionError(
                    f"tool origin drifted: {candidate['name']}")
    for call in artifact["calls"]:
        visible = call["visible_tools"]
        matching = _matching_tool_nodes(nodes, call["scenario"], visible)
        if call["matching_tool_nodes"] != [node["index"] for node in matching]:
            raise AssertionError("provider call ToolNode matches drifted")
    trace = CensusTrace(calls=artifact["calls"], tool_nodes=nodes)
    expected_capabilities = _capabilities(
        trace, artifact["observations"], artifact["source_manifest"])
    if artifact["capabilities"] != expected_capabilities:
        raise AssertionError("capability surfaces drifted from recorded evidence")
    _validate_skill_tool_declarations(
        expected_capabilities, artifact["source_manifest"])
    if artifact["findings"] != _findings(
            trace, expected_capabilities, artifact["observations"]):
        raise AssertionError("findings drifted from recorded evidence")


def _assert_hygiene(artifact: dict[str, Any], synthetic_root: Path) -> None:
    _assert_declared_inputs(artifact)
    _assert_source_links(artifact)
    _assert_semantic_views(artifact)
    raw = json.dumps(artifact, ensure_ascii=False, sort_keys=True)
    forbidden = [
        str(synthetic_root.resolve()),
        str(Path.home().resolve()),
        "/home/",
        "/var/",
        "/root/",
        "/etc/",
        "/tmp/",
    ]
    for value in forbidden:
        if value and value in raw:
            raise AssertionError(f"census leaked host path: {value}")
    secret_patterns = (
        r"\bsk-[A-Za-z0-9_-]{20,}\b",
        r"\bghp_[A-Za-z0-9]{20,}\b",
        r"BEGIN OPENSSH PRIVATE KEY",
        r"\bre_live_[A-Za-z0-9_]{16,}\b",
        r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
        r"\bBearer\s+[A-Za-z0-9._~+/-]{12,}=*",
        r"(?:\+1[ .-]+)?\(?\d{3}\)?[ .-]+\d{3}[ .-]+\d{4}\b",
    )
    for pattern in secret_patterns:
        if re.search(pattern, raw):
            raise AssertionError(f"census may contain a secret pattern: {pattern}")
    for key, value in os.environ.items():
        if re.search(r"(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|COOKIE)", key) \
                and len(value) >= 8 and value in raw:
            raise AssertionError(f"census contains configured secret from {key}")
    if "- general-purpose:" in raw:
        index = raw.index("- general-purpose:")
        raise AssertionError(
            "OpenAI harness profile was not preserved: general-purpose resurfaced near "
            f"{raw[max(0, index - 120):index + 180]!r}")
    if "artifact_sha256" in artifact:
        unsigned = {key: value for key, value in artifact.items()
                    if key != "artifact_sha256"}
        if artifact["artifact_sha256"] != _sha(unsigned):
            raise AssertionError("census artifact fingerprint drifted")


def capture_census() -> dict[str, Any]:
    """Run the process-global census observers in a dedicated subprocess."""
    artifact = _run_isolated_json("--stdout-census")
    _assert_hygiene(artifact, Path("/synthetic-parent-validation-root"))
    return artifact


def capture_prompt_rewrite_profiles() -> dict[str, dict[str, Any]]:
    """Capture the exact legacy and candidate eval-helper prompt profiles.

    The production census intentionally constructs ThreadManager and therefore
    observes only the deployed web-main composition. Prompt-rewrite behavioral
    comparisons instead use ``prompt_rewrite_web_main_spec``. This small,
    deterministic companion capture uses that same helper for both environment
    profiles so a result ledger can identify each static prompt honestly.
    """
    from assist.agent import create_agent
    from assist.promptable import env
    from edd.eval.utils import prompt_rewrite_web_main_spec

    profiles: dict[str, dict[str, Any]] = {}
    original_clock = env.globals["current_datetime"]
    with tempfile.TemporaryDirectory(prefix="assist-prompt-rewrite-profiles-") as tmp:
        root = Path(tmp)
        env.globals["current_datetime"] = lambda: FIXED_NOW
        try:
            for name, candidate in (("baseline", False), ("candidate", True)):
                trace = CensusTrace()
                workspace = root / name / "workspace"
                agent_dir = root / name / "agent"
                workspace.mkdir(parents=True)
                agent_dir.mkdir()
                environment = ({"ASSIST_PROMPT_REWRITE_CANDIDATE": "1"}
                               if candidate else {})
                with patch.dict(os.environ, environment, clear=False):
                    if not candidate:
                        os.environ.pop("ASSIST_PROMPT_REWRITE_CANDIDATE", None)
                    with _instrument(trace), _scenario("observer-probe"):
                        agent = create_agent(
                            RecordingChatModel(trace), str(workspace),
                            agent_dir=str(agent_dir),
                            spec=prompt_rewrite_web_main_spec())
                        _invoke_agent(agent, "observer-probe")
                if trace.faults:
                    raise trace.faults[0]
                if len(trace.calls) != 1:
                    raise AssertionError(
                        f"prompt-rewrite {name} profile made {len(trace.calls)} calls")
                call = trace.calls[0]
                profiles[name] = {
                    "candidate_env": candidate,
                    "initial_text": call["provenance"]["initial_text"],
                    "final_text": _system_prompt(call),
                    "initial_sha256": call["provenance"]["initial_sha256"],
                    "final_sha256": call["provenance"]["final_sha256"],
                    "visible_tools": call["visible_tools"],
                    "transition_owners": [
                        event["owner"] for event in call["prompt_events"]],
                }
        finally:
            env.globals["current_datetime"] = original_clock
    return profiles


def capture_prompt_rewrite_guidance_skill_profiles() -> dict[str, dict[str, Any]]:
    """Capture pre- and post-migration web-main eval-helper profiles.

    Unlike the historical layout comparison above, both sides use the ordinary
    web-main prompt composition. The only profile switch is whether the
    candidate-only grounding/research prompt source is mounted.
    """
    from assist.agent import create_agent
    from assist.promptable import env
    from edd.eval.utils import prompt_rewrite_web_main_spec

    profiles: dict[str, dict[str, Any]] = {}
    original_clock = env.globals["current_datetime"]
    with tempfile.TemporaryDirectory(prefix="assist-guidance-skill-profiles-") as tmp:
        root = Path(tmp)
        env.globals["current_datetime"] = lambda: FIXED_NOW
        try:
            for name, enabled in (("baseline", False), ("candidate", True)):
                trace = CensusTrace()
                workspace = root / name / "workspace"
                agent_dir = root / name / "agent"
                workspace.mkdir(parents=True)
                agent_dir.mkdir()
                with patch.dict(
                        os.environ,
                        {"ASSIST_PROMPT_REWRITE_GUIDANCE_SKILLS": str(int(enabled))},
                        clear=False), _instrument(trace), _scenario("observer-probe"):
                    agent = create_agent(
                        RecordingChatModel(trace), str(workspace),
                        agent_dir=str(agent_dir),
                        spec=prompt_rewrite_web_main_spec())
                    _invoke_agent(agent, "observer-probe")
                if trace.faults:
                    raise trace.faults[0]
                if len(trace.calls) != 1:
                    raise AssertionError(
                        f"guidance-skill {name} profile made {len(trace.calls)} calls")
                call = trace.calls[0]
                profiles[name] = {
                    "guidance_skills_env": enabled,
                    "initial_text": call["provenance"]["initial_text"],
                    "final_text": _system_prompt(call),
                    "initial_sha256": call["provenance"]["initial_sha256"],
                    "final_sha256": call["provenance"]["final_sha256"],
                    "visible_tools": call["visible_tools"],
                    "transition_owners": [
                        event["owner"] for event in call["prompt_events"]],
                }
        finally:
            env.globals["current_datetime"] = original_clock
    return profiles


def _capture_census() -> dict[str, Any]:
    trace = CensusTrace()
    observations: dict[str, Any] = {}
    from assist.promptable import env

    with tempfile.TemporaryDirectory(prefix="assist-prompt-census-") as tmp:
        root = Path(tmp)
        original_clock = env.globals["current_datetime"]
        env.globals["current_datetime"] = lambda: FIXED_NOW
        try:
            with _instrument(trace):
                observations["web-main-core"] = _invoke_web(trace, root, full=False)
                observations["web-main-full"] = _invoke_web(trace, root, full=True)
                observations["web-delegate"] = _invoke_web(
                    trace, root, full=False, delegate=True)
                observations["legacy-main"] = _invoke_legacy(trace, root)
                observations["skill-precedence-built-in"] = _invoke_skill_precedence(
                    trace, root, embedder=False)
                observations["skill-precedence-embedder"] = _invoke_skill_precedence(
                    trace, root, embedder=True)
                observations["context-read-only"] = _invoke_context(trace, root)
                observations["research-lead"] = _invoke_research(
                    trace, root, leaf=False, scenario="research-lead")
                observations["research-leaf-provenance"] = _invoke_research(
                    trace, root, leaf=True, scenario="research-leaf-provenance")
                for scenario in (
                        "nested-research-worker", "nested-fact-check",
                        "nested-report-critique"):
                    observations[scenario] = _invoke_research(
                        trace, root, leaf=False, scenario=scenario)
                observations["receptionist"] = _invoke_receptionist(trace)
                observations["thread-description"] = _invoke_description(trace)
                observations["capture"] = _invoke_capture(trace, root)
                observations["async-task-return-contract"] = \
                    _invoke_async_return_contract()
        finally:
            env.globals["current_datetime"] = original_clock

        if trace.faults:
            raise trace.faults[0]
        source_manifest = _source_manifest(trace)
        capabilities = _capabilities(trace, observations, source_manifest)
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "fixed_clock": FIXED_NOW,
            "source_manifest": source_manifest,
            "calls": trace.calls,
            "tool_nodes": trace.tool_nodes,
            "capabilities": capabilities,
            "observations": observations,
            "findings": _findings(trace, capabilities, observations),
        }
        paths = {call["path"] for call in trace.calls}
        missing = REQUIRED_PATHS - paths
        if missing:
            raise AssertionError(f"census missed required paths: {sorted(missing)}")
        _assert_hygiene(artifact, root)
        artifact["artifact_sha256"] = _sha(artifact)
        return artifact


def observer_probe(*, instrumented: bool) -> dict[str, Any]:
    """Return one isolated provider request with observers on or off."""
    mode = "instrumented" if instrumented else "plain"
    payload = _run_isolated_json("--stdout-observer", mode)
    _assert_observer_payload(payload)
    return payload


def _assert_observer_payload(payload: dict[str, Any]) -> None:
    expected_keys = {
        "extra_body", "messages", "model", "stream", "temperature", "tools"}
    if set(payload) != expected_keys \
            or payload["model"] != "synthetic-qwen-census" \
            or payload["stream"] is not False \
            or payload["temperature"] != 0.1 \
            or payload["extra_body"] != {
                "chat_template_kwargs": {"enable_thinking": False}}:
        raise AssertionError("isolated observer provider settings drifted")
    messages = payload["messages"]
    if not isinstance(messages, list) or len(messages) != 2 \
            or messages[0].get("role") != "system" \
            or not isinstance(messages[0].get("content"), list) \
            or not messages[0]["content"] \
            or messages[1] != {
                "role": "user", "content": "SYNTHETIC OBSERVER PROBE"}:
        raise AssertionError("isolated observer message shape drifted")
    tools = payload["tools"]
    expected_tools = [
        "write_todos", "ls", "read_file", "write_file", "edit_file", "glob",
        "grep", "load_skill", "map_data",
    ]
    if not isinstance(tools, list) or [
            tool.get("function", {}).get("name") for tool in tools
    ] != expected_tools:
        raise AssertionError("isolated observer tool shape drifted")
    for tool in tools:
        if set(tool) != {"type", "function"} or tool["type"] != "function" \
                or set(tool["function"]) != {
                    "name", "description", "parameters"}:
            raise AssertionError("isolated observer tool schema drifted")
        _assert_json_schema(tool["function"]["parameters"])


def _run_isolated_json(*arguments: str) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="assist-prompt-parent-") as scratch:
        child_env = {
            "HOME": str(Path.home()),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONPATH": str(repo),
            "TMPDIR": scratch,
        }
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "edd.prompt_census", *arguments],
                check=True,
                capture_output=True,
                timeout=120,
                env=child_env,
                cwd=repo,
            )
        except subprocess.TimeoutExpired as exc:
            stderr = (exc.stderr or b"")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            tail = stderr[-4_000:].strip() or "no child diagnostic"
            raise RuntimeError(
                f"isolated prompt census timed out after {exc.timeout}s: "
                f"{tail}") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            tail = stderr[-4_000:].strip() or "no child diagnostic"
            raise RuntimeError(
                f"isolated prompt census exited with status "
                f"{exc.returncode}: {tail}") from exc
        try:
            artifact = json.loads(completed.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "isolated prompt census returned malformed JSON") from exc
        if not isinstance(artifact, dict):
            raise RuntimeError(
                "isolated prompt census returned a non-object JSON value")
        return artifact


def _observer_probe(*, instrumented: bool) -> dict[str, Any]:
    from assist.agent import create_agent
    from assist.promptable import env
    from assist.spec import AgentSpec

    trace = CensusTrace()
    original_clock = env.globals["current_datetime"]
    env.globals["current_datetime"] = lambda: FIXED_NOW
    try:
        with tempfile.TemporaryDirectory(prefix="assist-prompt-observer-") as tmp, \
                _scenario("observer-probe"):
            if instrumented:
                with _instrument(trace):
                    agent = create_agent(
                        RecordingChatModel(trace), tmp,
                        spec=AgentSpec(async_subagent_tools=()))
                    agent.invoke(
                        {"messages": [HumanMessage(content="SYNTHETIC OBSERVER PROBE")]},
                        {"configurable": {"thread_id": "synthetic-observer"}},
                    )
            else:
                agent = create_agent(
                    RecordingChatModel(trace, attribute=False), tmp,
                    spec=AgentSpec(async_subagent_tools=()))
                agent.invoke(
                    {"messages": [HumanMessage(content="SYNTHETIC OBSERVER PROBE")]},
                    {"configurable": {"thread_id": "synthetic-observer"}},
                )
    finally:
        env.globals["current_datetime"] = original_clock
    if trace.faults:
        raise trace.faults[0]
    if len(trace.calls) != 1:
        raise AssertionError(
            f"observer probe recorded {len(trace.calls)} provider calls")
    return trace.calls[0]["provider_payload"]


def artifact_bytes(artifact: dict[str, Any]) -> bytes:
    return (json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n").encode()


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing an existing label."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is required for no-replace captures")
    result = renameat2(
        -100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(error, os.strerror(error), destination)
    raise OSError(error, os.strerror(error), destination)


def _fsync(path: Path) -> None:
    flags = os.O_RDONLY | (os.O_DIRECTORY if path.is_dir() else 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_capture(label: str, *, output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    """Atomically publish one bounded directory under a new label."""
    if not label or label in {".", ".."} or "/" in label or "\\" in label:
        raise ValueError("label must be one path segment")
    output_root = output_root.expanduser()
    destination = output_root / label
    if destination.exists():
        raise FileExistsError(f"capture already exists: {destination}")
    if not output_root.is_dir():
        raise FileNotFoundError(
            f"capture output root must already exist: {output_root}")
    # Preflight the filesystem's directory-sync contract before the no-replace
    # label can be consumed.  Requiring a provisioned root also makes the root
    # entry itself an operator-owned durability concern, not part of publication.
    _fsync(output_root)
    staging = Path(tempfile.mkdtemp(prefix=f".{label}-", dir=output_root))
    try:
        artifact = capture_census()
        raw = artifact_bytes(artifact)
        main = next(call for call in artifact["calls"]
                    if call["scenario"] == "web-main-core" and call["call_index"] == 0)
        main_raw = _system_prompt(main).encode()
        if len(raw) + len(main_raw) > MAX_RUN_BYTES:
            raise ValueError(
                f"capture is {len(raw) + len(main_raw)} bytes; "
                f"limit is {MAX_RUN_BYTES}")
        census_path = staging / "census.json"
        bootstrap_path = staging / "web-main-bootstrap.txt"
        census_path.write_bytes(raw)
        bootstrap_path.write_bytes(main_raw)
        _fsync(census_path)
        _fsync(bootstrap_path)
        _fsync(staging)
        _rename_no_replace(staging, destination)
        try:
            _fsync(output_root)
        except OSError as exc:
            raise RuntimeError(
                f"capture was published at {destination}, but directory "
                "durability is uncertain") from exc
        return destination
    except BaseException as primary:
        if staging.exists():
            try:
                shutil.rmtree(staging)
                _fsync(output_root)
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    f"capture failed and staging cleanup failed at {staging}",
                    [primary, cleanup],
                )
        raise


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", nargs="?",
                        help="new phase label, e.g. p0-baseline")
    parser.add_argument("--stdout-census", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--stdout-observer", choices=("instrumented", "plain"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stdout_census:
        if args.label or args.stdout_observer:
            parser.error("isolated census mode does not accept a label")
        sys.stdout.buffer.write(artifact_bytes(_capture_census()))
        return 0
    if args.stdout_observer:
        if args.label:
            parser.error("isolated observer mode does not accept a label")
        payload = _observer_probe(instrumented=args.stdout_observer == "instrumented")
        sys.stdout.buffer.write((json.dumps(
            payload, ensure_ascii=False, sort_keys=True) + "\n").encode())
        return 0
    if not args.label:
        parser.error("label is required")
    path = write_capture(args.label)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
