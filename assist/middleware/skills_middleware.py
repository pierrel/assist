"""Small-model skill loading and bundled-tool progressive disclosure.

The model always sees the skill catalog and ``load_skill``. Tools declared by
winning packaged Assist skills are withheld until that exact skill loads
successfully in the current graph invocation. Domain and embedder skill/tool
sources remain baseline-visible until P2b.3.
"""
from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import Awaitable, Callable, Iterable
from typing import Annotated, Any, NotRequired, TypedDict

import deepagents.middleware.skills as deepagents_skills
from deepagents.middleware.skills import (
    MAX_SKILL_FILE_SIZE,
    SkillsMiddleware,
    SkillsState,
)
from langchain.agents.middleware.types import (
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
    ToolCallRequest,
)
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command, Overwrite

from assist.middleware.output_sanitization import _sanitize


_LOAD_ARTIFACT_SCHEMA = "assist.skill-load.v1"

SMALL_MODEL_SKILLS_PROMPT = """

## Skills

You have access to named skills. Each skill is a self-contained set of
rules for a specific domain. The list below shows each skill's *name*
and *description*; the rules themselves are revealed only when you load
the skill.

**Available skills:**

{skills_list}

### How to use a skill

1. **Match.** If a skill's description fits what you're about to do, continue to step 2.
2. **Load.** Call `load_skill(name="<skill name>")`. Make it the only tool call in that response and wait for its result.
3. **Apply.** On the next response, use the newly available tools and the loaded rules.

The descriptions only summarize *when* to load. You will not know the rules
until the load completes. A referential follow-up such as "make it 8 instead"
still matches the skill for the active task established by the conversation.

### Pre-action check (MANDATORY — apply on every turn before any tool call)

Before issuing your first tool call on a turn, scan the user's latest message
and its active task against every skill description above. If a skill matches,
call only `load_skill(name="<matched skill>")` in that response. Do not combine
it with `ls`, `read_file`, `task`, `edit_file`, or a newly disclosed tool. Only
if no skill matches may you proceed directly to the task.

Tools available for this response: {tools_available}.
"""


class SmallModelSkillsState(SkillsState):
    """Private activation state for one graph invocation."""

    loaded_skill_tools: NotRequired[
        Annotated[frozenset[str], PrivateStateAttr, operator.or_]
    ]


class _LoadArtifact(TypedDict):
    schema: str
    requested_name: str
    winner_fingerprint: str
    result_sha256: str


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_contains(source: str, path: str) -> bool:
    return path.startswith(f"{source.rstrip('/')}/")


def _tool_name(tool_value: BaseTool | dict[str, Any]) -> str | None:
    """Return the exact provider tool name from a final request entry."""
    if isinstance(tool_value, BaseTool):
        return tool_value.name
    name = tool_value.get("name")
    if isinstance(name, str):
        return name
    function = tool_value.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    return None


def _winner(state: dict[str, Any], requested_name: str) -> dict[str, Any] | None:
    return next(
        (skill for skill in state.get("skills_metadata", ())
         if skill.get("name") == requested_name),
        None,
    )


def _load_failure(name: str, reason: str | None = None) -> str:
    detail = f" ({reason})" if reason else ""
    return (
        f"Skill '{name}' could not be loaded{detail}. The system prompt's "
        "'## Skills' section lists every available name; use one of those."
    )


def _make_load_skill_tool(middleware: "SmallModelSkillsMiddleware"):
    """Build the exact-winner loader bound to this middleware instance."""

    @tool
    def load_skill(name: str, runtime: ToolRuntime) -> str | Command:
        """Load and return the full body of the named skill.

        Use this whenever a skill description matches your task. Pass
        only the skill's short name (e.g. "org-format") — no paths.
        Returns the full body of the skill, including the rules you
        must follow before continuing with the task.
        """
        skill = _winner(runtime.state, name)
        if skill is None:
            return _load_failure(name)

        backend = middleware._get_backend(  # noqa: SLF001 - subclass-owned seam
            runtime.state, runtime, runtime.config)
        try:
            responses = backend.download_files([skill["path"]])
        except Exception as exc:
            return _load_failure(name, type(exc).__name__)
        if len(responses) != 1:
            return _load_failure(name)
        response = responses[0]
        if response.error or response.content is None:
            return _load_failure(name, response.error)
        if len(response.content) > MAX_SKILL_FILE_SIZE:
            return _load_failure(name, "skill file is too large")
        try:
            body = _sanitize(response.content.decode("utf-8"))
        except UnicodeDecodeError:
            return _load_failure(name, "skill file is not UTF-8")

        newly_available = middleware._bundled_declared_tools(runtime.state, skill)
        already_loaded = frozenset(runtime.state.get("loaded_skill_tools", ()))
        new_names = sorted(newly_available - already_loaded)
        disclosure = (
            "Newly available tools: " + ", ".join(new_names) + "."
            if new_names else "No additional tools became available."
        )
        content = f"{body}\n\n{disclosure}"
        fingerprint_payload = {
            "allowed_tools": list(skill.get("allowed_tools", ())),
            "body_sha256": _sha256(body),
            "description": skill["description"],
            "name": skill["name"],
        }
        artifact: _LoadArtifact = {
            "schema": _LOAD_ARTIFACT_SCHEMA,
            "requested_name": name,
            "winner_fingerprint": _sha256(json.dumps(
                fingerprint_payload, ensure_ascii=False, separators=(",", ":"),
                sort_keys=True)),
            "result_sha256": _sha256(content),
        }
        message = ToolMessage(
            content=content,
            artifact=artifact,
            name="load_skill",
            status="success",
            tool_call_id=runtime.tool_call_id,
        )
        return Command(update={
            "messages": [message],
            "loaded_skill_tools": newly_available,
        })

    return load_skill


class SmallModelSkillsMiddleware(SkillsMiddleware):
    """Name-based skill loader with invocation-local bundled-tool disclosure."""

    state_schema = SmallModelSkillsState

    def __init__(self, *, backend, sources, bundled_sources: Iterable[str] = ()):
        super().__init__(backend=backend, sources=sources)
        self.system_prompt_template = SMALL_MODEL_SKILLS_PROMPT
        self._bundled_sources = frozenset(bundled_sources)
        self.tools = [_make_load_skill_tool(self)]

    def _format_skills_list(self, skills):
        """Render only name and description; declarations remain undisclosed."""
        if not skills:
            return "(No skills available.)"
        return "\n".join(
            f"- **{skill['name']}**: {skill['description']}" for skill in skills
        )

    def _is_bundled(self, skill: dict[str, Any]) -> bool:
        path = skill.get("path")
        return isinstance(path, str) and any(
            _source_contains(source, path) for source in self._bundled_sources)

    def _bundled_declared_tools(
            self, state: dict[str, Any], skill: dict[str, Any]) -> frozenset[str]:
        if not self._is_bundled(skill) or _winner(state, skill["name"]) is not skill:
            return frozenset()
        return frozenset(skill.get("allowed_tools", ()))

    def _gated_tools(self, state: dict[str, Any]) -> frozenset[str]:
        gated: set[str] = set()
        for skill in state.get("skills_metadata", ()):
            if self._is_bundled(skill):
                gated.update(skill.get("allowed_tools", ()))
        return frozenset(gated)

    def _tool_is_allowed(self, state: dict[str, Any], name: str) -> bool:
        return name not in self._gated_tools(state) or name in state.get(
            "loaded_skill_tools", ())

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        """Add the catalog and expose exactly the tools allowed for this request."""
        retained = [
            tool_value for tool_value in request.tools
            if (name := _tool_name(tool_value)) is None
            or self._tool_is_allowed(request.state, name)
        ]
        names = [name for tool_value in retained
                 if (name := _tool_name(tool_value)) is not None]
        skills_section = self.system_prompt_template.format(
            skills_locations=self._format_skills_locations(),
            skills_load_warnings=self._format_skills_load_warnings(
                request.state.get("skills_load_errors", [])),
            skills_list=self._format_skills_list(
                request.state.get("skills_metadata", [])),
            tools_available=", ".join(names) if names else "(none)",
        )
        return request.override(
            tools=retained,
            system_message=deepagents_skills.append_to_system_message(
                request.system_message, skills_section),
        )

    def before_agent(self, state, runtime, config):
        update = super().before_agent(state, runtime, config) or {}
        return {**update, "loaded_skill_tools": Overwrite(frozenset())}

    async def abefore_agent(self, state, runtime, config):
        update = await super().abefore_agent(state, runtime, config) or {}
        return {**update, "loaded_skill_tools": Overwrite(frozenset())}

    def _hidden_call_message(self, tool_call: dict[str, Any]) -> ToolMessage:
        return ToolMessage(
            content=(
                f"Tool '{tool_call['name']}' is unavailable in this response. "
                "Load its matching skill, wait for the result, and call the tool "
                "in a later response."
            ),
            name=tool_call["name"],
            status="error",
            tool_call_id=tool_call["id"],
        )

    def after_model(self, state, runtime):
        """Reject hidden hallucinated calls before HITL sees outward effects."""
        last_ai = next(
            (message for message in reversed(state.get("messages", ()))
             if isinstance(message, AIMessage)),
            None,
        )
        if last_ai is None or not last_ai.tool_calls:
            return None
        allowed = []
        rejected = []
        for tool_call in last_ai.tool_calls:
            if self._tool_is_allowed(state, tool_call["name"]):
                allowed.append(tool_call)
            else:
                rejected.append(self._hidden_call_message(tool_call))
        if not rejected:
            return None
        last_ai.tool_calls = allowed
        return {"messages": [last_ai, *rejected]}

    async def aafter_model(self, state, runtime):
        return self.after_model(state, runtime)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        if not self._tool_is_allowed(request.state, request.tool_call["name"]):
            return self._hidden_call_message(request.tool_call)
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest], Awaitable[ToolMessage | Command]
        ],
    ) -> ToolMessage | Command:
        if not self._tool_is_allowed(request.state, request.tool_call["name"]):
            return self._hidden_call_message(request.tool_call)
        return await handler(request)
