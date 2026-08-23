"""Small-model skill loading and progressive tool disclosure.

The model always sees the skill catalog and ``load_skill``. In normal
progressive compositions, tools declared by the winning skill from any
mounted source are withheld until that exact skill loads successfully. Their
compact callable schemas remain available until the catalog changes. Inbound
SMS triage deliberately retains its legacy composition.
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
    PrivateStateAttr,
    ToolCallRequest,
    hook_config,
)
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_core.utils.function_calling import convert_to_openai_tool
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
and its active task against every skill description above. A matching skill may
already have tools available; use them directly. Reload the skill when you need
operating guidance that is no longer in context. If a skill must be
loaded, call only `load_skill(name="<matched skill>")` in that response. Do not
combine it with `ls`, `read_file`, `task`, `edit_file`, or a newly disclosed
tool. Only if no skill matches may you proceed directly to the task.

Tools available for this response: {tools_available}.
"""

# Compatibility boundary: this historical prompt and the legacy loader's tool
# docstring call the complete SKILL.md file a "body". Inbound SMS triage must
# remain byte-for-byte on that pre-P2b prompt and schema, so keep the wording.
LEGACY_SMALL_MODEL_SKILLS_PROMPT = """

## Skills

You have access to named skills. Each skill is a self-contained set of
rules for a specific domain. The list below shows each skill's *name*
and *description*; the rules themselves are revealed only when you load
the skill.

**Available skills:**

{skills_list}

### How to use a skill

1. **Match.** If a skill's description fits what you're about to do, continue to step 2.
2. **Load.** Call `load_skill(name="<skill name>")`. The tool returns the full skill body.
3. **Apply.** Use the rules from the loaded skill when you compose your action or response.

The descriptions only summarize *when* to load — they do not contain
the rules. You will not know the rules until you complete step 2. You
MUST complete step 2 before performing the matching action; relying on
the description alone leads to incorrect outcomes.

### Pre-action check (MANDATORY — apply on every turn before any tool call)

Before issuing your first tool call on a turn, scan the user's latest
message against every skill description above:

1. Look for any keyword, file extension, filename, or topic from a
   skill description that appears in the user's message.
2. If a skill matches, your FIRST tool call this turn MUST be
   `load_skill(name="<matched skill>")`. Do not run `ls`, `read_file`,
   `task` to a sub-agent, or `edit_file` first — those steps come AFTER
   the skill is loaded.
3. Only if no skill matches may you proceed directly to your task.

Skipping this check is a bug. The skill exists precisely because acting
without it produces incorrect output for that domain.
"""


class SmallModelSkillsState(SkillsState):
    """Private catalog and activation state for the current thread."""

    skills_catalog_fingerprint: NotRequired[Annotated[str, PrivateStateAttr]]
    """Version of the mounted skill sources used to produce the catalog."""

    loaded_skill_tools: NotRequired[
        Annotated[frozenset[str], PrivateStateAttr, operator.or_]
    ]
    active_skills: NotRequired[
        Annotated[dict[str, "_ActiveSkill"], PrivateStateAttr, operator.or_]
    ]
    historical_gated_tools: NotRequired[
        Annotated[frozenset[str], PrivateStateAttr, operator.or_]
    ]


class _LoadArtifact(TypedDict):
    schema: str
    requested_name: str
    winner_fingerprint: str
    result_sha256: str


class _ActiveSkill(TypedDict):
    """The catalog-bound callable capability retained after a load."""

    schema_fingerprint: str | None
    tools: frozenset[str]


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _state_value(state: dict[str, Any], key: str, default):
    """Read a private update whether LangGraph has reduced it yet or not."""
    value = state.get(key, default)
    return value.value if isinstance(value, Overwrite) else value


def _source_contains(source: str, path: str) -> bool:
    return path.startswith(f"{source.rstrip('/')}/")


def _tool_name(tool_value: Any) -> str | None:
    """Return the exact provider tool name from a final request entry."""
    if isinstance(tool_value, BaseTool):
        return tool_value.name
    if isinstance(tool_value, dict):
        name = tool_value.get("name")
        if isinstance(name, str):
            return name
        function = tool_value.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return function["name"]
    name = getattr(tool_value, "name", None) or getattr(tool_value, "__name__", None)
    return name if isinstance(name, str) else None


def _openai_tool_schema(tool_value: BaseTool | dict[str, Any] | Callable) -> dict[str, Any]:
    """Return the provider contract, wrapping runtime-injected callables first.

    A normal Assist tool may be a plain function with a ``ToolRuntime``
    parameter. LangChain's direct converter treats that injected parameter as a
    JSON Schema field; its normal ``tool(...)`` wrapper correctly removes it.
    """
    try:
        return convert_to_openai_tool(tool_value)
    except Exception:
        if isinstance(tool_value, BaseTool) or isinstance(tool_value, dict):
            raise
        return convert_to_openai_tool(tool(tool_value))


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


def _make_legacy_load_skill_tool(backend, sources):
    """Build the pre-P2b loader for compositions excluded from disclosure."""

    @tool
    def load_skill(name: str) -> str:
        """Load and return the full body of the named skill.

        Use this whenever a skill description matches your task. Pass
        only the skill's short name (e.g. "org-format") — no paths.
        Returns the full body of the skill, including the rules you
        must follow before continuing with the task.
        """
        for source in reversed(sources):
            path = f"{source.rstrip('/')}/{name}/SKILL.md"
            try:
                responses = backend.download_files([path])
            except Exception:
                continue
            if not responses:
                continue
            response = responses[0]
            if response.error or response.content is None:
                continue
            try:
                return response.content.decode("utf-8")
            except UnicodeDecodeError:
                continue
        return (
            f"Skill '{name}' not found. The system prompt's '## Skills' "
            "section lists every available name; use one of those."
        )

    return load_skill


def _make_load_skill_tool(middleware: "SmallModelSkillsMiddleware"):
    """Build the exact-winner loader bound to this middleware instance."""

    @tool
    def load_skill(name: str, runtime: ToolRuntime) -> str | Command:
        """Load and return the complete named skill file.

        Use this whenever a skill description matches your task. Pass
        only the skill's short name (e.g. "org-format") — no paths.
        Returns the full SKILL.md content and the tools newly available
        after the load.
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
            return _load_failure(name, "backend error")
        if len(response.content) > MAX_SKILL_FILE_SIZE:
            return _load_failure(name, "skill file is too large")
        try:
            raw_skill_file = response.content.decode("utf-8")
        except UnicodeDecodeError:
            return _load_failure(name, "skill file is not UTF-8")
        skill_file = _sanitize(raw_skill_file)

        newly_available = middleware._disclosed_declared_tools(skill)
        already_loaded = frozenset(_state_value(
            runtime.state, "loaded_skill_tools", ()))
        declared_names = tuple(skill.get("allowed_tools", ()))
        new_names = [name for name in declared_names
                     if name in newly_available and name not in already_loaded]
        disclosure = (
            "Newly available tools: " + ", ".join(new_names) + "."
            if new_names else "No additional tools became available."
        )
        unavailable = sorted(set(declared_names) - newly_available)
        if unavailable:
            disclosure += " Unavailable declared tools ignored: " + \
                ", ".join(unavailable) + "."
        tool_contract = middleware._tool_contract(newly_available)
        if tool_contract is None:
            return _load_failure(name, "declared tool definition unavailable")
        content = f"{skill_file}\n\n{tool_contract}{disclosure}"
        fingerprint_payload = {
            "allowed_tools": list(skill.get("allowed_tools", ())),
            "skill_file_sha256": _sha256(raw_skill_file),
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
        update: dict[str, Any] = {
            "messages": [message],
            "loaded_skill_tools": newly_available,
        }
        if middleware.retains_activation:
            active = dict(_state_value(runtime.state, "active_skills", {}))
            active[name] = {
                "schema_fingerprint": middleware._schema_fingerprint(
                    newly_available),
                "tools": newly_available,
            }
            update["active_skills"] = active
        return Command(update=update)

    return load_skill


class SmallModelSkillsMiddleware(SkillsMiddleware):
    """Name-based loader with catalog-bound progressive tool disclosure."""

    state_schema = SmallModelSkillsState

    def __init__(self, *, backend, sources, bundled_sources: Iterable[str] = (),
                 gated_sources: Iterable[str] | None = None,
                 registered_tools: Iterable[str] | None = None,
                 tool_definitions: Iterable[
                     BaseTool | dict[str, Any] | Callable] = ()):
        super().__init__(backend=backend, sources=sources)
        self._bundled_sources = frozenset(bundled_sources)
        self._gated_sources = frozenset(
            self._bundled_sources if gated_sources is None else gated_sources)
        self._registered_tools = (None if registered_tools is None
                                  else frozenset(registered_tools))
        self._tool_definitions = {
            name: value for value in tool_definitions
            if (name := _tool_name(value)) is not None
        }
        if self._gated_sources:
            self.system_prompt_template = SMALL_MODEL_SKILLS_PROMPT
            self.tools = [_make_load_skill_tool(self)]
        else:
            self.system_prompt_template = LEGACY_SMALL_MODEL_SKILLS_PROMPT
            self.tools = [_make_legacy_load_skill_tool(backend, sources)]

    @property
    def retains_activation(self) -> bool:
        """Normal progressive compositions retain, legacy ones stay unchanged."""
        return bool(self._gated_sources)

    def _tool_contract(self, names: Iterable[str]) -> str | None:
        """Return a complete ToolMessage contract, or fail closed if one is absent."""
        names = tuple(sorted(names))
        if any(name not in self._tool_definitions for name in names):
            return None
        try:
            schemas = [_openai_tool_schema(self._tool_definitions[name])
                       for name in names]
        except Exception:
            return None
        if not schemas:
            return ""
        return "## Tool contracts\n\n```json\n" + json.dumps(
            schemas, ensure_ascii=False, indent=2, sort_keys=True) + "\n```\n\n"

    def _schema_fingerprint(self, names: Iterable[str]) -> str | None:
        """Identify the exact native schemas paired with an activation."""
        names = tuple(sorted(names))
        if any(name not in self._tool_definitions for name in names):
            return None
        try:
            schemas = [_openai_tool_schema(self._tool_definitions[name])
                       for name in names]
        except Exception:
            return None
        return _sha256(json.dumps(
            schemas, ensure_ascii=False, separators=(",", ":"), sort_keys=True))

    @staticmethod
    def _compact_schema(
            tool_value: BaseTool | dict[str, Any] | Callable) -> dict[str, Any]:
        """Keep the native callable shape while moving explanatory prose to history."""
        schema = json.loads(json.dumps(_openai_tool_schema(tool_value)))

        def strip(node):
            if isinstance(node, dict):
                for key in ("description", "title", "examples", "default"):
                    node.pop(key, None)
                for value in node.values():
                    strip(value)
            elif isinstance(node, list):
                for value in node:
                    strip(value)

        strip(schema)
        return schema

    def _format_skills_list(self, skills):
        """Render only name and description; declarations remain undisclosed."""
        if not skills:
            return "(No skills available.)"
        return "\n".join(
            f"- **{skill['name']}**: {skill['description']}" for skill in skills
        )

    def _is_gated(self, skill: dict[str, Any]) -> bool:
        path = skill.get("path")
        if not isinstance(path, str):
            return False
        matches = [source for source in self.sources
                   if _source_contains(source, path)]
        return bool(matches) and max(matches, key=len) in self._gated_sources

    def _disclosed_declared_tools(self, skill: dict[str, Any]) -> frozenset[str]:
        if not self._is_gated(skill):
            return frozenset()
        declared = frozenset(skill.get("allowed_tools", ()))
        return (declared if self._registered_tools is None
                else declared & self._registered_tools)

    def _gated_tools(self, state: dict[str, Any]) -> frozenset[str]:
        gated = set(_state_value(state, "historical_gated_tools", ()))
        gated.update(self._catalog_gated_tools(state.get("skills_metadata", ())))
        return (frozenset(gated) if self._registered_tools is None
                else frozenset(gated) & self._registered_tools)

    def _catalog_gated_tools(self, skills: Iterable[dict[str, Any]]) -> frozenset[str]:
        return frozenset(
            tool for skill in skills if self._is_gated(skill)
            for tool in skill.get("allowed_tools", ()))

    def _tool_is_allowed(self, state: dict[str, Any], name: str) -> bool:
        return name not in self._gated_tools(state) or name in _state_value(
            state, "loaded_skill_tools", ())

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        """Add the catalog and expose exactly the tools allowed for this request."""
        active_tools = frozenset(
            tool for activation in _state_value(
                request.state, "active_skills", {}).values()
            for tool in activation["tools"])
        retained = [
            tool_value for tool_value in request.tools
            if (name := _tool_name(tool_value)) is None
            or self._tool_is_allowed(request.state, name)
        ]
        retained = [
            (self._compact_schema(tool_value)
             if _tool_name(tool_value) in active_tools else tool_value)
            for tool_value in retained
        ]
        names = [name for tool_value in retained
                 if (name := _tool_name(tool_value)) is not None]
        skills_section = self.system_prompt_template.format(
            skills_list=self._format_skills_list(
                request.state.get("skills_metadata", [])),
            tools_available=", ".join(names) if names else "(none)",
        )
        return request.override(
            tools=retained,
            system_message=deepagents_skills.append_to_system_message(
                request.system_message, skills_section),
        )

    def _catalog_from_responses(self, source: str, ls_result, responses):
        """Parse one source and return metadata, content identity, and host records."""
        source_error = None
        if isinstance(ls_result, deepagents_skills.LsResult) and ls_result.error:
            source_error = deepagents_skills._format_skills_source_error(  # noqa: SLF001
                source, ls_result.error)
            deepagents_skills.logger.warning("%s", source_error)
        skill_dirs, skill_paths = self._skill_paths(ls_result)
        if len(responses) != len(skill_paths):
            source_error = deepagents_skills._format_skills_source_error(  # noqa: SLF001
                source, "skill download response count mismatch")
            deepagents_skills.logger.warning("%s", source_error)
        metadata = []
        fingerprint_entries = []
        records = []
        for index, (directory, path) in enumerate(zip(skill_dirs, skill_paths,
                                                       strict=True)):
            response = responses[index] if index < len(responses) else None
            if response is None:
                fingerprint_entries.append({
                    "path": path,
                    "content_sha256": None,
                    "error": True,
                })
                continue
            content = response.content
            fingerprint_entries.append({
                "path": path,
                "content_sha256": (_bytes_sha256(content)
                                   if content is not None else None),
                "error": bool(response.error),
            })
            skill = deepagents_skills._skill_metadata_from_response(  # noqa: SLF001
                response, directory, path)
            if skill is not None:
                metadata.append(skill)
                try:
                    raw = response.content.decode("utf-8")
                except UnicodeDecodeError:
                    # The metadata helper has already rejected this response;
                    # keep the defensive branch local to this one-read seam.
                    continue
                records.append({"source": source, "skill": skill, "raw": raw})
        return metadata, source_error, fingerprint_entries, records

    @staticmethod
    def _skill_paths(ls_result):
        items = (ls_result.entries if isinstance(ls_result, deepagents_skills.LsResult)
                 else ls_result)
        directories = [item["path"] for item in items or [] if item.get("is_dir")]
        paths = [
            str(deepagents_skills.PurePosixPath(
                deepagents_skills.to_posix_path(directory)) / "SKILL.md")
            for directory in directories
        ]
        return directories, paths

    @staticmethod
    def _snapshot_result(source_results):
        all_skills = {}
        errors = []
        sources = []
        for source, metadata, error, entries in source_results:
            if error is not None:
                errors.append(error)
            for skill in metadata:
                all_skills[skill["name"]] = skill
            sources.append({
                "source": source,
                "skills": entries,
                "unavailable": error is not None,
            })
        payload = json.dumps(
            {"schema": "assist.skill-catalog.v1", "sources": sources},
            ensure_ascii=False, separators=(",", ":"), sort_keys=True,
        )
        return list(all_skills.values()), errors, _sha256(payload)

    def _catalog_source_results(self, backend):
        """Read each mounted source once for both Deep state and Pi catalog use."""
        source_results = []
        for source in self.sources:
            ls_result = backend.ls(source)
            _, paths = self._skill_paths(ls_result)
            metadata, error, entries, records = self._catalog_from_responses(
                source, ls_result,
                backend.download_files(paths) if paths else [])
            source_results.append((source, metadata, error, entries, records))
        return source_results

    def _catalog_snapshot(self, backend):
        """Discover mounted skills once and fingerprint the exact read bytes."""
        source_results = self._catalog_source_results(backend)
        return self._snapshot_result([result[:4] for result in source_results])

    def catalog_snapshot_records(self, backend):
        """Return host-only raw records from the same one-read snapshot.

        This is intentionally not graph state: Deep persists only metadata and
        the fingerprint, while another trusted host consumer may select a
        record without re-downloading the file it just described.
        """
        source_results = self._catalog_source_results(backend)
        skills, errors, fingerprint = self._snapshot_result(
            [result[:4] for result in source_results])
        return skills, errors, fingerprint, tuple(
            record for _source, _metadata, _error, _entries, records in source_results
            for record in records)

    async def _acatalog_snapshot(self, backend):
        """Async counterpart to :meth:`_catalog_snapshot`."""
        source_results = []
        for source in self.sources:
            ls_result = await backend.als(source)
            _, paths = self._skill_paths(ls_result)
            metadata, error, entries, _records = self._catalog_from_responses(
                source, ls_result,
                await backend.adownload_files(paths) if paths else [])
            source_results.append((source, metadata, error, entries))
        return self._snapshot_result(source_results)

    def _activation_is_current(self, name: object, activation: object,
                               skills: Iterable[dict[str, Any]]) -> bool:
        """Accept only an activation that exactly matches this catalog.

        Private checkpoint state is durable input, not an authority grant. A
        retained tool remains callable only when its named winning skill and
        its exact declared tool set still exist in the current catalog.
        """
        if not isinstance(name, str) or not isinstance(activation, dict):
            return False
        winner = next((skill for skill in skills if skill.get("name") == name), None)
        if winner is None:
            return False
        expected_tools = self._disclosed_declared_tools(winner)
        actual_tools = activation.get("tools")
        expected_fingerprint = self._schema_fingerprint(expected_tools)
        return (
            isinstance(actual_tools, frozenset)
            and actual_tools == expected_tools
            and expected_fingerprint is not None
            and activation.get("schema_fingerprint") == expected_fingerprint
        )

    def _activation_update(self, state, skills, fingerprint):
        """Keep an activation only while the exact catalog snapshot is unchanged."""
        catalog_changed = state.get("skills_catalog_fingerprint") != fingerprint
        persisted_active = _state_value(state, "active_skills", {})
        active = dict(persisted_active) if isinstance(persisted_active, dict) else {}
        invalid_activation = (
            not isinstance(persisted_active, dict)
            or any(not self._activation_is_current(name, activation, skills)
                   for name, activation in active.items())
        )
        if catalog_changed or invalid_activation:
            active_update: dict[str, _ActiveSkill] | Overwrite = Overwrite({})
            active = {}
        else:
            active_update = active
        historical_gated = frozenset(_state_value(
            state, "historical_gated_tools", ())) | \
            self._catalog_gated_tools(skills)
        active_tools = frozenset(
            tool for activation in active.values() for tool in activation["tools"])
        return {
            "active_skills": active_update,
            "historical_gated_tools": historical_gated,
            "loaded_skill_tools": Overwrite(active_tools),
        }

    def before_agent(self, state, runtime, config):
        backend = self._get_backend(state, runtime, config)
        if "skills_metadata" not in state:
            # Preserve Deep Agents' initial discovery hook and its census
            # provenance.  The snapshot immediately below is authoritative so
            # its metadata and fingerprint always come from the same reads.
            super().before_agent(state, runtime, config)
            skills, errors, fingerprint = self._catalog_snapshot(backend)
            update = {
                "skills_metadata": skills,
                "skills_load_errors": errors,
                "skills_catalog_fingerprint": fingerprint,
            }
        else:
            skills, errors, fingerprint = self._catalog_snapshot(backend)
            if state.get("skills_catalog_fingerprint") == fingerprint:
                update = {}
            else:
                update = {
                    "skills_metadata": skills,
                    "skills_load_errors": errors,
                    "skills_catalog_fingerprint": fingerprint,
                }
        if not self.retains_activation:
            return {**update, "loaded_skill_tools": Overwrite(frozenset())}
        return {**update, **self._activation_update(state, skills, fingerprint)}

    async def abefore_agent(self, state, runtime, config):
        backend = self._get_backend(state, runtime, config)
        if "skills_metadata" not in state:
            # Keep the async discovery hook aligned with the synchronous path;
            # use one snapshot for the state actually persisted below.
            await super().abefore_agent(state, runtime, config)
            skills, errors, fingerprint = await self._acatalog_snapshot(backend)
            update = {
                "skills_metadata": skills,
                "skills_load_errors": errors,
                "skills_catalog_fingerprint": fingerprint,
            }
        else:
            skills, errors, fingerprint = await self._acatalog_snapshot(backend)
            if state.get("skills_catalog_fingerprint") == fingerprint:
                update = {}
            else:
                update = {
                    "skills_metadata": skills,
                    "skills_load_errors": errors,
                    "skills_catalog_fingerprint": fingerprint,
                }
        if not self.retains_activation:
            return {**update, "loaded_skill_tools": Overwrite(frozenset())}
        return {**update, **self._activation_update(state, skills, fingerprint)}

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

    def _cancelled_sibling_message(self, tool_call: dict[str, Any]) -> ToolMessage:
        return ToolMessage(
            content=(
                f"Tool '{tool_call['name']}' was not run because the response also "
                "contained an unavailable tool. Retry this call by itself."
            ),
            name=tool_call["name"],
            status="error",
            tool_call_id=tool_call["id"],
        )

    @hook_config(can_jump_to=["model"])
    def after_model(self, state, runtime):
        """Reject hidden hallucinated calls before HITL sees outward effects."""
        last_ai = next(
            (message for message in reversed(state.get("messages", ()))
             if isinstance(message, AIMessage)),
            None,
        )
        if last_ai is None or not last_ai.tool_calls:
            return None
        hidden = [tool_call for tool_call in last_ai.tool_calls
                  if not self._tool_is_allowed(state, tool_call["name"])]
        if not hidden:
            return None
        hidden_ids = {tool_call["id"] for tool_call in hidden}
        results = [
            (self._hidden_call_message(tool_call)
             if tool_call["id"] in hidden_ids
             else self._cancelled_sibling_message(tool_call))
            for tool_call in last_ai.tool_calls
        ]
        # ``last_ai`` is already in the graph state.  Returning it again can
        # duplicate an ID-less adapter message when ``add_messages`` merges the
        # update; only the paired results are new state.
        return {"messages": results, "jump_to": "model"}

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
