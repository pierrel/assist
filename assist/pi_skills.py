"""Host-owned Pi skill catalog and progressive-capability authority."""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Iterable

from assist.middleware.skills_middleware import SmallModelSkillsMiddleware
from assist.middleware.output_sanitization import _sanitize


_MAX_CATALOG_BYTES = 128 * 1024
_MAX_SKILL_BYTES = 96 * 1024
_PORTABLE = frozenset({"edit-files", "org-format", "pdf", "regexp", "render"})
_DECLARED_TOOLS = {"render": ("map_data",)}
_FIXED_PROVIDER_TOOLS = frozenset({"read", "write", "edit", "bash", "load_skill"})
_RIDER = """\n\n## Pi tool vocabulary\n\nIn this Pi worker, use `read` for `read_file`, `write` for `write_file`,\nand `bash` for `execute`, `ls`, `glob`, or `grep`. For an `edit_file`\ninstruction, call `edit` with `path` and `edits: [{oldText, newText}]`; do not\nuse `old_string` or `new_string`. For `execute(...)`, call `bash` with its\n`command` and `cwd: "/workspace"`. These are the only workspace tools\navailable.\n"""


class PiSkillError(ValueError):
    """A Pi catalog or its capability transition is unsafe or inconsistent."""


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PiSkill:
    """One already-read, already-sanitized winning skill visible to Pi."""

    name: str
    description: str
    body: str
    body_sha256: str
    declared_tools: tuple[str, ...]

    def manifest(self) -> dict[str, object]:
        """Return the worker-safe trigger data without source or body details."""
        return {"name": self.name, "description": self.description,
                "declaredTools": list(self.declared_tools)}


@dataclass(frozen=True)
class PiSkillCatalog:
    """Immutable per-turn winners, produced from one mounted-source read."""

    skills: tuple[PiSkill, ...]

    def get(self, name: str) -> PiSkill | None:
        return next((skill for skill in self.skills if skill.name == name), None)

    def manifest(self) -> list[dict[str, object]]:
        return [skill.manifest() for skill in self.skills]

    def prompt_section(self) -> str:
        """Render only bounded triggers; complete bodies stay host-side."""
        listing = "\n".join(f"- **{skill.name}**: {skill.description}" for skill in self.skills)
        return (
            "\n\n## Skills\n\nAvailable skills:\n"
            f"{listing or '(No skills available.)'}\n\n"
            "Before a matching action, call `load_skill(name=...)` as the only "
            "tool call in that response. Wait for its result before continuing."
        )


def empty_pi_skill_catalog() -> PiSkillCatalog:
    """Return the bounded catalog used only by non-filesystem runtime doubles."""
    return PiSkillCatalog(())


def build_pi_skill_catalog(backend: object, sources: Iterable[str], *,
                           trusted_sources: Iterable[str] | None = None) -> PiSkillCatalog:
    """Read each source once, apply Deep's rightmost-winner rule, then profile it."""
    winners: dict[str, tuple[str, PiSkill]] = {}
    source_list = tuple(sources)
    trusted = None if trusted_sources is None else frozenset(trusted_sources)
    try:
        _skills, errors, _fingerprint, records = SmallModelSkillsMiddleware(
            backend=backend, sources=source_list, gated_sources=()
        ).catalog_snapshot_records(backend)
    except Exception as error:
        raise PiSkillError("Pi skill catalog source is unavailable") from error
    if errors:
        raise PiSkillError("Pi skill catalog source is unavailable")
    for record in records:
        source = record["source"]
        metadata = record["skill"]
        raw = record["raw"]
        if not isinstance(source, str) or not isinstance(metadata, dict) or not isinstance(raw, str):
            raise PiSkillError("Pi skill catalog is invalid")
        # A final untrusted winner shadows a trusted one, but Pi must neither
        # validate nor expose its unrelated declared capability.
        name = metadata.get("name")
        if not isinstance(name, str):
            raise PiSkillError("Pi skill metadata is invalid")
        if name not in _PORTABLE:
            winners.pop(name, None)
            continue
        if trusted is not None and source not in trusted:
            winners[name] = (source, PiSkill(name, "", "", "", ()))
            continue
        if len(raw.encode("utf-8")) > _MAX_SKILL_BYTES:
            raise PiSkillError("Pi skill file exceeds its Pi bound")
        description = metadata.get("description")
        if not isinstance(description, str):
            raise PiSkillError("Pi skill metadata is invalid")
        declared = tuple(metadata.get("allowed_tools", ()))
        expected = _DECLARED_TOOLS.get(name, ())
        if declared != expected:
            # A changed trusted skill cannot silently acquire a Pi capability.
            raise PiSkillError("Pi skill declares an unsupported tool")
        body = _sanitize(raw) + _RIDER
        if len(body.encode("utf-8")) > _MAX_SKILL_BYTES:
            raise PiSkillError("Pi skill body exceeds its Pi bound")
        winners[name] = (source, PiSkill(name, description, body, _sha(body), expected))
    skills = tuple(sorted((skill for source, skill in winners.values()
                           if trusted is None or source in trusted), key=lambda skill: skill.name))
    payload = json.dumps([{"name": skill.name, "description": skill.description,
                           "body_sha256": skill.body_sha256,
                           "declared_tools": skill.declared_tools}
                          for skill in skills], separators=(",", ":"), sort_keys=True)
    if len(payload.encode("utf-8")) > _MAX_CATALOG_BYTES:
        raise PiSkillError("Pi skill catalog exceeds its bound")
    return PiSkillCatalog(skills)


@dataclass(frozen=True)
class _ObservedLoader:
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class _PendingLoader:
    observed: _ObservedLoader
    result: str
    result_sha256: str
    tools: tuple[str, ...]


class PiSkillAuthority:
    """Fail-closed host authority shared by the Pi broker and provider relay."""

    def __init__(self, catalog: PiSkillCatalog) -> None:
        self._catalog = catalog
        self._lock = threading.Lock()
        self._active_tools: frozenset[str] = frozenset()
        self._observed: _ObservedLoader | None = None
        self._pending: _PendingLoader | None = None

    @property
    def active_tools(self) -> frozenset[str]:
        with self._lock:
            return self._active_tools

    @staticmethod
    def _canonical_args(value: object) -> str | None:
        if not isinstance(value, dict) or set(value) != {"name"} or not isinstance(value["name"], str):
            return None
        return json.dumps(value, separators=(",", ":"), sort_keys=True)

    def observe_loader(self, call_id: object, name: object, arguments: object) -> None:
        """Record the sole provider loader call; it still grants no tool."""
        canonical = self._canonical_args(arguments)
        if (not isinstance(call_id, str) or not call_id
                or name != "load_skill" or canonical is None
                or self._catalog.get(json.loads(canonical)["name"]) is None):
            self.clear_loader()
            return
        with self._lock:
            self._observed = _ObservedLoader(call_id, "load_skill", canonical)
            self._pending = None

    def load_skill(self, call_id: object, name: object, arguments: object) -> str:
        """Issue a host result only for the immediately observed provider call."""
        canonical = self._canonical_args(arguments)
        if not isinstance(call_id, str) or not isinstance(name, str) or canonical is None:
            raise PiSkillError("Pi skill load is invalid")
        with self._lock:
            observed = self._observed
            if (observed is None or observed.call_id != call_id or observed.name != name
                    or observed.arguments != canonical):
                self._observed = None
                self._pending = None
                raise PiSkillError("Pi skill load is not provider-observed")
            requested = json.loads(canonical)["name"]
            skill = self._catalog.get(requested)
            if skill is None:
                self._observed = None
                raise PiSkillError("Pi skill is unavailable")
            disclosure = ("Newly available tools: " + ", ".join(skill.declared_tools) + "."
                          if skill.declared_tools else "No additional tools became available.")
            result = f"{skill.body}\n\n{disclosure}"
            self._pending = _PendingLoader(observed, result, _sha(result), skill.declared_tools)
            self._observed = None
            return result

    def _continue_request(self, payload: object, *, activate: bool) -> bool:
        """Validate the next continuation and activate it only after relay admission."""
        with self._lock:
            pending = self._pending
            if not isinstance(payload, dict):
                self._pending = None
                return False
            tools = payload.get("tools")
            if not isinstance(tools, list):
                self._pending = None
                return False
            names = _provider_tool_names(tools)
            if names is None:
                self._pending = None
                return False
            expected = set(_FIXED_PROVIDER_TOOLS) | set(self._active_tools)
            if pending is not None:
                expected.update(pending.tools)
            if names != expected:
                self._pending = None
                return False
            if pending is None:
                return True
            messages = payload.get("messages")
            if not isinstance(messages, list):
                self._pending = None
                return False
            assistant = next((item for item in reversed(messages)
                              if isinstance(item, dict) and item.get("role") == "assistant"), None)
            result = next((item for item in reversed(messages)
                           if isinstance(item, dict) and item.get("role") == "tool"), None)
            if not isinstance(assistant, dict) or not isinstance(result, dict):
                self._pending = None
                return False
            calls = assistant.get("tool_calls")
            if (not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict)
                    or calls[0].get("id") != pending.observed.call_id):
                self._pending = None
                return False
            function = calls[0].get("function")
            if not isinstance(function, dict) or function.get("name") != "load_skill":
                self._pending = None
                return False
            try:
                args = json.loads(function.get("arguments", ""))
            except (TypeError, json.JSONDecodeError):
                self._pending = None
                return False
            if self._canonical_args(args) != pending.observed.arguments:
                self._pending = None
                return False
            content = result.get("content")
            if (result.get("tool_call_id") != pending.observed.call_id
                    or not isinstance(content, str) or _sha(content) != pending.result_sha256):
                self._pending = None
                return False
            if activate:
                self._pending = None
                self._active_tools = frozenset(set(self._active_tools) | set(pending.tools))
            return True

    def can_continue_request(self, payload: object) -> bool:
        """Validate a continuation before the relay forwards it without granting a tool."""
        return self._continue_request(payload, activate=False)

    def continue_request(self, payload: object) -> bool:
        """Promote only the exact forwarded assistant/tool continuation, otherwise clear."""
        return self._continue_request(payload, activate=True)

    def require(self, name: str) -> None:
        """Reject a broker operation that has not crossed the relay boundary."""
        with self._lock:
            if name not in self._active_tools:
                raise PiSkillError("Pi skill capability is unavailable")

    def clear_loader(self) -> None:
        with self._lock:
            self._observed = None
            self._pending = None


def _provider_tool_names(values: list[object]) -> frozenset[str] | None:
    """Require one well-formed function entry per expected provider tool name."""
    names: set[str] = set()
    for value in values:
        name = _provider_tool_name(value)
        if name is None or name in names:
            return None
        names.add(name)
    return frozenset(names)


def _provider_tool_name(value: object) -> str | None:
    if not isinstance(value, dict) or value.get("type") != "function":
        return None
    function = value.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    return None
