"""Pi's catalog is host-owned and its disclosed capability is fail-closed."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from assist.pi_skills import (PiSkill, PiSkillAuthority, PiSkillCatalog,
                              PiSkillError, build_pi_skill_catalog)


class _Backend:
    def __init__(self, files: dict[str, str]) -> None:
        self.files = files

    def ls(self, source: str):
        return [{"path": f"{source.rstrip('/')}/{name}", "is_dir": True}
                for name in self.files]

    def download_files(self, paths: list[str]):
        return [SimpleNamespace(content=self.files[path.rsplit("/", 2)[-2]].encode(), error=None)
                for path in paths]


def _catalog() -> PiSkillCatalog:
    body = "render rules"
    skill = PiSkill("render", "show a file", body, hashlib.sha256(body.encode()).hexdigest(), ("map_data",))
    return PiSkillCatalog((skill,))


def test_catalog_applies_rightmost_winner_then_filters_untrusted_source() -> None:
    bundled = _Backend({"render": "---\nname: render\ndescription: bundled\nallowed-tools: map_data\n---\nbody"})
    domain = _Backend({"render": "---\nname: render\ndescription: domain\nallowed-tools: send_email\n---\nbody"})
    class Composite:
        def ls(self, source: str):
            return (bundled if source == "/skills/" else domain).ls(source)
        def download_files(self, paths: list[str]):
            return (bundled if paths[0].startswith("/skills/") else domain).download_files(paths)

    catalog = build_pi_skill_catalog(Composite(), ("/skills/", "/.claude/skills/"),
                                     trusted_sources=("/skills/",))

    # The untrusted later winner removes the bundled winner rather than quietly
    # falling back to it; Pi never advertises a source it did not admit.
    assert catalog.skills == ()


def test_authority_requires_observed_load_and_exact_following_continuation() -> None:
    authority = PiSkillAuthority(_catalog())
    with pytest.raises(PiSkillError, match="not provider-observed"):
        authority.load_skill("call-1", "load_skill", {"name": "render"})

    authority.observe_loader("call-1", "load_skill", {"name": "render"})
    result = authority.load_skill("call-1", "load_skill", {"name": "render"})
    assert authority.active_tools == frozenset()
    assert not authority.continue_request({"messages": [], "tools": []})
    assert authority.active_tools == frozenset()

    authority.observe_loader("call-2", "load_skill", {"name": "render"})
    result = authority.load_skill("call-2", "load_skill", {"name": "render"})
    assert authority.continue_request({
        "messages": [
            {"role": "assistant", "tool_calls": [{"id": "call-2", "function": {
                "name": "load_skill", "arguments": '{"name":"render"}'}}]},
            {"role": "tool", "tool_call_id": "call-2", "content": result},
        ],
        "tools": [
            {"type": "function", "function": {"name": name}}
            for name in ("read", "write", "edit", "bash", "load_skill", "map_data")
        ],
    })
    assert authority.active_tools == frozenset({"map_data"})
    authority.require("map_data")


def test_authority_preflight_does_not_promote_a_pending_capability() -> None:
    authority = PiSkillAuthority(_catalog())
    authority.observe_loader("call-1", "load_skill", {"name": "render"})
    result = authority.load_skill("call-1", "load_skill", {"name": "render"})
    continuation = {
        "messages": [
            {"role": "assistant", "tool_calls": [{"id": "call-1", "function": {
                "name": "load_skill", "arguments": '{"name":"render"}'}}]},
            {"role": "tool", "tool_call_id": "call-1", "content": result},
        ],
        "tools": [
            {"type": "function", "function": {"name": name}}
            for name in ("read", "write", "edit", "bash", "load_skill", "map_data")
        ],
    }

    assert authority.can_continue_request(continuation)
    assert authority.active_tools == frozenset()
    assert authority.continue_request(continuation)
    assert authority.active_tools == frozenset({"map_data"})


def test_authority_returns_only_host_promoted_skill_records() -> None:
    authority = PiSkillAuthority(_catalog(), load_run_id="run-1")
    authority.observe_loader("call-1", "load_skill", {"name": "render"})
    result = authority.load_skill("call-1", "load_skill", {"name": "render"})
    continuation = {
        "messages": [
            {"role": "assistant", "tool_calls": [{"id": "call-1", "function": {
                "name": "load_skill", "arguments": '{"name":"render"}'}}]},
            {"role": "tool", "tool_call_id": "call-1", "content": result},
        ],
        "tools": [
            {"type": "function", "function": {"name": name}}
            for name in ("read", "write", "edit", "bash", "load_skill", "map_data")
        ],
    }

    assert authority.continue_request(continuation)
    retained = authority.committed_skills
    assert [skill.name for skill in retained] == ["render"]
    assert retained[0].run_id == "run-1"

    next_turn = PiSkillAuthority(_catalog(), retained=retained)
    assert next_turn.active_tools == frozenset({"map_data"})
    next_turn.require("map_data")


def test_authority_non_object_preflight_clears_a_pending_capability() -> None:
    authority = PiSkillAuthority(_catalog())
    authority.observe_loader("call-1", "load_skill", {"name": "render"})
    authority.load_skill("call-1", "load_skill", {"name": "render"})

    assert not authority.can_continue_request([])
    assert authority.active_tools == frozenset()


def test_authority_rejects_missing_or_extra_provider_schemas() -> None:
    authority = PiSkillAuthority(_catalog())
    fixed = [{"type": "function", "function": {"name": name}}
             for name in ("read", "write", "edit", "bash", "load_skill")]

    assert authority.continue_request({"messages": [], "tools": fixed})
    assert not authority.continue_request({"messages": [], "tools": fixed[:-1]})
    assert not authority.continue_request({
        "messages": [], "tools": fixed + [{"type": "function", "function": {"name": "send_email"}}],
    })
    assert not authority.continue_request({"messages": [], "tools": fixed + [fixed[0]]})
    assert not authority.continue_request({"messages": [], "tools": fixed + [{"type": "function"}]})
    assert not authority.continue_request({
        "messages": [], "tools": fixed + [{"name": "send_email"}],
    })


def test_catalog_rejects_a_portable_skill_that_acquires_an_unreviewed_tool() -> None:
    backend = _Backend({"render": "---\nname: render\ndescription: no\nallowed-tools: send_email\n---\nbody"})
    with pytest.raises(PiSkillError, match="unsupported tool"):
        build_pi_skill_catalog(backend, ("/skills/",))


def test_catalog_adds_the_actual_pi_edit_and_bash_call_shapes() -> None:
    backend = _Backend({"edit-files": "---\nname: edit-files\ndescription: edit\n---\nbody"})
    catalog = build_pi_skill_catalog(backend, ("/skills/",))

    body = catalog.get("edit-files").body  # type: ignore[union-attr]
    assert body.startswith("## Pi tool vocabulary")
    assert body.endswith("available.\n")
