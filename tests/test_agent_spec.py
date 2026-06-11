"""Unit tests for the AgentSpec embedder contract (assist/spec.py).

These pin the contract itself: defaults, immutability, normalization,
and validation.  The wiring of a spec through ``create_agent`` /
``Thread`` is pinned separately (spec-equivalence tests alongside the
legacy forwarding tests).
"""

import dataclasses
from types import MappingProxyType

import pytest

from assist.spec import AgentSpec


def _tool(name="t"):
    def fn():
        return name
    fn.__name__ = name
    return fn


class TestDefaults:
    def test_empty_spec_is_todays_defaults(self):
        spec = AgentSpec()
        assert spec.tools == ()
        assert len(spec.skill_sources) == 0
        assert spec.default_backend is None

    def test_two_default_specs_are_equal(self):
        assert AgentSpec() == AgentSpec()


class TestFrozen:
    def test_field_assignment_raises(self):
        spec = AgentSpec()
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.tools = (_tool(),)

    def test_skill_sources_is_read_only(self):
        spec = AgentSpec(skill_sources={"/x/": object()})
        assert isinstance(spec.skill_sources, MappingProxyType)
        with pytest.raises(TypeError):
            spec.skill_sources["/y/"] = object()


class TestNormalization:
    def test_tools_list_normalized_to_tuple(self):
        t = _tool()
        spec = AgentSpec(tools=[t])
        assert spec.tools == (t,)
        assert isinstance(spec.tools, tuple)

    def test_skill_sources_copied_from_caller_dict(self):
        backend = object()
        sources = {"/x/": backend}
        spec = AgentSpec(skill_sources=sources)
        sources["/y/"] = object()  # embedder mutates its own dict later
        assert dict(spec.skill_sources) == {"/x/": backend}


class TestValidation:
    def test_tools_string_rejected(self):
        with pytest.raises(TypeError, match="tools must be a sequence"):
            AgentSpec(tools="read_url")

    def test_tools_non_sequence_rejected(self):
        with pytest.raises(TypeError, match="tools must be a sequence"):
            AgentSpec(tools=42)

    def test_skill_sources_non_mapping_rejected(self):
        with pytest.raises(TypeError, match="skill_sources must be a mapping"):
            AgentSpec(skill_sources=[("/x/", object())])
