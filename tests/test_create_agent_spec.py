"""Wiring tests for the AgentSpec embedder contract.

Pins three things (docs/2026-06-11-embedder-contract.org):

  1. *Equivalence*: `create_agent(spec=AgentSpec(...))` produces the
     same `create_deep_agent` call as the legacy kwargs it replaces.
     These become the canonical surface pins once the legacy kwargs
     are removed.
  2. *Mutual exclusion*: spec + any legacy kwarg is a TypeError whose
     message names the replacement field (the migration doc for
     embedders).
  3. *Forwarding gaps*: checkpointer / sandbox_backend forwarding,
     previously unpinned.

`Thread`-level `spec=` / `configurable=` wiring is here too.
"""

import tempfile
from unittest.mock import patch, MagicMock

import pytest

from assist.spec import AgentSpec


def _tool_a(x: str) -> str:
    return x


def _tool_b(y: int) -> int:
    return y


class _CreateAgentHarness:
    """Patch the heavy bits of create_agent; return create_deep_agent kwargs."""

    def _build(self, **kwargs):
        from assist.agent import create_agent
        from langgraph.checkpoint.memory import InMemorySaver

        with patch("assist.agent.create_deep_agent") as fake, \
             patch("assist.agent.create_context_agent") as fake_ctx, \
             patch("assist.agent.create_research_agent") as fake_res:
            fake.return_value = MagicMock()
            fake_ctx.return_value = MagicMock()
            fake_res.return_value = MagicMock()
            with tempfile.TemporaryDirectory() as wd:
                kwargs.setdefault("checkpointer", InMemorySaver())
                create_agent(MagicMock(), wd, **kwargs)
                return fake.call_args.kwargs


class TestSpecEquivalence(_CreateAgentHarness):
    """spec path and legacy path produce identical deepagents wiring."""

    def test_default_spec_equals_no_kwargs(self):
        legacy = self._build()
        spec = self._build(spec=AgentSpec())
        assert spec["tools"] == legacy["tools"] == []

    def test_spec_tools_match_legacy_extra_tools(self):
        legacy = self._build(extra_tools=[_tool_a, _tool_b])
        spec = self._build(spec=AgentSpec(tools=(_tool_a, _tool_b)))
        assert spec["tools"] == legacy["tools"] == [_tool_a, _tool_b]

    def test_spec_skill_sources_match_legacy(self):
        from assist.middleware.skills_middleware import SmallModelSkillsMiddleware

        def _sources(kwargs):
            mw = next(m for m in kwargs["middleware"]
                      if isinstance(m, SmallModelSkillsMiddleware))
            return mw.sources

        backend = MagicMock()
        legacy = self._build(extra_skill_sources={"/client-skills/": backend})
        spec = self._build(
            spec=AgentSpec(skill_sources={"/client-skills/": backend}))
        assert _sources(spec) == _sources(legacy)
        assert "/client-skills/" in _sources(spec)


class TestSpecLegacyMutualExclusion(_CreateAgentHarness):
    def test_spec_plus_extra_tools_raises_with_replacement(self):
        with pytest.raises(TypeError, match=r"AgentSpec\.tools"):
            self._build(spec=AgentSpec(), extra_tools=[_tool_a])

    def test_spec_plus_skill_sources_raises_with_replacement(self):
        with pytest.raises(TypeError, match=r"AgentSpec\.skill_sources"):
            self._build(spec=AgentSpec(),
                        extra_skill_sources={"/x/": MagicMock()})

    def test_spec_plus_deprecated_loop_kwarg_raises_drop_hint(self):
        with pytest.raises(TypeError, match="drop the kwarg"):
            self._build(spec=AgentSpec(),
                        loop_exploration_tools=frozenset({"eval_elisp"}))

    def test_spec_default_backend_excludes_sandbox_backend(self):
        with pytest.raises(ValueError, match="not both"):
            self._build(spec=AgentSpec(default_backend=MagicMock()),
                        sandbox_backend=MagicMock())


class TestForwardingGaps(_CreateAgentHarness):
    """create_agent-level forwarding that was previously unpinned:
    checkpointer to create_deep_agent, sandbox_backend into the
    subagent factories.  (Thread-level forwarding of both is pinned in
    TestThreadSpecForwarding.)"""

    def test_checkpointer_forwarded_to_create_deep_agent(self):
        from langgraph.checkpoint.memory import InMemorySaver
        saver = InMemorySaver()
        kwargs = self._build(checkpointer=saver)
        assert kwargs["checkpointer"] is saver

    def test_sandbox_backend_forwarded_to_subagent_factories(self):
        from assist.agent import create_agent
        from langgraph.checkpoint.memory import InMemorySaver

        sandbox = MagicMock()
        sandbox.work_dir = "/workspace"
        with patch("assist.agent.create_deep_agent") as fake, \
             patch("assist.agent.create_context_agent") as fake_ctx, \
             patch("assist.agent.create_research_agent") as fake_res, \
             patch("assist.agent.create_sandbox_composite_backend"):
            fake.return_value = MagicMock()
            fake_ctx.return_value = MagicMock()
            fake_res.return_value = MagicMock()
            with tempfile.TemporaryDirectory() as wd:
                create_agent(MagicMock(), wd, checkpointer=InMemorySaver(),
                             sandbox_backend=sandbox)
        assert fake_ctx.call_args.kwargs["sandbox_backend"] is sandbox
        assert fake_res.call_args.kwargs["sandbox_backend"] is sandbox


class _ThreadHarness:
    def _build(self, **kwargs):
        from assist.thread import Thread
        with patch("assist.thread.create_agent") as fake_ca, \
             patch("assist.thread.select_assistant_model") as fake_model:
            fake_ca.return_value = MagicMock()
            fake_model.return_value = MagicMock()
            with tempfile.TemporaryDirectory() as wd:
                t = Thread(working_dir=wd, **kwargs)
                return t, fake_ca.call_args.kwargs


class TestThreadSpecForwarding(_ThreadHarness):
    def test_spec_forwarded_to_create_agent(self):
        spec = AgentSpec(tools=(_tool_a,))
        _, ca_kwargs = self._build(spec=spec)
        assert ca_kwargs["spec"] is spec

    def test_default_spec_none_forwarded(self):
        _, ca_kwargs = self._build()
        assert ca_kwargs["spec"] is None

    def test_sandbox_backend_forwarded(self):
        sandbox = MagicMock()
        _, ca_kwargs = self._build(sandbox_backend=sandbox)
        assert ca_kwargs["sandbox_backend"] is sandbox

    def test_checkpointer_forwarded(self):
        saver = MagicMock()
        _, ca_kwargs = self._build(checkpointer=saver)
        assert ca_kwargs["checkpointer"] is saver


class TestThreadConfigurable(_ThreadHarness):
    """The narrowed replacement for extra_config."""

    def test_merges_into_runconfig_configurable(self):
        t, _ = self._build(configurable={"phone_context": "ctx"})
        assert t.runconfig["configurable"]["phone_context"] == "ctx"
        assert "thread_id" in t.runconfig["configurable"]  # built-in survives

    def test_reserved_keys_raise(self):
        for key in ("thread_id", "checkpoint_ns", "checkpoint_id"):
            with pytest.raises(ValueError, match="reserved langgraph keys"):
                self._build(configurable={key: "x"})

    def test_non_mapping_raises(self):
        with pytest.raises(TypeError, match="configurable must be a mapping"):
            self._build(configurable=["not", "a", "mapping"])

    def test_mutually_exclusive_with_extra_config(self):
        with pytest.raises(TypeError, match="not both"):
            self._build(configurable={"a": 1},
                        extra_config={"configurable": {"b": 2}})

    def test_embedder_mutation_after_construction_is_isolated(self):
        shared = {"phone_context": "original"}
        t, _ = self._build(configurable=shared)
        shared["phone_context"] = "MUTATED"
        shared["new_key"] = "added"
        assert t.runconfig["configurable"]["phone_context"] == "original"
        assert "new_key" not in t.runconfig["configurable"]
