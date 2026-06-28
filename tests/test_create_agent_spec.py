"""Wiring tests for the AgentSpec embedder contract — the canonical
surface pins (docs/2026-06-11-embedder-contract.org): spec fields
reaching `create_deep_agent`, checkpointer/sandbox_backend forwarding,
and `Thread`-level `spec=` / `configurable=` wiring.
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


class TestSpecWiring(_CreateAgentHarness):
    """The spec's fields reach create_deep_agent."""

    def test_default_spec_has_only_builtin_travel_tool(self):
        # `travel` is a built-in always on the main agent; default spec adds nothing else.
        from assist.tools import travel
        assert self._build()["tools"] == [travel]
        assert self._build(spec=AgentSpec())["tools"] == [travel]

    def test_spec_tools_reach_create_deep_agent(self):
        from assist.tools import travel
        kwargs = self._build(spec=AgentSpec(tools=(_tool_a, _tool_b)))
        assert kwargs["tools"] == [_tool_a, _tool_b, travel]

    def test_spec_skill_sources_reach_middleware(self):
        from assist.middleware.skills_middleware import SmallModelSkillsMiddleware
        backend = MagicMock()
        kwargs = self._build(
            spec=AgentSpec(skill_sources={"/client-skills/": backend}))
        mw = next(m for m in kwargs["middleware"]
                  if isinstance(m, SmallModelSkillsMiddleware))
        assert "/client-skills/" in mw.sources

    def test_loop_detection_present_by_default(self):
        """The hardened stack ships with the plain A/B loop detector and
        no per-tool exploration knob (the rollback contract; moved here
        from the deleted legacy-kwarg test file)."""
        from assist.middleware.loop_detection import LoopDetectionMiddleware
        mws = self._build()["middleware"]
        mw = next(m for m in mws if isinstance(m, LoopDetectionMiddleware))
        assert not hasattr(mw, "exploration_tools")

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

    def test_embedder_mutation_after_construction_is_isolated(self):
        shared = {"phone_context": "original"}
        t, _ = self._build(configurable=shared)
        shared["phone_context"] = "MUTATED"
        shared["new_key"] = "added"
        assert t.runconfig["configurable"]["phone_context"] == "original"
        assert "new_key" not in t.runconfig["configurable"]


class TestSpecTypeValidation(_CreateAgentHarness):
    def test_non_spec_raises_clear_typeerror(self):
        with pytest.raises(TypeError, match="spec must be an AgentSpec, got dict"):
            self._build(spec={"tools": ()})
