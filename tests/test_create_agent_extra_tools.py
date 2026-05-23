"""Tests for the `extra_tools` parameter on `create_agent` and the
`extra_tools` / `extra_config` parameters on `Thread.__init__`.

Embedders (notably emacsos-server) inject per-request tools the agent
can call back into the embedder for — eg. emacsos-server registers an
`eval_elisp` tool that drives `emacsclient` against the phone.  The
contract:

  1. `extra_tools` reaches `create_deep_agent(tools=...)` so the bound
     tools end up on the model.
  2. `Thread.__init__(extra_tools=...)` forwards to `create_agent`.
  3. `Thread.__init__(extra_config=...)` two-level-merges into
     `self.runconfig` — the inner `configurable` dict gets a shallow
     `.update()` from the embedder's `configurable` (adds alongside
     built-in `thread_id`), top-level keys are overridden wholesale.
     Not a recursive deep merge — a key whose value is itself a dict
     replaces any existing dict at that key.
  4. Defaults preserve the pre-2026-05-19 behavior (no tools added, no
     extra configurable keys).
"""

import tempfile
from unittest.mock import patch, MagicMock


class TestCreateAgentExtraTools:
    """`create_agent` is heavy (sub-agents, model probes); patch
    `create_deep_agent` and verify only the wiring."""

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
                model = MagicMock()
                create_agent(
                    model, wd, checkpointer=InMemorySaver(),
                    sandbox_backend=None, **kwargs,
                )
                return fake.call_args.kwargs

    def test_default_extra_tools_is_empty_list(self):
        kwargs = self._build()
        # Default `None` collapses to an empty list passed through.
        assert kwargs["tools"] == []

    def test_extra_tools_forwarded_to_create_deep_agent(self):
        def fake_tool_a(x: str) -> str:
            return x

        def fake_tool_b(y: int) -> int:
            return y

        kwargs = self._build(extra_tools=[fake_tool_a, fake_tool_b])
        # The list reaches deepagents intact.
        assert kwargs["tools"] == [fake_tool_a, fake_tool_b]

    def test_extra_tools_accepts_sequence_not_just_list(self):
        """Mirror deepagents' `Sequence[...]` signature: tuples work too."""
        def t1(x: str) -> str: return x
        def t2(x: str) -> str: return x

        kwargs = self._build(extra_tools=(t1, t2))
        # Conversion to list at the boundary.
        assert isinstance(kwargs["tools"], list)
        assert kwargs["tools"] == [t1, t2]


class TestCreateAgentLoopExplorationTools:
    """`loop_exploration_tools` reaches the MAIN agent's
    `LoopDetectionMiddleware(exploration_tools=...)`; default leaves the
    dev/code agent unchanged (empty set)."""

    def _main_loop_mw(self, **kwargs):
        from assist.agent import create_agent
        from assist.middleware.loop_detection import LoopDetectionMiddleware
        from langgraph.checkpoint.memory import InMemorySaver

        with patch("assist.agent.create_deep_agent") as fake, \
             patch("assist.agent.create_context_agent") as fake_ctx, \
             patch("assist.agent.create_research_agent") as fake_res:
            fake.return_value = MagicMock()
            fake_ctx.return_value = MagicMock()
            fake_res.return_value = MagicMock()
            with tempfile.TemporaryDirectory() as wd:
                create_agent(MagicMock(), wd, checkpointer=InMemorySaver(),
                             sandbox_backend=None, **kwargs)
            mws = fake.call_args.kwargs["middleware"]
            return next(m for m in mws if isinstance(m, LoopDetectionMiddleware))

    def test_default_exploration_tools_is_empty(self):
        # The dev/code agent (no exploration tools) is unchanged.
        assert self._main_loop_mw().exploration_tools == frozenset()

    def test_loop_exploration_tools_forwarded_to_middleware(self):
        mw = self._main_loop_mw(loop_exploration_tools=frozenset({"eval_elisp"}))
        assert mw.exploration_tools == frozenset({"eval_elisp"})


class TestThreadExtraTools:
    """`Thread.__init__` is heavy too; patch `create_agent` and verify
    the wiring through to `create_agent` + `self.runconfig`."""

    def _build(self, **kwargs):
        from assist.thread import Thread
        with patch("assist.thread.create_agent") as fake_ca, \
             patch("assist.thread.select_chat_model") as fake_model:
            fake_ca.return_value = MagicMock()
            fake_model.return_value = MagicMock()
            with tempfile.TemporaryDirectory() as wd:
                t = Thread(working_dir=wd, **kwargs)
                return t, fake_ca.call_args.kwargs

    def test_default_extra_tools_none_passed_through(self):
        _, ca_kwargs = self._build()
        assert ca_kwargs["extra_tools"] is None

    def test_extra_tools_passed_through_to_create_agent(self):
        def my_tool(x: str) -> str: return x
        _, ca_kwargs = self._build(extra_tools=[my_tool])
        assert ca_kwargs["extra_tools"] == [my_tool]

    def test_default_loop_exploration_tools_none_passed_through(self):
        _, ca_kwargs = self._build()
        assert ca_kwargs["loop_exploration_tools"] is None

    def test_loop_exploration_tools_forwarded_to_create_agent(self):
        _, ca_kwargs = self._build(loop_exploration_tools=frozenset({"eval_elisp"}))
        assert ca_kwargs["loop_exploration_tools"] == frozenset({"eval_elisp"})


class TestThreadExtraConfig:
    """`Thread.__init__(extra_config=...)` merges into `self.runconfig`.
    Built-in `configurable.thread_id` must survive a merge that doesn't
    name it; embedder keys must win over built-ins on collision."""

    def _build(self, **kwargs):
        from assist.thread import Thread
        with patch("assist.thread.create_agent") as fake_ca, \
             patch("assist.thread.select_chat_model") as fake_model:
            fake_ca.return_value = MagicMock()
            fake_model.return_value = MagicMock()
            with tempfile.TemporaryDirectory() as wd:
                return Thread(working_dir=wd, **kwargs)

    def test_default_runconfig_unchanged_when_extra_config_none(self):
        t = self._build()
        # Only built-in keys.
        assert "thread_id" in t.runconfig["configurable"]
        assert t.runconfig["max_concurrency"] == 5

    def test_extra_configurable_merges_into_configurable(self):
        t = self._build(extra_config={
            "configurable": {"phone_context": {"auth_contents": "x", "phone_host": "h"}}
        })
        # Built-in survives.
        assert "thread_id" in t.runconfig["configurable"]
        # Embedder key landed.
        assert t.runconfig["configurable"]["phone_context"] == {
            "auth_contents": "x", "phone_host": "h"
        }

    def test_extra_configurable_cannot_override_thread_id(self):
        """Constructor-owned key: `thread_id` lives both as
        `self.thread_id` (read by THREAD_QUEUE for affinity, by
        `message()` for log lines) and as
        `runconfig.configurable.thread_id`.  Letting `extra_config`
        override the runconfig copy would diverge the two; protected
        instead — pass via the `thread_id=` constructor param if you
        need a non-default id."""
        t = self._build(thread_id="ctor-set", extra_config={
            "configurable": {"thread_id": "embedder-attempt"}
        })
        # Attribute holds the ctor value.
        assert t.thread_id == "ctor-set"
        # Runconfig agrees — embedder's attempt was silently dropped.
        assert t.runconfig["configurable"]["thread_id"] == "ctor-set"

    def test_extra_top_level_max_concurrency_protected(self):
        """Same rationale as thread_id: `self.max_concurrency` is the
        public attribute; if an embedder overrode just the runconfig
        copy, callers reading `self.max_concurrency` would see a
        different value than langgraph does."""
        t = self._build(max_concurrency=7, extra_config={
            "max_concurrency": 99,
        })
        assert t.max_concurrency == 7
        assert t.runconfig["max_concurrency"] == 7

    def test_extra_top_level_non_protected_keys_pass_through(self):
        """Top-level keys other than `configurable` / `max_concurrency`
        flow through unchanged — embedder freely adds new langgraph
        config knobs."""
        t = self._build(extra_config={"recursion_limit": 42})
        assert t.runconfig["recursion_limit"] == 42

    def test_extra_config_does_not_leak_across_threads(self):
        """Two Threads built with different extra_config must not share
        state — guards against accidental dict-aliasing in the merge."""
        t1 = self._build(extra_config={
            "configurable": {"phone_context": "one"}
        })
        t2 = self._build(extra_config={
            "configurable": {"phone_context": "two"}
        })
        assert t1.runconfig["configurable"]["phone_context"] == "one"
        assert t2.runconfig["configurable"]["phone_context"] == "two"

    def test_extra_config_non_dict_raises_clear_typeerror(self):
        """Public-API validation: embedder passing a non-dict gets a
        clear TypeError naming the actual type instead of a downstream
        AttributeError on `.items()`."""
        import pytest as _pytest
        with _pytest.raises(TypeError, match="extra_config must be a dict, got str"):
            self._build(extra_config="not a dict")

    def test_extra_config_falsy_non_dict_still_validates(self):
        """`if extra_config is not None` (not `if extra_config:`) so
        falsy-but-wrong-type values (eg. `[]`) still hit the
        isinstance check instead of silently skipping validation."""
        import pytest as _pytest
        with _pytest.raises(TypeError, match="extra_config must be a dict, got list"):
            self._build(extra_config=[])
        with _pytest.raises(TypeError, match="extra_config must be a dict, got str"):
            self._build(extra_config="")

    def test_extra_config_empty_dict_is_noop(self):
        """Explicit `{}` is a harmless no-op (matches the prior
        `if extra_config:` behavior for the empty-dict case)."""
        t = self._build(extra_config={})
        # Built-in keys still present, nothing extra.
        assert "thread_id" in t.runconfig["configurable"]
        assert t.runconfig["max_concurrency"] == 5

    def test_extra_config_configurable_non_dict_raises_clear_typeerror(self):
        """Same shape for the nested `configurable` key."""
        import pytest as _pytest
        with _pytest.raises(TypeError,
                            match=r"extra_config\['configurable'\] must be a dict, got list"):
            self._build(extra_config={"configurable": ["not", "a", "dict"]})

    def test_embedder_mutating_extra_config_after_construction_is_isolated(self):
        """Defensive shallow-copy: if the embedder mutates its own
        `extra_config["configurable"]` dict AFTER constructing the
        Thread, the Thread's runconfig must not see the mutation.
        Protects against the embedder reusing one config dict across
        many Threads and mutating in place."""
        shared = {"configurable": {"phone_context": "original"}}
        t = self._build(extra_config=shared)
        assert t.runconfig["configurable"]["phone_context"] == "original"

        # Mutate the embedder's input AFTER construction.
        shared["configurable"]["phone_context"] = "MUTATED"
        shared["configurable"]["new_key"] = "added"

        # Thread's runconfig is unaffected.
        assert t.runconfig["configurable"]["phone_context"] == "original"
        assert "new_key" not in t.runconfig["configurable"]
