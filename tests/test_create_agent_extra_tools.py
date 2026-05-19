"""Tests for the `extra_tools` parameter on `create_agent` and the
`extra_tools` / `extra_config` parameters on `Thread.__init__`.

Embedders (notably emacsos-server) inject per-request tools the agent
can call back into the embedder for — eg. emacsos-server registers an
`eval_elisp` tool that drives `emacsclient` against the phone.  The
contract:

  1. `extra_tools` reaches `create_deep_agent(tools=...)` so the bound
     tools end up on the model.
  2. `Thread.__init__(extra_tools=...)` forwards to `create_agent`.
  3. `Thread.__init__(extra_config=...)` deep-merges into
     `self.runconfig` so the embedder's `configurable` keys land
     alongside the built-in `thread_id` without overwriting it.
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

    def test_extra_configurable_collision_embedder_wins(self):
        """If the embedder explicitly sets `thread_id`, the embedder
        intends to override.  Document this behavior so the next
        reader knows."""
        t = self._build(extra_config={
            "configurable": {"thread_id": "embedder-set"}
        })
        assert t.runconfig["configurable"]["thread_id"] == "embedder-set"

    def test_extra_top_level_keys_override_built_in(self):
        """Top-level (non-configurable) keys are overridden wholesale."""
        t = self._build(extra_config={"max_concurrency": 99})
        assert t.runconfig["max_concurrency"] == 99
        # Built-in `configurable` survives untouched.
        assert "thread_id" in t.runconfig["configurable"]

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
