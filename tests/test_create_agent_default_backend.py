"""Tests for the `default_backend` injection seam.

An embedder (emacsos-server) supplies the composite backend's *default* —
the target for every non-routed path — so the agent operates against a
custom backend (e.g. a remote/emacs backend) instead of a FilesystemBackend
rooted at `working_dir`.  assist still wraps it with the standard
STATEFUL_PATHS -> StateBackend routing (so summarization/scratch stay
ephemeral and never hit the injected backend), and if the injected default
implements `SandboxBackendProtocol`, deepagents' `supports_execution`
enables the `execute` tool for it automatically.
"""

import tempfile
from unittest.mock import patch, MagicMock

import pytest

from assist.backends import (
    SKILLS_ROUTE,
    STATEFUL_PATHS,
    create_composite_backend,
)
from deepagents.backends import FilesystemBackend, StateBackend
from deepagents.backends.protocol import SandboxBackendProtocol
from deepagents.middleware.filesystem import supports_execution


class _FakeSandboxBackend(FilesystemBackend, SandboxBackendProtocol):
    """A SandboxBackendProtocol default: fs methods from FilesystemBackend
    plus `execute` — the shape of emacsos's planned EmacsBackend."""

    def execute(self, command, timeout=None):
        return {"command": command, "exit_code": 0, "output": ""}


def _fs():
    return FilesystemBackend(root_dir=tempfile.mkdtemp(), virtual_mode=True)


class TestCompositeDefaultBackend:
    def test_default_backend_becomes_composite_default(self):
        inj = _fs()
        cb = create_composite_backend(stateful_paths=STATEFUL_PATHS,
                                      default_backend=inj)
        assert cb.default is inj
        # STATEFUL_PATHS still route to StateBackend — internal scratch
        # (question.txt, large_tool_results/, conversation_history/) must NOT
        # land on the injected default.
        for p in STATEFUL_PATHS:
            assert p in cb.routes
            assert isinstance(cb.routes[p], StateBackend)
        assert SKILLS_ROUTE in cb.routes

    def test_default_backend_ignores_fs_root(self):
        inj = _fs()
        cb = create_composite_backend(fs_root="/should/be/ignored",
                                      default_backend=inj)
        assert cb.default is inj


class TestCreateAgentDefaultBackend:
    """`create_agent` is heavy; patch `create_deep_agent` and inspect the
    `backend` it was handed (mirrors test_create_agent_extra_skill_sources)."""

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
                create_agent(MagicMock(), wd, checkpointer=InMemorySaver(),
                             **kwargs)
                return fake.call_args.kwargs

    def test_injected_default_reaches_deep_agent(self):
        inj = _fs()
        backend = self._build(default_backend=inj)["backend"]
        assert backend.default is inj

    def test_sandbox_default_enables_execute(self):
        inj = _FakeSandboxBackend(root_dir=tempfile.mkdtemp(), virtual_mode=True)
        backend = self._build(default_backend=inj)["backend"]
        assert backend.default is inj
        # The hinge: a SandboxBackendProtocol default => execute tool enabled.
        assert supports_execution(backend) is True

    def test_non_sandbox_default_does_not_enable_execute(self):
        backend = self._build(default_backend=_fs())["backend"]
        assert supports_execution(backend) is False

    def test_no_default_preserves_filesystem_backend(self):
        backend = self._build()["backend"]
        assert isinstance(backend.default, FilesystemBackend)
        assert supports_execution(backend) is False

    def test_default_and_sandbox_are_mutually_exclusive(self):
        with pytest.raises(ValueError):
            self._build(default_backend=_fs(), sandbox_backend=MagicMock())


class TestThreadDefaultBackend:
    """`Thread.__init__` forwards `default_backend` to `create_agent` —
    mirrors the extra_tools / loop_exploration_tools / extra_skill_sources
    forwarding tests in test_create_agent_extra_tools.py."""

    def _build(self, **kwargs):
        from assist.thread import Thread

        with patch("assist.thread.create_agent") as fake_ca, \
             patch("assist.thread.select_chat_model") as fake_model:
            fake_ca.return_value = MagicMock()
            fake_model.return_value = MagicMock()
            with tempfile.TemporaryDirectory() as wd:
                Thread(working_dir=wd, **kwargs)
                return fake_ca.call_args.kwargs

    def test_default_backend_none_passed_through(self):
        assert self._build()["default_backend"] is None

    def test_default_backend_forwarded_to_create_agent(self):
        inj = _fs()
        assert self._build(default_backend=inj)["default_backend"] is inj

    def test_thread_both_backends_raise(self):
        # create_agent is intentionally NOT patched so its mutual-exclusion
        # guard runs (it raises on the first statement, before any heavy work).
        from assist.thread import Thread

        with patch("assist.thread.select_chat_model") as fake_model:
            fake_model.return_value = MagicMock()
            with tempfile.TemporaryDirectory() as wd:
                with pytest.raises(ValueError):
                    Thread(working_dir=wd,
                           sandbox_backend=MagicMock(),
                           default_backend=_fs())
