"""Tests for the `default_backend` injection seam.

Pins the `AgentSpec.default_backend` wiring through create_agent and
Thread, plus the backend-factory and create_context_agent
`default_backend` params (which are direct parameters, not part of
the embedder spec).

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
from assist.spec import AgentSpec
from deepagents.backends import FilesystemBackend, StateBackend
from deepagents.backends.protocol import SandboxBackendProtocol, WriteResult
from deepagents.middleware.filesystem import supports_execution


def request_egress(host: str, port: int, task: str) -> str:
    return f"{host}:{port} {task}"


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

    def test_user_alias_writes_to_the_default_workspace(self, tmp_path):
        backend = create_composite_backend(str(tmp_path))
        result = backend.write("/user/notes.md", "# Notes\n")
        assert result.error is None or result.error == ""
        assert (tmp_path / "notes.md").read_text() == "# Notes\n"
        # The alias is implemented by the default backend, not a second route:
        # root glob/grep must not duplicate the same user files.
        assert "/user/" not in backend.routes
        paths = [entry["path"] for entry in backend.glob("*.md", "/").matches]
        assert paths.count("/notes.md") == 1
        assert "/user/notes.md" not in paths

    def test_write_recovers_an_existing_empty_user_file(self, tmp_path):
        """An empty existing memory source is safe to initialize once."""
        (tmp_path / "AGENTS.md").touch()
        backend = create_composite_backend(str(tmp_path))

        result = backend.write("/user/AGENTS.md", "User has 3 cats.\n")

        assert result.error is None or result.error == ""
        assert (tmp_path / "AGENTS.md").read_text() == "User has 3 cats.\n"

    def test_write_keeps_nonempty_user_file_no_clobber(self, tmp_path):
        """Empty-file recovery must not turn write_file into overwrite."""
        (tmp_path / "AGENTS.md").write_text("User has 3 cats.\n")
        backend = create_composite_backend(str(tmp_path))

        result = backend.write("/user/AGENTS.md", "User has 2 dogs.\n")

        assert result.error
        assert (tmp_path / "AGENTS.md").read_text() == "User has 3 cats.\n"

    def test_write_does_not_retry_a_non_collision_error(self, tmp_path):
        backend = create_composite_backend(str(tmp_path)).default
        failure = WriteResult(error="Cannot write to /AGENTS.md: permission denied")
        with patch.object(FilesystemBackend, "write", return_value=failure), \
             patch.object(FilesystemBackend, "edit") as edit:
            result = backend.write("/user/AGENTS.md", "User has 3 cats.\n")

        assert result == failure
        edit.assert_not_called()

    def test_scratch_route_keeps_tmp_out_of_user_workspace(self, tmp_path):
        scratch = tmp_path / "scratch"
        backend = create_composite_backend(
            str(tmp_path), extra_routes={"/tmp/": FilesystemBackend(
                root_dir=scratch, virtual_mode=True)})
        result = backend.write("/tmp/listing.md", "# Listing\n")
        assert result.error is None or result.error == ""
        assert (scratch / "listing.md").read_text() == "# Listing\n"
        assert not (tmp_path / "tmp" / "listing.md").exists()


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
        backend = self._build(spec=AgentSpec(default_backend=inj))["backend"]
        assert backend.default is inj

    def test_sandbox_default_enables_execute(self):
        inj = _FakeSandboxBackend(root_dir=tempfile.mkdtemp(), virtual_mode=True)
        backend = self._build(spec=AgentSpec(default_backend=inj))["backend"]
        assert backend.default is inj
        # The hinge: a SandboxBackendProtocol default => execute tool enabled.
        assert supports_execution(backend) is True

    def test_execution_automatically_adds_configured_egress_tools(self):
        """The execution capability, not a profile-specific call site, owns egress."""
        from assist import agent as agent_mod
        from assist.agent import set_execution_egress_tools
        from assist.middleware.skills_middleware import SmallModelSkillsMiddleware

        prior = agent_mod._execution_egress_tools
        set_execution_egress_tools((request_egress,))
        try:
            kwargs = self._build(spec=AgentSpec(default_backend=_FakeSandboxBackend(
                root_dir=tempfile.mkdtemp(), virtual_mode=True)))
        finally:
            set_execution_egress_tools(prior)

        names = {getattr(tool, "name", getattr(tool, "__name__", None))
                 for tool in kwargs["tools"]}
        assert "request_egress" in names
        skills = next(item for item in kwargs["middleware"]
                      if isinstance(item, SmallModelSkillsMiddleware))
        assert "request_egress" in skills._registered_tools

    def test_non_sandbox_default_does_not_enable_execute(self):
        backend = self._build(spec=AgentSpec(default_backend=_fs()))["backend"]
        assert supports_execution(backend) is False

    def test_no_default_preserves_filesystem_backend(self):
        backend = self._build()["backend"]
        assert isinstance(backend.default, FilesystemBackend)
        assert supports_execution(backend) is False

    def test_local_scratch_dir_is_routed_at_tmp(self, tmp_path):
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        backend = self._build(scratch_dir=str(scratch))["backend"]
        result = backend.write("/tmp/listing.md", "# Listing\n")
        assert result.error is None or result.error == ""
        assert (scratch / "listing.md").read_text() == "# Listing\n"

    def test_default_and_sandbox_are_mutually_exclusive(self):
        with pytest.raises(ValueError):
            self._build(spec=AgentSpec(default_backend=_fs()), sandbox_backend=MagicMock())


class TestSubagentDefaultBackendInheritance:
    """`default_backend` must reach `create_context_agent` so the subagent's
    filesystem tools resolve against the embedder's backend.  Without this,
    the context-agent (which the parent delegates to for "find files" work)
    falls back to a standard FilesystemBackend rooted at `working_dir`, and
    file-chat queries list the server's `/skills/` instead of the phone's
    workdir.  Bit emacsos's file-chat live-test on 2026-05-28: agent reply
    was \"The only files present are 5 SKILL.md files under /skills/\" —
    StateBackend territory, not the user's playground.

    Mirror of `TestCreateAgentDefaultBackend`'s patching pattern."""

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
                return fake_ctx.call_args.kwargs

    def test_default_backend_reaches_context_agent(self):
        inj = _fs()
        kwargs = self._build(spec=AgentSpec(default_backend=inj))
        assert kwargs.get("default_backend") is inj

    def test_no_default_backend_passes_none_to_context_agent(self):
        kwargs = self._build()
        # Either absent or explicitly None — both mean "no override."
        assert kwargs.get("default_backend") is None

    def test_context_agent_uses_default_backend_in_standard_path(self):
        """End-to-end through `create_context_agent`: when a default backend
        is supplied (no sandbox_backend), the resulting composite's default
        is the injected backend, not a fresh FilesystemBackend."""
        from assist.agent import create_context_agent

        inj = _fs()
        with patch("assist.agent.create_deep_agent") as fake:
            fake.return_value = MagicMock()
            create_context_agent(MagicMock(), "/tmp", default_backend=inj)
            backend = fake.call_args.kwargs["backend"]
            assert backend.default is inj


class TestThreadDefaultBackend:
    """`Thread.__init__` forwards the spec (carrying default_backend)
    to `create_agent`; the mutual exclusion with sandbox_backend is
    create_agent's check."""

    def _build(self, **kwargs):
        from assist.thread import Thread

        with patch("assist.thread.create_agent") as fake_ca, \
             patch("assist.thread.select_assistant_model") as fake_model:
            fake_ca.return_value = MagicMock()
            fake_model.return_value = MagicMock()
            with tempfile.TemporaryDirectory() as wd:
                Thread(working_dir=wd, **kwargs)
                return fake_ca.call_args.kwargs

    def test_default_spec_none_passed_through(self):
        assert self._build()["spec"] is None

    def test_default_backend_forwarded_via_spec(self):
        inj = _fs()
        spec = AgentSpec(default_backend=inj)
        assert self._build(spec=spec)["spec"].default_backend is inj

    def test_thread_both_backends_raise(self):
        # create_agent is intentionally NOT patched so its mutual-exclusion
        # guard runs (it raises on the first statement, before any heavy work).
        from assist.thread import Thread

        with patch("assist.thread.select_assistant_model") as fake_model:
            fake_model.return_value = MagicMock()
            with tempfile.TemporaryDirectory() as wd:
                with pytest.raises(ValueError):
                    Thread(working_dir=wd,
                           sandbox_backend=MagicMock(),
                           spec=AgentSpec(default_backend=_fs()))
