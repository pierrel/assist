"""Native agent state is shared by trusted factories and survives generations."""
import os
import shlex
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest


def test_agent_mount_rejects_failed_workspace_owner_alignment(tmp_path, monkeypatch):
    from assist import sandbox_manager as sandbox_module
    from assist.sandbox_manager import SandboxManager
    work = tmp_path / "domain"
    agent = tmp_path / "agent"
    work.mkdir()
    agent.mkdir()
    original_stat = os.stat
    values = list(original_stat(work))
    values[4] += 1
    foreign_owner = os.stat_result(values)
    monkeypatch.setattr(sandbox_module.os, "stat", lambda path, **kw:
        foreign_owner if path == work or path == str(work) else original_stat(path, **kw))
    monkeypatch.setattr(sandbox_module.os, "chown", MagicMock(side_effect=PermissionError("not owner")))
    client = MagicMock()
    monkeypatch.setattr(SandboxManager, "_get_docker_client", lambda: client)
    monkeypatch.setattr(SandboxManager, "_ensure_egress_proxy_running", lambda *_a: None)
    with pytest.raises(RuntimeError, match="owned by the sandbox user"):
        SandboxManager.get_sandbox_backend(str(work), agent_dir=str(agent))
    client.containers.run.assert_not_called()


@pytest.mark.skipif(os.getenv("ASSIST_BROWSER_DOCKER_TEST") != "1",
                    reason="requires an explicit isolated sandbox image and Docker")
def test_parent_child_and_later_turn_share_native_agent_state(tmp_path, monkeypatch):
    import docker
    from assist import sandbox_manager as sandbox_module
    from assist.browser import authority
    from assist.middleware.read_only_enforcer import ReadOnlyEnforcerMiddleware
    from assist.run_service import RunService
    from assist.sandbox_manager import SandboxManager
    from assist.thread_manager import ThreadManager

    client = docker.from_env(timeout=5)
    image = os.environ["ASSIST_BROWSER_TEST_SANDBOX_IMAGE"]
    network = client.networks.create("assist-agent-state-" + uuid4().hex[:10], internal=True)
    root = tmp_path / "threads"
    work = root / "parent" / "domain"
    work.mkdir(parents=True)
    agent = root / "parent" / "agent"
    agent.mkdir(mode=0o755)
    (agent / "memory.md").write_text("PARENT_NOTE\n")
    outside = tmp_path / "host-only.txt"
    outside.write_text("UNMOUNTED_HOST_CANARY")
    for tid in ("delegate", "research", "context"):
        (root / tid).mkdir()
    (tmp_path / "map").mkdir(mode=0o700)
    monkeypatch.setenv("ASSIST_THREADS_DIR", str(root))
    monkeypatch.setenv("ASSIST_EGRESS_CLIENT_MAP_DIR", str(tmp_path / "map"))
    monkeypatch.setenv("ASSIST_EGRESS_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("ASSIST_PRIVATE_PROFILE_CANARY", "HOST_ENV_CANARY")
    monkeypatch.setattr(SandboxManager, "_containers", {})
    monkeypatch.setattr(SandboxManager, "_generation_owners", {})
    monkeypatch.setattr(SandboxManager, "_get_docker_client", lambda: client)
    monkeypatch.setattr(SandboxManager, "_ensure_egress_proxy_running", lambda *_a: None)
    monkeypatch.setattr(SandboxManager, "_record_egress_client", lambda *_a, **_k: None)
    monkeypatch.setattr(SandboxManager, "_forget_egress_client", lambda *_a, **_k: None)
    monkeypatch.setattr(sandbox_module, "SANDBOX_IMAGE", image)
    monkeypatch.setattr(sandbox_module, "EGRESS_NETWORK", network.name)
    authority.mark_new_thread(str(root), "parent")
    runs = RunService(str(root))
    manager = ThreadManager(str(root))
    manager._model = MagicMock()  # Graph factories run; no provider request does.
    generations = []
    backends = []

    def composition(tid, role, backend):
        with patch("assist.agent.create_deep_agent") as graph:
            graph.return_value = MagicMock()
            manager.get(tid, working_dir=str(work), sandbox_backend=backend, assistant_id=role)
            return graph.call_args.kwargs

    try:
        for turn in (1, 2):
            run = runs.create("parent", "general-agent", "Synthetic shared-state contract", user_origin=True)
            backend = SandboxManager.get_sandbox_backend(
                str(work), agent_dir=str(agent), browser_capable=True,
                thread_scope=(str(root), "parent"), owner_run_id=run.id)
            assert backend is not None
            backends.append(backend)
            monkeypatch.setattr(backend, "_proxy_logs_since", lambda *_a: None)
            generations.append(backend.container.id)
            assert backend.native_agent_dir is True
            assert agent.stat().st_mode & 0o777 == 0o700
            assert backend.execute("cat /agent/memory.md").exit_code == 0
            denied = backend.container.exec_run(["cat", "/agent/memory.md"], user="10001:10001")
            assert denied.exit_code != 0 and b"PARENT_NOTE" not in denied.output
            assert backend.execute("test ! -e " + shlex.quote(str(outside))).exit_code == 0
            backend.container.reload()
            assert not any(value.startswith("ASSIST_") for value in backend.container.attrs["Config"]["Env"])
            parent = composition("parent", "general-agent", backend)
            assert "/agent/" not in parent["backend"].routes
            if turn == 1:
                child = composition("delegate", "delegate-agent", backend)
                assert "/agent/" not in child["backend"].routes
                assert {item["name"] for item in child["subagents"]} == {
                    "context-agent", "research-agent", "critique-agent"}
                assert child["backend"].write("/agent/child.md", "CHILD_NOTE\n").error is None
                assert child["backend"].execute("printf 'CHILD_APPEND\\n' >> /agent/memory.md").exit_code == 0
                research = composition("research", "research-agent", backend)
                assert research["backend"].read("/agent/memory.md").error is None
                assert research["backend"].write("/agent/research.md", "RESEARCH_NOTE\n").error is None
                context = composition("context", "context-agent", backend)
                assert context["backend"].read("/agent/memory.md").error is None
                assert any(isinstance(m, ReadOnlyEnforcerMiddleware) for m in context["middleware"])
            else:
                assert "CHILD_APPEND" in backend.execute("cat /agent/memory.md").output
                assert parent["backend"].read("/agent/child.md").error is None
                assert parent["backend"].read("/agent/research.md").error is None
            SandboxManager.cleanup(str(work), backend.container)
        assert generations[0] != generations[1]
        print({"generations": generations, "trusted_child_state_persisted": True,
               "browser_uid_denied": True, "unmounted_host_denied": True})
    finally:
        for backend in backends:
            try:
                SandboxManager.cleanup(str(work), backend.container)
            finally:
                try:
                    backend.container.remove(force=True)
                except docker.errors.NotFound:
                    pass
        manager.conn.close()
        network.remove()
