"""A turn browser cannot start before its exact network boundary is armed."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from assist.browser import turn_runtime
from assist.egress.client_map import ClientRecord


def _fixture(monkeypatch, shell_user="1000:1000"):
    sandbox = MagicMock(status="running", id="generation")
    sandbox.attrs = {
        "Config": {"User": shell_user},
        "Mounts": [{"Destination": "/workspace", "Source": "/tmp/thread/domain"}],
        "NetworkSettings": {"Networks": {"assist-egress-network": {
            "NetworkID": "network", "IPAddress": "172.20.0.8"}}},
    }
    proxy = MagicMock(status="running", id="proxy-generation")
    proxy.attrs = {
        "Mounts": [{"Destination": "/client-map", "Source": "/tmp/map"}],
        "NetworkSettings": {"Networks": {
            "assist-egress-network": {
                "NetworkID": "network", "IPAddress": "172.20.0.3"},
            "assist-browser-network": {
                "NetworkID": "browser-network", "IPAddress": "172.21.0.3"},
            "bridge": {"NetworkID": "default-network", "IPAddress": "172.17.0.3"}}},
    }
    client = MagicMock()
    client.containers.get.side_effect = lambda name: (
        sandbox if name == "generation" else proxy)
    monkeypatch.setattr(turn_runtime.SandboxManager, "_get_docker_client",
                        classmethod(lambda _cls: client))
    monkeypatch.setattr(turn_runtime, "_proxy_setup_lock", nullcontext)
    monkeypatch.setattr(turn_runtime.runtime_state, "assert_admissible",
                        lambda *_args: None)
    monkeypatch.setattr(turn_runtime, "_check_lease", lambda _request: None)
    monkeypatch.setattr(turn_runtime, "read_client", lambda *_args: ClientRecord(
        "thread", "generation", "sandbox"))
    request = {"generation": "generation", "map_dir": "/tmp/map",
               "ip": "172.20.0.8", "thread_id": "thread",
               "work_dir": "/tmp/thread/domain", "threads_root": "/tmp",
               "run_id": "run", "token": "private-token"}
    return sandbox, request


def test_turn_start_fences_before_map_publication_and_worker(monkeypatch):
    sandbox, request = _fixture(monkeypatch)
    events = []
    from assist.browser import manager
    def fence(argv, **_kwargs):
        assert argv[-5:] == ["/opt/assist/install-browser-fence.py",
                             "172.20.0.3", "172.17.0.3",
                             "172.20.0.3", "172.21.0.3"]
        events.append("fence")
        return b""

    monkeypatch.setattr(manager, "_bounded_cli", fence)
    monkeypatch.setattr(turn_runtime, "arm_sandbox_browser",
                        lambda *_args, **_kwargs: events.append("publish"))
    monkeypatch.setattr(manager, "_docker_exec",
                        lambda *_args, **_kwargs: (
                            b'{"result":{"pid":1,"uid":["10001","10001","10001","10001"],'
                            b'"gid":["10001","10001","10001","10001"],'
                            b'"caps":["0000000000000000","0000000000000000",'
                            b'"0000000000000000","0000000000000000",'
                            b'"0000000000000000"],"no_new_privs":"1",'
                            b'"seccomp":"2"}}'))

    def execute(command, **kwargs):
        events.append("start" if command[0] == "/usr/bin/unshare" else "ready")
        return SimpleNamespace(exit_code=0)

    sandbox.exec_run.side_effect = execute
    assert turn_runtime.launch(request) == {
        "generation": "generation", "ip": "172.20.0.8", "map_dir": "/tmp/map"}
    assert events == ["fence", "publish", "start", "ready"]
    assert sandbox.exec_run.call_args_list[0].kwargs["user"] == "0:0"
    assert sandbox.exec_run.call_args_list[0].kwargs["privileged"] is True
    assert sandbox.exec_run.call_args_list[0].args[0] == [
        "/usr/bin/unshare", "--pid", "--fork", "--kill-child", "--",
        "/usr/bin/setpriv", "--reuid=10001", "--regid=10001",
        "--clear-groups", "--inh-caps=-all", "--ambient-caps=-all",
        "--bounding-set=-all", "--no-new-privs", "--",
        "/usr/bin/python", "-I", "/opt/assist/launch-turn-browser.py"]
    assert sandbox.exec_run.call_args_list[0].kwargs["environment"][
        "BROWSER_PROXY_HOST"] == "172.20.0.3"


def test_browser_uid_collision_refuses_before_fence(monkeypatch):
    sandbox, request = _fixture(monkeypatch, shell_user="10001:10001")
    from assist.browser import manager
    monkeypatch.setattr(manager, "_bounded_cli", lambda *_args, **_kwargs: (
        _ for _ in ()).throw(AssertionError("fence must not run")))
    with pytest.raises(RuntimeError, match="identity changed"):
        turn_runtime.launch(request)
    sandbox.exec_run.assert_not_called()


def test_missing_proxy_attachment_address_refuses_before_fence(monkeypatch):
    sandbox, request = _fixture(monkeypatch)
    proxy = turn_runtime.SandboxManager._get_docker_client().containers.get(
        turn_runtime.EGRESS_PROXY_NAME)
    proxy.attrs["NetworkSettings"]["Networks"]["bridge"]["IPAddress"] = ""
    from assist.browser import manager
    monkeypatch.setattr(manager, "_bounded_cli", lambda *_args, **_kwargs: (
        _ for _ in ()).throw(AssertionError("fence must not run")))
    with pytest.raises(RuntimeError, match="lacks IPv4"):
        turn_runtime.launch(request)
    sandbox.exec_run.assert_not_called()


def test_changed_proxy_attachment_refuses_map_publication(monkeypatch):
    sandbox, request = _fixture(monkeypatch)
    proxy = turn_runtime.SandboxManager._get_docker_client().containers.get(
        turn_runtime.EGRESS_PROXY_NAME)
    from assist.browser import manager
    def change_after_fence(_argv, **_kwargs):
        proxy.attrs["NetworkSettings"]["Networks"]["bridge"]["IPAddress"] = "172.17.0.9"
        return b""
    monkeypatch.setattr(manager, "_bounded_cli", change_after_fence)
    monkeypatch.setattr(turn_runtime, "arm_sandbox_browser", lambda *_args: (
        _ for _ in ()).throw(AssertionError("map must not be armed")))
    with pytest.raises(RuntimeError, match="changed during network fence"):
        turn_runtime.launch(request)
    sandbox.exec_run.assert_not_called()
