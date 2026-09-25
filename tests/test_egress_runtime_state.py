"""Exact-generation retirement of shared Docker egress mutations."""
import fcntl
import os
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from assist.egress import runtime_state
from assist.sandbox_manager import (
    BROWSER_NETWORK, EGRESS_NETWORK, EGRESS_PROXY_NAME, SandboxManager,
    _bounded_egress_worker, _egress_proxy_config_hash, _load_egress_allowlist,
    _proxy_setup_lock)


def _client():
    from docker.errors import NotFound

    client = MagicMock()
    client.api._version = "1.43"
    ordinary = MagicMock()
    ordinary.id = "ordinary-id"
    ordinary.attrs = {
        "Id": ordinary.id, "Driver": "bridge", "Internal": True,
        "EnableIPv6": False, "IPAM": {"Config": [{"Subnet": "172.30.0.0/16"}]}}
    browser = MagicMock()
    browser.id = "browser-id"
    browser.attrs = {
        "Id": browser.id, "Driver": "bridge", "Internal": True,
        "EnableIPv6": False,
        "Options": {"com.docker.network.bridge.gateway_mode_ipv4": "isolated"},
        "IPAM": {"Config": [{"Subnet": "172.31.0.0/16", "Gateway": ""}]}}
    networks = {EGRESS_NETWORK: ordinary, BROWSER_NETWORK: browser}
    client.networks.get.side_effect = lambda key: networks[key]
    current = {"proxy": None}
    old = MagicMock()
    old.id = "proxy-generation-A"
    old.status = "exited"
    old.labels = {"assist.egress-allowlist-hash": "stale"}
    old.attrs = {"NetworkSettings": {"Networks": {
        EGRESS_NETWORK: {"NetworkID": ordinary.id},
        BROWSER_NETWORK: {"NetworkID": browser.id}}}}
    current["proxy"] = old

    def get(key):
        proxy = current["proxy"]
        if proxy is not None and key in {EGRESS_PROXY_NAME, proxy.id}:
            return proxy
        raise NotFound("no such proxy")

    def create(**kwargs):
        new = MagicMock()
        new.id = "proxy-generation-B"
        new.status = "running"
        new.labels = kwargs["labels"]
        new.attrs = {"NetworkSettings": {"Networks": {
            EGRESS_NETWORK: {"NetworkID": ordinary.id},
            **({BROWSER_NETWORK: {"NetworkID": browser.id}}
               if kwargs["environment"]["EGRESS_BROWSER_CIDR"] else {})}}}
        new.logs.return_value = b"egress-proxy: listening on 0.0.0.0:8888"
        current["proxy"] = new
        return {"Id": new.id}

    client.containers.get.side_effect = get
    client.api.create_container.side_effect = create
    return client, ordinary, old, current


def test_late_proxy_remove_cannot_readmit_old_generation_after_restart(tmp_path,
                                                                     monkeypatch):
    threads = tmp_path / "threads"
    threads.mkdir()
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("ASSIST_THREADS_DIR", str(threads))
    monkeypatch.setenv("ASSIST_EGRESS_RUNTIME_DIR", str(runtime))
    client_map = tmp_path / "map"
    monkeypatch.setenv("ASSIST_EGRESS_CLIENT_MAP_DIR", str(client_map))
    client, _, old, current = _client()
    calls = 0

    def delayed_remove(*, force):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("daemon remove has not completed")
        if current["proxy"] is old:
            current["proxy"] = None

    old.remove.side_effect = delayed_remove
    with pytest.raises(TimeoutError, match="not completed"):
        SandboxManager._ensure_egress_proxy_running_direct(client)
    assert runtime_state.retirement("proxy", old.id) == "remove"

    # A clean-looking A is still retired after process restart, even when the
    # optional browser map is now unsafe. Ordinary base-only setup must use B.
    old.status = "running"
    old.labels["assist.egress-allowlist-hash"] = _egress_proxy_config_hash(
        ",".join(_load_egress_allowlist()), None,
        "ordinary-id:172.30.0.0/16|:")
    monkeypatch.setenv("ASSIST_EGRESS_CLIENT_MAP_DIR", str(threads / "bad-map"))
    SandboxManager._docker_client = None
    assert SandboxManager._ensure_egress_proxy_running_direct(client) == EGRESS_PROXY_NAME
    assert current["proxy"].id == "proxy-generation-B"
    assert calls == 2
    # A delayed daemon completion targets only A, never B by mutable name.
    old.remove(force=True)
    assert current["proxy"].id == "proxy-generation-B"
    assert runtime_state.retirement("proxy", old.id) == "remove"
    runtime_state.assert_admissible("proxy", current["proxy"].id)


def test_uncertain_network_connect_never_readmits_same_network(tmp_path,
                                                               monkeypatch):
    threads = tmp_path / "threads"
    threads.mkdir()
    monkeypatch.setenv("ASSIST_THREADS_DIR", str(threads))
    monkeypatch.setenv("ASSIST_EGRESS_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("ASSIST_EGRESS_CLIENT_MAP_DIR", raising=False)
    client, ordinary, old, current = _client()
    old.remove.side_effect = lambda *, force: current.update(proxy=None)
    ordinary.connect.side_effect = TimeoutError("daemon connect pending")
    with pytest.raises(TimeoutError, match="pending"):
        SandboxManager._ensure_egress_proxy_running_direct(client)
    assert runtime_state.retirement("network", ordinary.id) == (
        "connect:proxy-generation-B")
    current["proxy"].remove.side_effect = lambda *, force: current.update(proxy=None)
    with pytest.raises(RuntimeError, match="retired"):
        SandboxManager._ensure_egress_proxy_running_direct(client)


def test_unsafe_runtime_directory_blocks_mutation(tmp_path, monkeypatch):
    threads = tmp_path / "threads"
    threads.mkdir()
    monkeypatch.setenv("ASSIST_THREADS_DIR", str(threads))
    monkeypatch.setenv("ASSIST_EGRESS_RUNTIME_DIR", str(threads / "runtime"))
    client, _, old, _ = _client()
    with pytest.raises(RuntimeError, match="inside container mounts"):
        SandboxManager._ensure_egress_proxy_running_direct(client)
    old.remove.assert_not_called()
    client.api.create_container.assert_not_called()


@pytest.mark.parametrize("mount_name", [
    "ASSIST_EGRESS_CLIENT_MAP_DIR", "ASSIST_EGRESS_APPROVALS_DIR",
])
def test_ledger_inside_proxy_mount_blocks_mutation(tmp_path, monkeypatch,
                                                   mount_name):
    threads = tmp_path / "threads"
    threads.mkdir()
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    monkeypatch.setenv("ASSIST_THREADS_DIR", str(threads))
    monkeypatch.setenv(mount_name, str(mounted))
    monkeypatch.setenv("ASSIST_EGRESS_RUNTIME_DIR", str(mounted / "runtime"))
    client, _, old, _ = _client()
    with pytest.raises(RuntimeError, match="inside container mounts"):
        SandboxManager._ensure_egress_proxy_running_direct(client)
    old.remove.assert_not_called()
    client.api.create_container.assert_not_called()


def test_ledger_inside_run_mount_rejected(tmp_path, monkeypatch):
    mounted = tmp_path / "agent"
    mounted.mkdir()
    monkeypatch.setenv("ASSIST_THREADS_DIR", str(tmp_path / "threads"))
    monkeypatch.setenv("ASSIST_EGRESS_RUNTIME_DIR", str(mounted / "runtime"))
    with pytest.raises(RuntimeError, match="inside container mounts"):
        runtime_state.assert_outside_mounts(str(mounted))


def test_missing_initialized_ledger_fails_closed_before_mutation(tmp_path,
                                                               monkeypatch):
    threads = tmp_path / "threads"
    threads.mkdir()
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("ASSIST_THREADS_DIR", str(threads))
    monkeypatch.setenv("ASSIST_EGRESS_RUNTIME_DIR", str(runtime))
    runtime_state.assert_admissible("proxy", "unused")
    (runtime / "state.json").unlink()
    client, _, old, _ = _client()
    with pytest.raises(RuntimeError, match="ledger disappeared"):
        SandboxManager._ensure_egress_proxy_running_direct(client)
    old.remove.assert_not_called()
    client.api.create_container.assert_not_called()


def test_repeated_worker_timeout_releases_interprocess_lock(tmp_path):
    lock_path = tmp_path / "setup.lock"
    child = ("import fcntl,sys,time; "
             "f=open(sys.argv[1],'a+b'); fcntl.flock(f,fcntl.LOCK_EX); "
             "open(sys.argv[2],'w').write('ready'); time.sleep(60)")
    started = time.monotonic()
    for index in range(3):
        marker = tmp_path / f"ready-{index}"
        with pytest.raises(RuntimeError, match="timed out"):
            _bounded_egress_worker(
                [sys.executable, "-c", child, str(lock_path), str(marker)],
                timeout=0.5)
        assert marker.read_text() == "ready"
        with open(lock_path, "a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
    assert time.monotonic() - started < 4


def test_real_client_attribution_uses_killable_worker():
    class Client:
        pass

    with (patch("docker.DockerClient", Client),
          patch("assist.sandbox_manager._bounded_egress_worker",
                side_effect=RuntimeError("egress Docker setup timed out")) as worker):
        with pytest.raises(RuntimeError, match="timed out"):
            SandboxManager._record_egress_client(
                Client(), MagicMock(id="a" * 64), "/tmp/thread/domain",
                kind="sandbox")
    assert worker.call_args.args[0][3:] == [
        "record", "a" * 64, "/tmp/thread/domain", "sandbox"]


def test_record_publication_precedes_network_replacement_and_map_clear(
        tmp_path, monkeypatch):
    from assist.egress import client_map
    map_dir = tmp_path / "map"
    map_dir.mkdir()
    monkeypatch.setenv("ASSIST_EGRESS_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(client_map, "configured_directory", lambda: str(map_dir))
    client = MagicMock()
    proxy = MagicMock(status="running")
    proxy.attrs = {
        "Mounts": [{"Destination": "/client-map", "Source": str(map_dir)}],
        "NetworkSettings": {"Networks": {
            EGRESS_NETWORK: {"NetworkID": "ordinary-N1"}}}}
    client.containers.get.return_value = proxy
    sandbox = MagicMock(id="sandbox-S", status="running")
    sandbox.attrs = {"NetworkSettings": {"Networks": {
        EGRESS_NETWORK: {"NetworkID": "ordinary-N1", "IPAddress": "172.30.0.2"}}}}
    attempted, replaced = threading.Event(), threading.Event()
    replacement = None
    original_record = client_map.record_client

    def race(directory, ip, record):
        nonlocal replacement

        def replace():
            attempted.set()
            with _proxy_setup_lock():
                proxy.attrs["NetworkSettings"]["Networks"][EGRESS_NETWORK][
                    "NetworkID"] = "ordinary-N2"
                client_map.clear_clients(directory)
                replaced.set()

        replacement = threading.Thread(target=replace, daemon=True)
        replacement.start()
        assert attempted.wait(1)
        assert not replaced.wait(0.05), "replacement crossed record fence"
        original_record(directory, ip, record)

    with patch.object(client_map, "record_client", side_effect=race):
        assert SandboxManager._record_egress_client_direct(
            client, sandbox, "/tmp/thread/domain", kind="sandbox") == (
                str(map_dir), "172.30.0.2", "sandbox-S")
    replacement.join(2)
    assert replaced.is_set()
    assert client_map.read_client(str(map_dir), "172.30.0.2") is None


@pytest.mark.parametrize("retired_kind, identity", [
    ("proxy", "proxy-P"), ("network", "ordinary-N1")])
def test_shell_publication_rejects_retired_generation(
        tmp_path, monkeypatch, retired_kind, identity):
    from assist.egress import client_map
    map_dir = tmp_path / "map"
    map_dir.mkdir()
    monkeypatch.setenv("ASSIST_EGRESS_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(client_map, "configured_directory", lambda: str(map_dir))
    proxy = MagicMock(id="proxy-P", status="running")
    proxy.attrs = {"Mounts": [{"Destination": "/client-map", "Source": str(map_dir)}],
                   "NetworkSettings": {"Networks": {
                       EGRESS_NETWORK: {"NetworkID": "ordinary-N1"}}}}
    client = MagicMock()
    client.containers.get.return_value = proxy
    sandbox = MagicMock(id="sandbox-S", status="running")
    sandbox.attrs = {"NetworkSettings": {"Networks": {
        EGRESS_NETWORK: {"NetworkID": "ordinary-N1", "IPAddress": "172.30.0.2"}}}}
    runtime_state.retire(retired_kind, identity, "uncertain mutation")
    with pytest.raises(RuntimeError, match="retired"):
        SandboxManager._record_egress_client_direct(
            client, sandbox, "/tmp/thread/domain", kind="sandbox")
    assert client_map.read_client(str(map_dir), "172.30.0.2") is None


def test_missing_proxy_image_fails_before_old_proxy_removal_or_create_token(
        tmp_path, monkeypatch):
    from docker.errors import NotFound
    monkeypatch.setenv("ASSIST_EGRESS_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("ASSIST_EGRESS_CLIENT_MAP_DIR", raising=False)
    client, _, old, _ = _client()
    client.images.get.side_effect = NotFound("image missing")
    with pytest.raises(NotFound):
        SandboxManager._ensure_egress_proxy_running_direct(client)
    old.remove.assert_not_called()
    client.api.create_container.assert_not_called()
    assert runtime_state.pending_creation("proxy", EGRESS_PROXY_NAME) is None


def test_daemon_rejected_proxy_create_settles_token_but_timeout_does_not(
        tmp_path, monkeypatch):
    from docker.errors import APIError
    from unittest.mock import MagicMock
    monkeypatch.setenv("ASSIST_EGRESS_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("ASSIST_EGRESS_CLIENT_MAP_DIR", raising=False)
    client, _, old, current = _client()
    old.remove.side_effect = lambda *, force: current.update(proxy=None)
    create = client.api.create_container.side_effect
    rejected = MagicMock(status_code=400)
    client.api.create_container.side_effect = APIError("invalid create config", response=rejected)
    with pytest.raises(APIError):
        SandboxManager._ensure_egress_proxy_running_direct(client)
    assert runtime_state.pending_creation("proxy", EGRESS_PROXY_NAME) is None
    client.api.create_container.side_effect = create
    assert SandboxManager._ensure_egress_proxy_running_direct(client) == EGRESS_PROXY_NAME

    # An I/O timeout has no terminal daemon response; recovery must retain
    # its durable pending token rather than trusting a clean name lookup.
    current["proxy"] = None
    client.api.create_container.side_effect = TimeoutError("create pending")
    with pytest.raises(TimeoutError):
        SandboxManager._ensure_egress_proxy_running_direct(client)
    assert runtime_state.pending_creation("proxy", EGRESS_PROXY_NAME)
    with pytest.raises(RuntimeError, match="creation outcome is uncertain"):
        SandboxManager._ensure_egress_proxy_running_direct(client)


def test_post_create_proxy_inspect_error_keeps_pending_token(tmp_path, monkeypatch):
    from docker.errors import NotFound
    monkeypatch.setenv("ASSIST_EGRESS_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("ASSIST_EGRESS_CLIENT_MAP_DIR", raising=False)
    client, _, old, current = _client()
    old.remove.side_effect = lambda *, force: current.update(proxy=None)
    original_get = client.containers.get.side_effect

    def get(key):
        if key == "proxy-generation-B":
            raise NotFound("inspect failed after successful POST")
        return original_get(key)

    client.containers.get.side_effect = get
    with pytest.raises(NotFound, match="inspect failed"):
        SandboxManager._ensure_egress_proxy_running_direct(client)
    assert runtime_state.pending_creation("proxy", EGRESS_PROXY_NAME)


def test_local_proxy_argument_error_never_journals_create(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSIST_EGRESS_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("ASSIST_EGRESS_CLIENT_MAP_DIR", raising=False)
    client, _, old, _ = _client()
    monkeypatch.setattr("docker.models.containers._create_container_args",
                        lambda _kwargs: (_ for _ in ()).throw(ValueError("bad local args")))
    with pytest.raises(ValueError, match="bad local args"):
        SandboxManager._ensure_egress_proxy_running_direct(client)
    assert runtime_state.pending_creation("proxy", EGRESS_PROXY_NAME) is None
    client.api.create_container.assert_not_called()
    old.remove.assert_called_once_with(force=True)


def test_daemon_rejected_network_create_settles_token(tmp_path, monkeypatch):
    from docker.errors import APIError, NotFound
    from unittest.mock import MagicMock
    monkeypatch.setenv("ASSIST_EGRESS_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("ASSIST_EGRESS_CLIENT_MAP_DIR", raising=False)
    client, _, _, _ = _client()
    original_get = client.networks.get.side_effect

    def get(key):
        if key == EGRESS_NETWORK:
            raise NotFound("network missing")
        return original_get(key)

    client.networks.get.side_effect = get
    client.api.create_network.side_effect = APIError(
        "invalid network config", response=MagicMock(status_code=400))
    with pytest.raises(APIError):
        SandboxManager._ensure_egress_proxy_running_direct(client)
    assert runtime_state.pending_creation("network", EGRESS_NETWORK) is None


def test_post_create_network_inspect_error_keeps_pending_token(tmp_path, monkeypatch):
    from docker.errors import NotFound
    monkeypatch.setenv("ASSIST_EGRESS_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("ASSIST_EGRESS_CLIENT_MAP_DIR", raising=False)
    client, _, _, _ = _client()
    client.networks.get.side_effect = NotFound("inspect failed after POST")
    client.api.create_network.return_value = {"Id": "new-network-id"}
    with pytest.raises(NotFound):
        SandboxManager._ensure_egress_proxy_running_direct(client)
    assert runtime_state.pending_creation("network", EGRESS_NETWORK)


def test_failed_old_proxy_remove_preserves_active_map(tmp_path, monkeypatch):
    from docker.errors import APIError
    from assist.egress.client_map import ClientRecord, read_client, record_client
    map_dir = tmp_path / "map"
    monkeypatch.setenv("ASSIST_EGRESS_CLIENT_MAP_DIR", str(map_dir))
    monkeypatch.setenv("ASSIST_EGRESS_RUNTIME_DIR", str(tmp_path / "runtime"))
    ip = "172.31.0.2"
    record_client(str(map_dir), ip, ClientRecord("active", "generation", "browser",
                                                 "public"))
    client, _, old, _ = _client()
    old.remove.side_effect = APIError(
        "remove rejected", response=MagicMock(status_code=500))
    with pytest.raises(RuntimeError, match="removal is unconfirmed"):
        SandboxManager._ensure_egress_proxy_running_direct(client)
    assert read_client(str(map_dir), ip).generation == "generation"


@pytest.mark.parametrize("old_status", ["running", "exited"])
def test_same_map_and_network_replacement_preserves_active_browser_record(
        tmp_path, monkeypatch, old_status):
    from assist.egress.client_map import ClientRecord, read_client, record_client
    map_dir = tmp_path / "map"
    monkeypatch.setenv("ASSIST_EGRESS_CLIENT_MAP_DIR", str(map_dir))
    monkeypatch.setenv("ASSIST_EGRESS_RUNTIME_DIR", str(tmp_path / "runtime"))
    ip = "172.31.0.2"
    record_client(str(map_dir), ip, ClientRecord("active", "generation", "browser",
                                                 "public"))
    client, _, old, current = _client()
    old.status = old_status
    old.attrs["Mounts"] = [{"Destination": "/client-map", "Source": str(map_dir)}]
    old.attrs["Config"] = {"User": f"{os.getuid()}:{os.getgid()}"}
    old.remove.side_effect = lambda *, force: current.update(proxy=None)
    assert SandboxManager._ensure_egress_proxy_running_direct(client) == EGRESS_PROXY_NAME
    assert read_client(str(map_dir), ip).generation == "generation"


def test_new_network_clears_target_map_after_confirmed_remove(tmp_path, monkeypatch):
    from assist.egress.client_map import ClientRecord, read_client, record_client
    map_dir = tmp_path / "map"
    monkeypatch.setenv("ASSIST_EGRESS_CLIENT_MAP_DIR", str(map_dir))
    monkeypatch.setenv("ASSIST_EGRESS_RUNTIME_DIR", str(tmp_path / "runtime"))
    ip = "172.31.0.2"
    record_client(str(map_dir), ip, ClientRecord("old", "old-generation", "browser",
                                                 "public"))
    client, _, old, current = _client()
    old.attrs["Mounts"] = [{"Destination": "/client-map", "Source": str(map_dir)}]
    old.attrs["Config"] = {"User": f"{os.getuid()}:{os.getgid()}"}
    old.attrs["NetworkSettings"]["Networks"][EGRESS_NETWORK]["NetworkID"] = "old-network"
    old.remove.side_effect = lambda *, force: current.update(proxy=None)
    assert SandboxManager._ensure_egress_proxy_running_direct(client) == EGRESS_PROXY_NAME
    assert read_client(str(map_dir), ip) is None
