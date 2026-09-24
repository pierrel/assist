"""Exact-generation retirement of shared Docker egress mutations."""
import fcntl
import sys
import time
from unittest.mock import MagicMock

import pytest

from assist.egress import runtime_state
from assist.sandbox_manager import (
    BROWSER_NETWORK, EGRESS_NETWORK, EGRESS_PROXY_NAME, SandboxManager,
    _bounded_egress_worker, _egress_proxy_config_hash, _load_egress_allowlist)


def _client():
    from docker.errors import NotFound

    client = MagicMock()
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

    def create(_image, **kwargs):
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
        return new

    client.containers.get.side_effect = get
    client.containers.run.side_effect = create
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
    client.containers.run.assert_not_called()


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
    client.containers.run.assert_not_called()


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
    client.containers.run.assert_not_called()


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
