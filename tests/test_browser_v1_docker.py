"""Isolated real-container proof for the turn browser's UID network fence.

The caller must build unique image tags and pass both names explicitly. This
test never names or changes production containers, networks, or image tags.
"""
import json
import os
from contextlib import nullcontext
from pathlib import Path
from uuid import uuid4

import pytest

from assist.browser import manager as browser, turn_runtime
from assist.egress.client_map import ClientRecord, record_client
from assist.sandbox_manager import SandboxManager


ORIGIN = """
from http.server import BaseHTTPRequestHandler, HTTPServer
import socket, threading
class Page(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'<html><body><h1>V1 isolated page</h1></body></html>'
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
def udp_echo():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
        server.bind(('0.0.0.0', 8001))
        while True:
            data, addr = server.recvfrom(128)
            server.sendto(data, addr)
threading.Thread(target=udp_echo, daemon=True).start()
HTTPServer(('0.0.0.0', 8000), Page).serve_forever()
"""

PROXY = """
import importlib.util, os
spec = importlib.util.spec_from_file_location('proxy', '/usr/local/bin/egress-proxy.py')
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)
original = proxy.vet_resolved
def fixture_address(host, port, *, global_only=True):
    if host == 'fixture-public.test' and global_only:
        return os.environ['BROWSER_TEST_ORIGIN_IP']
    return original(host, port, global_only=global_only)
proxy.vet_resolved = fixture_address
proxy.main()
"""


def _wait_absent(client, container_id, timeout=8):
    import docker
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            client.containers.get(container_id)
        except docker.errors.NotFound:
            return
        time.sleep(0.05)
    raise AssertionError(f"isolated container {container_id} was not removed")


def _chromium_alive(sandbox):
    processes = sandbox.top(ps_args="-eo pid,stat,comm")["Processes"]
    return any("chrome" in row[-1].lower() and not row[1].startswith("Z")
               for row in processes)


@pytest.mark.skipif(not (os.getenv("ASSIST_BROWSER_V1_SANDBOX_IMAGE") and
                         os.getenv("ASSIST_BROWSER_V1_PROXY_IMAGE")),
                    reason="requires explicit isolated image tags")
def test_browser_uid_reaches_only_dedicated_proxy(tmp_path, monkeypatch):
    import docker

    client = docker.from_env()
    suffix = uuid4().hex[:10]
    network_name = f"browser-v1-isolated-{suffix}"
    other_network_name = f"browser-v1-other-{suffix}"
    proxy_name = f"browser-v1-proxy-{suffix}"
    network = None
    other_network = None
    containers = []
    try:
        network = client.networks.create(
            network_name, driver="bridge", internal=True, enable_ipv6=False)
        other_network = client.networks.create(
            other_network_name, driver="bridge", internal=True,
            enable_ipv6=False)
        network.reload()
        cidr = network.attrs["IPAM"]["Config"][0]["Subnet"]
        origin = client.containers.run(
            os.environ["ASSIST_BROWSER_V1_PROXY_IMAGE"],
            ["python3", "-u", "-c", ORIGIN], detach=True, remove=True,
            network=network_name, name=f"browser-v1-origin-{suffix}")
        containers.append(origin)
        origin.reload()
        origin_ip = origin.attrs["NetworkSettings"]["Networks"][network_name]["IPAddress"]

        map_dir = tmp_path / "map"
        map_dir.mkdir(mode=0o700)
        proxy = client.containers.run(
            os.environ["ASSIST_BROWSER_V1_PROXY_IMAGE"],
            ["python3", "-u", "-c", PROXY], detach=True, remove=True,
            network=network_name, name=proxy_name,
            volumes={str(map_dir): {"bind": "/client-map", "mode": "ro"}},
            environment={"EGRESS_ALLOWLIST": "fixture-public.test",
                         "EGRESS_SANDBOX_CIDR": cidr,
                         "BROWSER_TEST_ORIGIN_IP": origin_ip})
        containers.append(proxy)
        other_network.connect(proxy)
        proxy.reload()
        SandboxManager._wait_for_egress_proxy_ready(proxy)
        proxy_ip = proxy.attrs["NetworkSettings"]["Networks"][network_name]["IPAddress"]
        other_proxy_ip = proxy.attrs["NetworkSettings"]["Networks"][
            other_network_name]["IPAddress"]

        work_dir = tmp_path / "thread" / "domain"
        work_dir.mkdir(parents=True)
        sandbox = client.containers.run(
            os.environ["ASSIST_BROWSER_V1_SANDBOX_IMAGE"],
            detach=True, remove=True, network=network_name,
            name=f"browser-v1-sandbox-{suffix}",
            user=f"{os.getuid()}:{os.getgid()}",
            volumes={str(work_dir): {"bind": "/workspace", "mode": "rw"}},
            tmpfs={"/run/assist-browser":
                   "rw,nosuid,nodev,size=134217728,uid=10001,gid=10001,mode=0700"},
            security_opt=["seccomp=" + (Path(__file__).resolve().parents[1]
                            / "dockerfiles/browser-seccomp.json").read_text()])
        containers.append(sandbox)
        sandbox.reload()
        sandbox_ip = sandbox.attrs["NetworkSettings"]["Networks"][network_name]["IPAddress"]
        record_client(str(map_dir), sandbox_ip,
                      ClientRecord("thread", sandbox.id, "sandbox"))

        monkeypatch.setattr(turn_runtime, "EGRESS_NETWORK", network_name)
        monkeypatch.setattr(turn_runtime, "EGRESS_PROXY_NAME", proxy_name)
        monkeypatch.setattr(turn_runtime, "_proxy_setup_lock", nullcontext)
        monkeypatch.setattr(turn_runtime.SandboxManager,
                            "_get_docker_client", classmethod(lambda _cls: client))
        monkeypatch.setattr(turn_runtime.runtime_state,
                            "assert_admissible", lambda *_args: None)
        monkeypatch.setattr(turn_runtime, "_check_lease", lambda _request: None)
        with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as stream:
            boot_id = stream.read().strip()
        import time
        request = {"generation": sandbox.id, "map_dir": str(map_dir),
                   "ip": sandbox_ip, "thread_id": "thread",
                   "threads_root": str(tmp_path), "work_dir": str(work_dir),
                   "run_id": "run", "mode": "public", "token": uuid4().hex,
                   "boot_id": boot_id,
                   "deadline_ns": time.clock_gettime_ns(time.CLOCK_BOOTTIME)
                   + 15_000_000_000}
        assert turn_runtime.launch(request)["generation"] == sandbox.id

        def socket_attempt(user, host, port):
            result = sandbox.exec_run([
                "python", "-c", "import socket; "
                f"socket.create_connection(({host!r},{port}),2).close()"],
                user=user)
            return result.exit_code

        assert socket_attempt("10001:10001", proxy_ip, 8889) == 0
        assert socket_attempt(f"{os.getuid()}:{os.getgid()}", proxy_ip, 8889) != 0
        assert socket_attempt(f"{os.getuid()}:{os.getgid()}", other_proxy_ip, 8889) != 0
        for host, port in ((proxy_ip, 8888), (origin_ip, 8000),
                           ("127.0.0.1", 8000)):
            assert socket_attempt("10001:10001", host, port) != 0
        gateway = network.attrs["IPAM"]["Config"][0].get("Gateway")
        if gateway:
            assert socket_attempt("10001:10001", gateway, 80) != 0
        ipv6 = sandbox.exec_run([
            "python", "-c", "import socket; "
            "socket.create_connection(('::1',8000),2).close()"],
            user="10001:10001")
        assert ipv6.exit_code != 0
        udp = sandbox.exec_run([
            "python", "-c", "import socket; "
            "s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); "
            "s.settimeout(1); "
            f"s.sendto(b'x',({origin_ip!r},8001)); s.recv(1)"],
            user="10001:10001")
        assert udp.exit_code != 0
        assert socket_attempt(f"{os.getuid()}:{os.getgid()}", proxy_ip, 8888) == 0

        payload = json.dumps({"session": request["token"], "operation": "open",
                              "args": {"url": "http://fixture-public.test:8000/"}})
        result = browser._docker_exec(sandbox.id, payload.encode(),
                                      in_sandbox=True)
        assert "V1 isolated page" in json.loads(result)["result"]["snapshot"]
        assert _chromium_alive(sandbox)
        # No host timer runs in this direct launch. PID 1's deadline must end
        # Chromium while the shell sandbox itself remains alive.
        deadline = time.monotonic() + 18
        while time.monotonic() < deadline and _chromium_alive(sandbox):
            time.sleep(0.1)
        sandbox.reload()
        assert sandbox.status == "running"
        assert not _chromium_alive(sandbox)
        assert socket_attempt(f"{os.getuid()}:{os.getgid()}", proxy_ip, 8889) != 0
        with pytest.raises(browser.BrowserUnavailable):
            browser._docker_exec(sandbox.id, payload.encode(), timeout=1,
                                 in_sandbox=True)
        # Host-owned exact-generation teardown stops the open Chromium tree.
        browser.BrowserSession._expire_sandbox(sandbox.id)
        _wait_absent(client, sandbox.id)
    finally:
        for container in reversed(containers):
            try:
                container.remove(force=True)
            except docker.errors.NotFound:
                pass
            except docker.errors.APIError as error:
                if error.status_code != 409:
                    raise
            _wait_absent(client, container.id)
        if network is not None:
            network.remove()
        if other_network is not None:
            other_network.remove()
