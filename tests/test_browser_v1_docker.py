"""Isolated Docker proof of the sandbox-owned browser's UID and lifetime.

The caller supplies unique image tags; this test never builds or retags them.
"""
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from uuid import uuid4

import pytest

from assist.browser import manager as browser, turn_runtime
from assist.egress.client_map import ClientRecord, record_client
from assist.sandbox_manager import SandboxManager

ORIGIN = """
from http.server import BaseHTTPRequestHandler, HTTPServer
class Page(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'<html><body><h1>Sandbox lifetime fixture</h1></body></html>'
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
HTTPServer(('0.0.0.0', 8000), Page).serve_forever()
"""


def _wait_absent(client, container_id, timeout=40):
    import docker
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            client.containers.get(container_id)
        except docker.errors.NotFound:
            return
        time.sleep(0.1)
    raise AssertionError("turn sandbox did not expire after test-only common TTL")


@pytest.mark.skipif(not (os.getenv("ASSIST_BROWSER_V1_SANDBOX_IMAGE") and
                         os.getenv("ASSIST_BROWSER_V1_PROXY_IMAGE")),
                    reason="requires explicit isolated image tags")
def test_uid_fence_and_outer_pid1_expiry(tmp_path, monkeypatch):
    import docker

    client = docker.from_env()
    suffix = uuid4().hex[:10]
    ordinary_name = f"browser-v1-ordinary-{suffix}"
    browser_name = f"browser-v1-isolated-{suffix}"
    proxy_name = f"browser-v1-proxy-{suffix}"
    created = []
    networks = []
    try:
        ordinary = client.networks.create(ordinary_name, driver="bridge", internal=True)
        isolated = client.networks.create(
            browser_name, driver="bridge", internal=True, enable_ipv6=False,
            options={"com.docker.network.bridge.gateway_mode_ipv4": "isolated"})
        networks.extend([ordinary, isolated])
        ordinary.reload()
        isolated.reload()
        ordinary_cidr = ordinary.attrs["IPAM"]["Config"][0]["Subnet"]
        browser_cidr = isolated.attrs["IPAM"]["Config"][0]["Subnet"]
        origin = client.containers.run(
            os.environ["ASSIST_BROWSER_V1_PROXY_IMAGE"],
            ["python3", "-u", "-c", ORIGIN], detach=True, remove=True,
            network=ordinary_name, name=f"browser-v1-origin-{suffix}")
        created.append(origin)
        origin.reload()
        origin_ip = origin.attrs["NetworkSettings"]["Networks"][ordinary_name]["IPAddress"]
        map_dir = tmp_path / "map"
        map_dir.mkdir(mode=0o700)
        proxy = client.containers.run(
            os.environ["ASSIST_BROWSER_V1_PROXY_IMAGE"],
            detach=True, remove=True, name=proxy_name,
            volumes={str(map_dir): {"bind": "/client-map", "mode": "ro"}},
            environment={"EGRESS_ALLOWLIST": origin_ip,
                         "EGRESS_SANDBOX_CIDR": ordinary_cidr,
                         "EGRESS_BROWSER_CIDR": browser_cidr})
        created.append(proxy)
        ordinary.connect(proxy)
        isolated.connect(proxy)
        SandboxManager._wait_for_egress_proxy_ready(proxy)
        proxy.reload()
        attachments = proxy.attrs["NetworkSettings"]["Networks"]
        proxy_ips = [entry["IPAddress"] for entry in attachments.values()]
        assert len(proxy_ips) == 3 and len(set(proxy_ips)) == 3
        ordinary_proxy_ip = attachments[ordinary_name]["IPAddress"]

        work_dir = tmp_path / "thread" / "domain"
        work_dir.mkdir(parents=True)
        sandbox = client.containers.run(
            os.environ["ASSIST_BROWSER_V1_SANDBOX_IMAGE"],
            ["sleep", "30"], detach=True, remove=True, network=ordinary_name,
            name=f"browser-v1-sandbox-{suffix}", user=f"{os.getuid()}:{os.getgid()}",
            volumes={str(work_dir): {"bind": "/workspace", "mode": "rw"}},
            tmpfs={"/run/assist-browser":
                   "rw,nosuid,nodev,size=134217728,uid=10001,gid=10001,mode=0700"},
            security_opt=["seccomp=" + (Path(__file__).resolve().parents[1]
                            / "dockerfiles/browser-seccomp.json").read_text()])
        created.append(sandbox)
        sandbox.reload()
        sandbox_ip = sandbox.attrs["NetworkSettings"]["Networks"][ordinary_name]["IPAddress"]
        record_client(str(map_dir), sandbox_ip,
                      ClientRecord("thread", sandbox.id, "sandbox"))

        monkeypatch.setattr(turn_runtime, "EGRESS_NETWORK", ordinary_name)
        monkeypatch.setattr(turn_runtime, "BROWSER_NETWORK", browser_name)
        monkeypatch.setattr(turn_runtime, "EGRESS_PROXY_NAME", proxy_name)
        monkeypatch.setattr(turn_runtime, "_proxy_setup_lock", nullcontext)
        monkeypatch.setattr(turn_runtime.SandboxManager,
                            "_get_docker_client", classmethod(lambda _cls: client))
        monkeypatch.setattr(turn_runtime.runtime_state,
                            "assert_admissible", lambda *_args: None)
        monkeypatch.setattr(turn_runtime, "_check_lease", lambda _request: None)
        token = uuid4().hex
        request = {"generation": sandbox.id, "map_dir": str(map_dir),
                   "ip": sandbox_ip, "thread_id": "thread",
                   "threads_root": str(tmp_path), "work_dir": str(work_dir),
                   "run_id": "run", "token": token}
        assert turn_runtime.launch(request)["generation"] == sandbox.id

        def socket_attempt(user, host, port):
            result = sandbox.exec_run([
                "python", "-c", "import socket; "
                f"socket.create_connection(({host!r},{port}),2).close()"], user=user)
            return result.exit_code

        assert socket_attempt("10001:10001", ordinary_proxy_ip, 8889) == 0
        for proxy_ip in proxy_ips:
            assert socket_attempt(f"{os.getuid()}:{os.getgid()}", proxy_ip, 8889) != 0
        for host, port in ((ordinary_proxy_ip, 8888), (origin_ip, 8000),
                           ("127.0.0.1", 8000)):
            assert socket_attempt("10001:10001", host, port) != 0
        ipv6 = sandbox.exec_run([
            "python", "-c", "import socket; "
            "socket.create_connection(('::1',8000),2).close()"], user="10001:10001")
        assert ipv6.exit_code != 0
        udp = sandbox.exec_run([
            "python", "-c", "import socket; "
            "s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); "
            "s.settimeout(1); "
            f"s.sendto(b'x',({origin_ip!r},8000)); s.recv(1)"],
            user="10001:10001")
        assert udp.exit_code != 0
        assert socket_attempt(f"{os.getuid()}:{os.getgid()}", ordinary_proxy_ip, 8888) == 0

        command = json.dumps({"session": token, "operation": "open",
                              "args": {"url": f"http://{origin_ip}:8000/"}})
        result = browser._docker_exec(sandbox.id, command.encode(), in_sandbox=True)
        assert "Sandbox lifetime fixture" in json.loads(result)["result"]["snapshot"]
        processes = sandbox.top(ps_args="-eo pid,stat,comm")["Processes"]
        assert any("chrome" in row[-1].lower() and not row[1].startswith("Z")
                   for row in processes)
        # No host callback runs after launch. The test-only outer PID 1 exits;
        # Docker removes the sandbox and its nested Chromium descendants.
        _wait_absent(client, sandbox.id)
    finally:
        for container in reversed(created):
            try:
                container.remove(force=True)
            except docker.errors.NotFound:
                pass
        for network in reversed(networks):
            try:
                network.remove()
            except docker.errors.NotFound:
                pass
