"""Real Docker/proxy integration; build current images before opting in.

`make browser-docker-test` builds them and verifies the browser image contains
this checkout's runner, so stale image tags cannot supply false evidence.
"""
import hashlib
import os
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import pytest

from assist.browser import manager as browser
from assist.browser import authority
from assist.run_service import RunService
from assist.egress.client_map import ClientRecord, read_client, record_client
from assist.egress import runtime_state
from assist.sandbox_manager import (SandboxManager, _egress_proxy_config_hash,
                                    _network_identity, _bounded_egress_worker)


ORIGIN_SCRIPT = r'''
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlsplit
class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def do_GET(self):
        if self.path.startswith('/redirect?'):
            target = parse_qs(urlsplit(self.path).query)['to'][0]
            self.send_response(302)
            self.send_header('Location', target)
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        if self.path.startswith('/cross-port?'):
            host = parse_qs(urlsplit(self.path).query)['host'][0]
            body = (f'<html><img src="http://{host}:8001/pixel"></html>').encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith('/worker?'):
            host = parse_qs(urlsplit(self.path).query)['host'][0]
            body = (('<html><body><button id="state">pending</button><script>'
                     'const code = `let ws = new WebSocket("ws://' + host + ':8000/ws");'
                     'ws.onerror=()=>postMessage("blocked");'
                     'ws.onopen=()=>postMessage("opened");`;'
                     'const worker = new Worker(URL.createObjectURL('
                     'new Blob([code], {type:"application/javascript"})));'
                     'worker.onmessage=e=>document.querySelector("#state").textContent=e.data;'
                     '</script></body></html>').encode())
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith('/hostile?'):
            host = parse_qs(urlsplit(self.path).query)['host'][0]
            target = 'http://' + host + ':8000/post'
            body = (f'<html><body><button id="fetch" onclick="'
                    f'fetch(\'{target}\',{{method:\'POST\'}}).catch(()=>{{}})'
                    f'">Fetch internal</button>'
                    f'<button id="popup" onclick="window.open(\'{target}\')">'
                    f'Popup internal</button>'
                    f'<form action="{target}" method="POST">'
                    '<button type="submit">Post internal</button></form>'
                    '</body></html>').encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == '/download':
            body = b'Browser fixture download\n'
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Content-Disposition', 'attachment; filename="report.txt"')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == '/large':
            body = b'x' * (20 * 1024 * 1024 + 1)
            self.send_response(200)
            self.send_header('Content-Disposition', 'attachment; filename="large.txt"')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = (b'<html><body><h1>Browser fixture</h1>'
                b'<a href="/next" onclick="event.preventDefault();'
                b'document.querySelector(\'#status\').textContent=\'first\'">First</a>'
                b'<a href="/next" onclick="event.preventDefault();'
                b'document.querySelector(\'#status\').textContent=\'second\'">Second</a>'
                b'<a href="/download">Download report</a>'
                b'<a href="/large">Download large file</a>'
                b'<input aria-label="Search" type="text">'
                b'<input aria-label="Password" type="password">'
                b'<button onclick="document.querySelector(\'#status\').textContent='
                b'document.querySelector(\'input[aria-label=Search]\').value">Use search</button>'
                b'<button onclick="setTimeout(()=>document.querySelector(\'#later\')'
                b'.innerHTML=\'<button>Ready now</button>\',200)">Load later</button>'
                b'<button onclick="window.open(\'about:blank\')">Open popup</button>'
                b'<div id="later"></div>'
                b'<p id="status">ready</p></body></html>')
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
HTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
'''

TEST_PROXY_SCRIPT = r'''
import importlib.util, os
spec = importlib.util.spec_from_file_location('proxy', '/usr/local/bin/egress-proxy.py')
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)
original = proxy.vet_resolved
def fixture_resolve(host, port, *, global_only=True):
    if host in ('fixture-public.test', 'fixture-public-forms.test') and global_only:
        return os.environ['BROWSER_TEST_PUBLIC_ORIGIN_IP']
    return original(host, port, global_only=global_only)
proxy.vet_resolved = fixture_resolve
# Fixture pages intentionally make many rapid origin requests. Keep the
# production target policy, but do not let unrelated host throttling mask
# redirect/subresource port-denial assertions (throttle has unit coverage).
proxy._UNTHROTTLED_HOSTS = proxy._UNTHROTTLED_HOSTS | {
    os.environ['BROWSER_TEST_PUBLIC_ORIGIN_IP']}
proxy.main()
'''


@pytest.mark.skipif(os.getenv("ASSIST_BROWSER_DOCKER_TEST") != "1",
                    reason="requires Docker Engine 29 isolated bridges and proxy image")
def test_killable_proxy_worker_recreates_only_exact_generation(tmp_path, monkeypatch):
    import docker

    client = docker.from_env()
    suffix = uuid4().hex[:10]
    ordinary_name = f"assist-browser-worker-ordinary-{suffix}"
    browser_name = f"assist-browser-worker-isolated-{suffix}"
    proxy_name = f"assist-browser-worker-proxy-{suffix}"
    threads_root = tmp_path / "threads"
    threads_root.mkdir()
    map_dir = tmp_path / "map"
    map_dir.mkdir(mode=0o700)
    allowlist = tmp_path / "allowlist.conf"
    allowlist.write_text("example.com\n")
    monkeypatch.setenv("ASSIST_THREADS_DIR", str(threads_root))
    monkeypatch.setenv("ASSIST_EGRESS_CLIENT_MAP_DIR", str(map_dir))
    monkeypatch.setenv("ASSIST_EGRESS_RUNTIME_DIR", str(tmp_path / "runtime"))
    script = (
        "import assist.sandbox_manager as s, "
        "assist.egress.runtime_worker as w; "
        f"s.EGRESS_NETWORK={ordinary_name!r}; "
        f"s.BROWSER_NETWORK={browser_name!r}; "
        f"s.EGRESS_PROXY_NAME={proxy_name!r}; "
        "s.EGRESS_PROXY_IMAGE='assist-egress-proxy'; "
        f"s.EGRESS_ALLOWLIST_FILE={str(allowlist)!r}; "
        "raise SystemExit(w.main())")

    def start():
        result = _bounded_egress_worker(
            [sys.executable, "-c", script, "proxy"], timeout=20)
        assert result.returncode == 0, result.stderr[-1000:]
        assert result.stdout.strip() == proxy_name.encode()
        return client.containers.get(proxy_name)

    try:
        first = start()
        assert start().id == first.id
        allowlist.write_text("example.com\npypi.org\n")
        second = start()
        assert second.id != first.id
        assert runtime_state.retirement("proxy", first.id) == "remove"
        attached = second.attrs["NetworkSettings"]["Networks"]
        assert attached[ordinary_name]["NetworkID"] == client.networks.get(ordinary_name).id
        assert attached[browser_name]["NetworkID"] == client.networks.get(browser_name).id
    finally:
        try:
            client.containers.get(proxy_name).remove(force=True)
        except docker.errors.NotFound:
            pass
        for name in (browser_name, ordinary_name):
            try:
                client.networks.get(name).remove()
            except docker.errors.NotFound:
                pass


@pytest.mark.skipif(os.getenv("ASSIST_BROWSER_DOCKER_TEST") != "1",
                    reason="requires Docker Engine 29 isolated bridges")
def test_recreated_browser_network_changes_proxy_identity():
    import docker

    client = docker.from_env()
    name = f"assist-browser-recreation-test-{uuid4().hex[:10]}"
    network = None
    try:
        first = client.networks.create(
            name, driver="bridge", internal=True, enable_ipv6=False,
            options={"com.docker.network.bridge.gateway_mode_ipv4": "isolated"})
        network = first
        first_id, first_cidr = _network_identity(first, isolated=True)
        first.remove()
        network = None
        second = client.networks.create(
            name, driver="bridge", internal=True, enable_ipv6=False,
            options={"com.docker.network.bridge.gateway_mode_ipv4": "isolated"})
        network = second
        second_id, second_cidr = _network_identity(second, isolated=True)
        assert first_id != second_id
        assert _egress_proxy_config_hash(
            "example.com", None, f"ordinary|{first_id}:{first_cidr}") != (
                _egress_proxy_config_hash(
                    "example.com", None, f"ordinary|{second_id}:{second_cidr}"))
    finally:
        if network is not None:
            network.remove()


@pytest.mark.skipif(os.getenv("ASSIST_BROWSER_DOCKER_TEST") != "1",
                    reason="requires locally built browser/proxy test images and Docker")
def test_isolated_browser_reaches_only_attributed_proxy(tmp_path, monkeypatch):
    import docker

    client = docker.from_env()
    expected_runner = hashlib.sha256((Path(__file__).resolve().parents[1]
                                      / "assist/browser/runner.py").read_bytes()).hexdigest()
    image_runner = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "--entrypoint", "python",
         "assist-browser", "-c", "import hashlib; print(hashlib.sha256("
         "open('/opt/assist/browser_runner.py','rb').read()).hexdigest())"],
        capture_output=True, check=True, timeout=10).stdout.decode().strip()
    assert image_runner == expected_runner, "browser image is stale; run make browser-smoke"
    suffix = uuid4().hex[:10]
    ordinary_name = f"assist-browser-test-ordinary-{suffix}"
    browser_name = f"assist-browser-test-isolated-{suffix}"
    proxy_name = f"assist-browser-test-proxy-{suffix}"
    created = []
    networks = []
    session = None
    try:
        ordinary = client.networks.create(ordinary_name, driver="bridge", internal=True)
        networks.append(ordinary)
        isolated = client.networks.create(
            browser_name, driver="bridge", internal=True, enable_ipv6=False,
            options={"com.docker.network.bridge.gateway_mode_ipv4": "isolated"})
        networks.append(isolated)
        ordinary.reload()
        isolated.reload()
        ordinary_cidr = ordinary.attrs["IPAM"]["Config"][0]["Subnet"]
        browser_cidr = isolated.attrs["IPAM"]["Config"][0]["Subnet"]
        origin = client.containers.run(
            "python:3.13-slim-bookworm", ["python", "-u", "-c", ORIGIN_SCRIPT],
            detach=True, network=ordinary_name, remove=True,
            name=f"assist-browser-test-origin-{suffix}")
        created.append(origin)
        origin.reload()
        origin_ip = origin.attrs["NetworkSettings"]["Networks"][ordinary_name]["IPAddress"]
        map_dir = tmp_path / "client-map"
        map_dir.mkdir(mode=0o700)
        monkeypatch.setenv("ASSIST_EGRESS_CLIENT_MAP_DIR", str(map_dir))
        proxy = client.containers.run(
            "assist-egress-proxy", detach=True, remove=True,
            name=proxy_name,
            entrypoint=["python3", "-c", TEST_PROXY_SCRIPT],
            volumes={str(map_dir): {"bind": "/client-map", "mode": "ro"}},
            environment={"EGRESS_ALLOWLIST": (
                             f"{origin_ip},example.com,fixture-public.test,"
                             "fixture-public-forms.test"),
                         "EGRESS_SANDBOX_CIDR": ordinary_cidr,
                         "EGRESS_BROWSER_CIDR": browser_cidr,
                         "BROWSER_TEST_PUBLIC_ORIGIN_IP": origin_ip})
        created.append(proxy)
        ordinary.connect(proxy)
        isolated.connect(proxy, aliases=["assist-egress-proxy"])
        SandboxManager._wait_for_egress_proxy_ready(proxy)
        # With approval grants disabled, an ordinary shell source retains
        # base-only access. The first admitted connection starts host throttle.
        for host, status in [(origin_ip, b"200"), ("unlisted.example", b"403")]:
            raw = origin.exec_run([
                "python", "-c", "import socket,sys; "
                f"s=socket.create_connection(('{proxy_name}',8888),3); "
                f"s.sendall(b'CONNECT {host}:8000 HTTP/1.1\\r\\n\\r\\n'); "
                "sys.stdout.buffer.write(s.recv(256))"])
            assert status in raw.output.split(b"\r\n", 1)[0]
        time.sleep(2.1)
        monkeypatch.setattr(browser, "BROWSER_NETWORK", browser_name)
        monkeypatch.setattr(browser, "EGRESS_PROXY_NAME", proxy_name)
        monkeypatch.setattr(browser, "BROWSER_IMAGE", "assist-browser")
        monkeypatch.setattr(browser, "_load_egress_allowlist",
                            lambda: [origin_ip, "example.com", "fixture-public.test",
                                     "fixture-public-forms.test"])
        script = (
            "import assist.browser.manager as b, assist.browser.startup_worker as w; "
            f"b.BROWSER_NETWORK={browser_name!r}; "
            f"b.EGRESS_PROXY_NAME={proxy_name!r}; "
            "b.BROWSER_IMAGE='assist-browser'; "
            "w._launch_sidecar_direct=lambda request: "
            "b._launch_sidecar_direct(request,ensure_proxy=False); "
            "raise SystemExit(w.main())")

        def start_in_fixture(request, timeout):
            result = browser._bounded_cli(
                [sys.executable, "-c", script],
                payload=json.dumps(request).encode(), limit=4096, timeout=timeout)
            return json.loads(result)

        monkeypatch.setattr(browser, "_launch_sidecar_bounded", start_in_fixture)
        work_dir = tmp_path / "workspace"
        work_dir.mkdir()
        (tmp_path / "test-thread").mkdir()
        authority.mark_new_thread(str(tmp_path), "test-thread")
        runs = RunService(str(tmp_path))
        internal_run = runs.create(
            "test-thread", "general-agent",
            f"Please visit http://{origin_ip}:8000 and inspect it.",
            user_origin=True)
        session = browser.BrowserSession(
            "test-thread", internal_run.id, str(work_dir), str(tmp_path),
            browser.BrowserUserRequest(
                internal_run.id, internal_run.work_id, internal_run.text,
                internal_run.admission_sequence),
            work_id=internal_run.work_id)
        # The browser network's live Docker settings, not its name, are the gate.
        initial = session.command("open", url=f"http://{origin_ip}:8000/")
        assert "error" not in initial, initial
        sidecar = client.containers.get(session.identity.generation)
        # Consent for :8000 never grants another port on the same host,
        # including an agent open, proxy request, redirect or subresource.
        with pytest.raises(browser.BrowserUnavailable, match="host and port"):
            session.command("open", url=f"http://{origin_ip}:8001/")
        for request in [
                f"CONNECT {origin_ip}:8001 HTTP/1.1\r\n\r\n",
                (f"GET http://{origin_ip}:8001/pixel HTTP/1.1\r\n"
                 f"Host: {origin_ip}:8001\r\n\r\n")]:
            raw = sidecar.exec_run([
                "python", "-c", "import socket,sys; "
                "s=socket.create_connection(('assist-egress-proxy',8888),3); "
                f"s.sendall({request!r}.encode()); "
                "sys.stdout.buffer.write(s.recv(512))"])
            assert b"browser_internal_policy" in raw.output
        denied_cross_port = proxy.logs().count(
            f"DENY {origin_ip} (browser_internal_policy)".encode())
        redirected_cross_port = session.command(
            "open", url=(f"http://{origin_ip}:8000/redirect?"
                         f"to=http%3A%2F%2F{origin_ip}%3A8001%2F"))
        assert "error" not in redirected_cross_port, redirected_cross_port
        denied_after_redirect = proxy.logs().count(
            f"DENY {origin_ip} (browser_internal_policy)".encode())
        assert denied_after_redirect > denied_cross_port, (
            redirected_cross_port, proxy.logs()[-1200:])
        subresource = session.command(
            "open", url=f"http://{origin_ip}:8000/cross-port?host={origin_ip}")
        assert "error" not in subresource, subresource
        assert proxy.logs().count(
            f"DENY {origin_ip} (browser_internal_policy)".encode()) > denied_after_redirect, (
                subresource, proxy.logs()[-1200:])
        observation = initial["result"]
        assert "Browser fixture" in observation["snapshot"]
        assert read_client(str(map_dir), session.identity.ip).generation == session.identity.generation
        links = [target for target in observation["targets"]
                 if target.get("href") == f"http://{origin_ip}:8000/next"]
        assert len(links) == 2, repr(observation["network_errors"])
        ambiguous = session.command(
            "act", page_id=observation["page_id"],
            snapshot_id=observation["snapshot_id"], action="click",
            target={"href": links[0]["href"]})
        assert "ambiguous" in ambiguous["error"]
        clicked = session.command(
            "act", page_id=observation["page_id"],
            snapshot_id=observation["snapshot_id"], action="click",
            target={"ref": links[1]["ref"]})
        assert "second" in clicked["result"]["snapshot"]
        stale = session.command(
            "act", page_id=observation["page_id"],
            snapshot_id=observation["snapshot_id"], action="click",
            target={"ref": links[0]["ref"]})
        assert "stale" in stale["error"]
        observed = clicked["result"]
        search = next(target for target in observed["targets"]
                      if target["name"] == "Search")
        filled = session.command(
            "act", page_id=observed["page_id"],
            snapshot_id=observed["snapshot_id"], action="fill",
            target={"ref": search["ref"], "text": "needle"})["result"]
        password = next(target for target in filled["targets"]
                        if target["name"] == "Password")
        denied_secret = session.command(
            "act", page_id=filled["page_id"],
            snapshot_id=filled["snapshot_id"], action="fill",
            target={"ref": password["ref"], "text": "secret"})
        assert "secret" in denied_secret["error"]
        use_search = next(target for target in filled["targets"]
                          if target["name"] == "Use search")
        searched = session.command(
            "act", page_id=filled["page_id"],
            snapshot_id=filled["snapshot_id"], action="click",
            target={"ref": use_search["ref"]})["result"]
        assert "needle" in searched["snapshot"]
        delayed = next(target for target in searched["targets"]
                       if target["name"] == "Load later")
        loading = session.command(
            "act", page_id=searched["page_id"],
            snapshot_id=searched["snapshot_id"], action="click",
            target={"ref": delayed["ref"]})["result"]
        waited = session.command(
            "wait", page_id=loading["page_id"], role="button", name="Ready now")
        assert "Ready now" in waited["result"]["snapshot"]
        popup_button = next(target for target in waited["result"]["targets"]
                            if target["name"] == "Open popup")
        popup = session.command(
            "act", page_id=waited["result"]["page_id"],
            snapshot_id=waited["result"]["snapshot_id"], action="click",
            target={"ref": popup_button["ref"]})["result"]
        assert popup["page_id"] != observation["page_id"]
        assert popup["url"] == "about:blank"
        time.sleep(4.1)
        observed = session.command("observe", page_id=observation["page_id"])["result"]
        report = next(target for target in observed["targets"]
                      if target.get("href") == f"http://{origin_ip}:8000/download")
        downloaded = session.command(
            "act", page_id=observed["page_id"],
            snapshot_id=observed["snapshot_id"], action="click",
            target={"ref": report["ref"]})["result"]
        assert downloaded["downloads"]
        download_id = downloaded["downloads"][-1]["download_id"]
        assert session.save_download(download_id) == {
            "path": "/workspace/downloads/report.txt", "bytes": 25}
        assert (work_dir / "downloads" / "report.txt").read_bytes() == (
            b"Browser fixture download\n")
        with pytest.raises(FileExistsError):
            session.save_download(download_id)
        with pytest.raises(browser.BrowserUnavailable, match="safe basename"):
            session.save_download(download_id, "AGENTS.md")
        large_page = session.command("observe", page_id=observed["page_id"])["result"]
        time.sleep(8.1)
        large_link = next(target for target in large_page["targets"]
                          if target.get("href") == f"http://{origin_ip}:8000/large")
        large = session.command(
            "act", page_id=large_page["page_id"],
            snapshot_id=large_page["snapshot_id"], action="click",
            target={"ref": large_link["ref"]})["result"]
        assert "20 MiB" in session.save_download(
            large["downloads"][-1]["download_id"])["error"]
        sidecar.reload()
        attrs = sidecar.attrs
        assert attrs["HostConfig"]["Memory"] == 1024 ** 3
        assert attrs["HostConfig"]["PidsLimit"] == 128
        assert attrs["HostConfig"]["NanoCpus"] == 1_000_000_000
        assert attrs["HostConfig"]["ReadonlyRootfs"] is True
        assert attrs["HostConfig"]["AutoRemove"] is True
        assert attrs["HostConfig"]["RestartPolicy"]["Name"] in {"", "no"}
        assert attrs["Mounts"] == []
        assert not any(item.startswith("ASSIST_") for item in attrs["Config"]["Env"])
        assert "size=134217728" in attrs["HostConfig"]["Tmpfs"]["/home/browser"]
        # An exec from the host can inspect routing; the product has no such tool.
        direct = sidecar.exec_run([
            "python", "-c", "import socket; s=socket.socket(); s.settimeout(2); "
            f"s.connect(('{origin_ip}',8000))"])
        assert direct.exit_code != 0
        direct_v6 = sidecar.exec_run([
            "python", "-c", "import socket; "
            "s=socket.socket(socket.AF_INET6,socket.SOCK_STREAM); "
            "s.settimeout(2); s.connect(('2001:4860:4860::8888',53))"])
        assert direct_v6.exit_code != 0
        for family, destination in [
                ("AF_INET", f"('{origin_ip}',8000)"),
                ("AF_INET6", "('2001:4860:4860::8888',53)")]:
            direct = sidecar.exec_run([
                "python", "-c", "import socket; "
                f"s=socket.socket(socket.{family},socket.SOCK_DGRAM); "
                f"s.settimeout(2); s.connect({destination})"])
            assert direct.exit_code != 0
        public_run = runs.create(
            "test-thread", "general-agent", "Read a public page", user_origin=True)
        public_only = browser.BrowserSession(
            "test-thread", public_run.id, str(work_dir), str(tmp_path), None)
        with pytest.raises(browser.BrowserUnavailable, match="revoked internal browsing"):
            public_only.command("open", url=f"http://{origin_ip}:8000/")
        session.close()
        session = browser.BrowserSession(
            "test-thread", public_run.id, str(work_dir), str(tmp_path), None)
        denied_http = session.command("open", url="http://unlisted.example:8000/")
        sidecar = client.containers.get(session.identity.generation)
        assert "error" not in denied_http, denied_http
        denied_probe = session.command("probe", host="unlisted.example", port=8000)
        assert "result" in denied_probe, (denied_probe,
                                          denied_http["result"]["url"],
                                          denied_http["result"]["network_errors"])
        assert denied_probe["result"]["reason"] == "host_not_approved", denied_probe
        # The same proxy gate handles a redirect CONNECT and a worker's
        # absolute-URL WebSocket upgrade; neither can use an internal base host
        # from a public sidecar.
        for request in [
                f"CONNECT {origin_ip}:8000 HTTP/1.1\r\n\r\n",
                (f"GET http://{origin_ip}:8000/ws HTTP/1.1\r\n"
                 f"Host: {origin_ip}:8000\r\nUpgrade: websocket\r\n\r\n")]:
            raw = sidecar.exec_run([
                "python", "-c", "import socket,sys; "
                "s=socket.create_connection(('assist-egress-proxy',8888),3); "
                f"s.sendall({request!r}.encode()); "
                "sys.stdout.buffer.write(s.recv(512))"])
            assert b"403 Forbidden" in raw.output
            assert b"browser_internal_policy" in raw.output
        denied_before = proxy.logs().count(
            f"DENY {origin_ip} (browser_internal_policy)".encode())
        redirected = session.command(
            "open", url=(f"http://fixture-public.test:8000/redirect?"
                         f"to=http%3A%2F%2F{origin_ip}%3A8000%2F"))
        assert "error" not in redirected, redirected
        denied_after_redirect = proxy.logs().count(
            f"DENY {origin_ip} (browser_internal_policy)".encode())
        assert denied_after_redirect > denied_before, (
            redirected, proxy.logs()[-1000:])
        time.sleep(2.1)
        worker = session.command(
            "open", url=f"http://fixture-public.test:8000/worker?host={origin_ip}")
        assert "error" not in worker, worker
        worker_page = worker["result"]["page_id"]
        worker_result = session.command(
            "wait", page_id=worker_page, role="button", name="blocked")
        assert "error" not in worker_result, worker_result
        denied_after_worker = proxy.logs().count(
            f"DENY {origin_ip} (browser_internal_policy)".encode())
        assert denied_after_worker > denied_after_redirect
        hostile = session.command(
            "open", url=f"http://fixture-public-forms.test:8000/hostile?host={origin_ip}")
        assert "error" not in hostile, hostile
        assert "Fetch internal" in hostile["result"]["snapshot"], hostile
        hostile_page = hostile["result"]["page_id"]
        denied_before_action = proxy.logs().count(
            f"DENY {origin_ip} (browser_internal_policy)".encode())
        for button_name in ("Fetch internal", "Popup internal", "Post internal"):
            current = session.command("observe", page_id=hostile_page)["result"]
            button = next((item for item in current["targets"]
                           if item["name"] == button_name), None)
            assert button is not None, (button_name, current)
            acted = session.command(
                "act", page_id=hostile_page,
                snapshot_id=current["snapshot_id"], action="click",
                target={"ref": button["ref"]})
            assert "error" not in acted, (button_name, acted)
            denied_after_action = proxy.logs().count(
                f"DENY {origin_ip} (browser_internal_policy)".encode())
            assert denied_after_action > denied_before_action, (
                button_name, acted, proxy.logs()[-1000:])
            denied_before_action = denied_after_action
        # A stopped but inspectable browser object cannot retain an internal
        # grant at an address later recycled by another container.
        old = client.containers.run(
            "python:3.13-slim-bookworm", ["python", "-c", "import time;time.sleep(60)"],
            detach=True, network=browser_name, remove=False,
            labels={"assist.browser": "true"},
            name=f"assist-browser-test-stopped-{suffix}")
        created.append(old)
        old.reload()
        old_ip = old.attrs["NetworkSettings"]["Networks"][browser_name]["IPAddress"]
        record_client(str(map_dir), old_ip, ClientRecord(
            "test-thread", old.id, "browser", "internal", origin_ip, 8000))
        old.stop(timeout=3)
        old.reload()
        assert old.status == "exited"
        browser.BrowserManager.prune_absent_clients(str(map_dir))
        assert read_client(str(map_dir), old_ip) is None
        old.start()
        restarted = old.exec_run([
            "python", "-c", "import socket,sys; "
            "s=socket.create_connection(('assist-egress-proxy',8888),3); "
            f"s.sendall(b'CONNECT {origin_ip}:8000 HTTP/1.1\\r\\n\\r\\n'); "
            "sys.stdout.buffer.write(s.recv(512))"])
        assert b"browser_attribution_missing" in restarted.output
        old.remove(force=True)
        recycled_id = subprocess.run([
            "docker", "run", "-d", "--name", f"assist-browser-test-recycled-{suffix}",
            "--network", browser_name, "--ip", old_ip,
            "python:3.13-slim-bookworm", "python", "-c",
            "import time;time.sleep(60)"],
            capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        recycled = client.containers.get(recycled_id)
        created.append(recycled)
        recycled.reload()
        assert recycled.attrs["NetworkSettings"]["Networks"][browser_name]["IPAddress"] == old_ip
        denied_reuse = recycled.exec_run([
            "python", "-c", "import socket,sys; "
            "s=socket.create_connection(('assist-egress-proxy',8888),3); "
            f"s.sendall(b'CONNECT {origin_ip}:8000 HTTP/1.1\\r\\n\\r\\n'); "
            "sys.stdout.buffer.write(s.recv(512))"])
        assert b"browser_attribution_missing" in denied_reuse.output
        unregistered = client.containers.run(
            "python:3.13-slim-bookworm", ["python", "-c", "import time;time.sleep(60)"],
            detach=True, network=browser_name, remove=True,
            name=f"assist-browser-test-unregistered-{suffix}")
        created.append(unregistered)
        raw = unregistered.exec_run([
            "python", "-c", "import socket,sys; "
            "s=socket.create_connection(('assist-egress-proxy',8888),3); "
            f"s.sendall(b'CONNECT {origin_ip}:8000 HTTP/1.1\\r\\n\\r\\n'); "
            "sys.stdout.buffer.write(s.recv(512))"])
        assert b"browser_attribution_missing" in raw.output
        if os.getenv("ASSIST_BROWSER_PUBLIC_TEST") == "1":
            public_page = session.command("open", url="https://example.com/")
            assert "error" not in public_page, public_page
            assert "Example Domain" in public_page["result"]["snapshot"]
    finally:
        if session is not None:
            session.close()
        for container in reversed(created):
            try:
                container.remove(force=True)
            except Exception:
                pass
        for network in reversed(networks):
            try:
                network.remove()
            except Exception:
                pass
