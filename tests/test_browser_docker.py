"""Real Docker/proxy checks using explicit isolated image tags."""
import hashlib
import os
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    events = []
    def do_GET(self):
        if self.path == '/events':
            body = '\n'.join(self.events).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == '/post':
            authenticated = 'session=synthetic-cookie-canary' in self.headers.get('Cookie', '')
            self.events.append('popup-get-auth=' + str(authenticated))
            body = (b'<h1>Authenticated popup fixture</h1>' if authenticated
                    else b'<h1>Denied popup fixture</h1>')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
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
        if self.path == '/login':
            body = (b'<html><h1>Login fixture</h1><script>'
                    b"localStorage.setItem('resume','synthetic-local-canary')"
                    b'</script></html>')
            self.send_response(200)
            self.send_header('Set-Cookie', 'session=synthetic-cookie-canary; Path=/')
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == '/account':
            authenticated = 'session=synthetic-cookie-canary' in self.headers.get('Cookie', '')
            heading = 'Authenticated fixture' if authenticated else 'Denied fixture'
            body = (f'<html><h1>{heading}</h1><p id="resume"></p>'
                    '<script>document.querySelector("#resume").textContent='
                    'localStorage.getItem("resume") || "missing"</script></html>').encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
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
    def do_POST(self):
        authenticated = 'session=synthetic-cookie-canary' in self.headers.get('Cookie', '')
        self.events.append('post-auth=' + str(authenticated))
        body = b'<h1>Private POST reached</h1>'
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
ThreadingHTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
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
        f"s.EGRESS_PROXY_IMAGE={os.environ['ASSIST_BROWSER_TEST_PROXY_IMAGE']!r}; "
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
                    reason="requires isolated tagged sandbox/proxy images and Docker")
def test_turn_sandbox_restores_private_storage_in_new_generation(tmp_path, monkeypatch):
    """The real worker restores cookie and localStorage after exact turn teardown."""
    import docker
    from assist import sandbox_manager as sandbox_module
    from assist.browser import storage
    from assist.egress.client_map import configured_directory

    sandbox_image = os.environ["ASSIST_BROWSER_TEST_SANDBOX_IMAGE"]
    proxy_image = os.environ["ASSIST_BROWSER_TEST_PROXY_IMAGE"]
    client = docker.from_env()
    checkout = Path(__file__).resolve().parents[1]
    for source, installed in (
            ("assist/browser/runner.py", "/opt/assist/browser_runner.py"),
            ("dockerfiles/install-browser-fence.py", "/opt/assist/install-browser-fence.py"),
            ("dockerfiles/launch-turn-browser.py", "/opt/assist/launch-turn-browser.py")):
        expected = hashlib.sha256((checkout / source).read_bytes()).hexdigest()
        actual = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", "--entrypoint",
             "/usr/bin/python", sandbox_image, "-c",
             "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())",
             installed], capture_output=True, check=True, timeout=15).stdout.decode().strip()
        assert actual == expected, f"test sandbox image has stale {source}"
    expected_proxy = hashlib.sha256((checkout / "dockerfiles/egress-proxy.py").read_bytes()).hexdigest()
    actual_proxy = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "--entrypoint", "python3",
         proxy_image, "-c",
         "import hashlib;print(hashlib.sha256(open('/usr/local/bin/egress-proxy.py','rb').read()).hexdigest())"],
        capture_output=True, check=True, timeout=15).stdout.decode().strip()
    assert actual_proxy == expected_proxy, "test proxy image has stale policy"
    suffix = uuid4().hex[:10]
    ordinary_name = f"assist-browser-test-ordinary-{suffix}"
    browser_name = f"assist-browser-test-isolated-{suffix}"
    proxy_name = f"assist-browser-test-proxy-{suffix}"
    networks = []
    created = []
    sessions = []
    work_dir = tmp_path / "threads" / "t" / "domain"
    work_dir.mkdir(parents=True)
    (tmp_path / "map").mkdir(mode=0o700)
    monkeypatch.setenv("ASSIST_EGRESS_CLIENT_MAP_DIR", str(tmp_path / "map"))
    monkeypatch.setenv("ASSIST_EGRESS_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("ASSIST_THREADS_DIR", str(tmp_path / "threads"))
    assert configured_directory() == str(tmp_path / "map")
    authority.mark_new_thread(str(tmp_path / "threads"), "t")
    runs = RunService(str(tmp_path / "threads"))
    original_cli = browser._bounded_cli

    def isolated_cli(argv, **kwargs):
        if argv[-2:] == ["-m", "assist.browser.turn_runtime"]:
            code = (
                "import assist.browser.turn_runtime as runtime; "
                f"runtime.EGRESS_NETWORK={ordinary_name!r}; "
                f"runtime.BROWSER_NETWORK={browser_name!r}; "
                f"runtime.EGRESS_PROXY_NAME={proxy_name!r}; "
                "runtime.main()")
            argv = [sys.executable, "-c", code]
        return original_cli(argv, **kwargs)

    def record_client_direct(cls, docker_client, container, path, *, kind):
        cls._egress_client_ips[path] = cls._record_egress_client_direct(
            docker_client, container, path, kind=kind)

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
            proxy_image, ["python3", "-u", "-c", ORIGIN_SCRIPT],
            detach=True, network=ordinary_name, remove=True,
            name=f"assist-browser-test-origin-{suffix}")
        created.append(origin)
        origin.reload()
        origin_ip = origin.attrs["NetworkSettings"]["Networks"][ordinary_name]["IPAddress"]
        proxy = client.containers.run(
            proxy_image, detach=True, remove=True, name=proxy_name,
            entrypoint=["python3", "-c", TEST_PROXY_SCRIPT],
            volumes={str(tmp_path / "map"): {"bind": "/client-map", "mode": "ro"}},
            environment={"EGRESS_ALLOWLIST": f"{origin_ip},fixture-public.test",
                         "EGRESS_SANDBOX_CIDR": ordinary_cidr,
                         "EGRESS_BROWSER_CIDR": browser_cidr,
                         "BROWSER_TEST_PUBLIC_ORIGIN_IP": origin_ip},
            extra_hosts={"fixture-public.test": origin_ip})
        created.append(proxy)
        ordinary.connect(proxy)
        isolated.connect(proxy, aliases=[proxy_name])
        SandboxManager._wait_for_egress_proxy_ready(proxy)
        monkeypatch.setattr(sandbox_module, "SANDBOX_IMAGE", sandbox_image)
        monkeypatch.setattr(sandbox_module, "EGRESS_NETWORK", ordinary_name)
        monkeypatch.setattr(sandbox_module, "BROWSER_NETWORK", browser_name)
        monkeypatch.setattr(sandbox_module, "EGRESS_PROXY_NAME", proxy_name)
        monkeypatch.setattr(SandboxManager, "_ensure_egress_proxy_running",
                            classmethod(lambda _cls, _client: proxy_name))
        monkeypatch.setattr(SandboxManager, "_record_egress_client",
                            classmethod(record_client_direct))
        monkeypatch.setattr(browser, "_bounded_cli", isolated_cli)
        for turn in (1, 2):
            run = runs.create("t", "general-agent", "Visit the fixture", user_origin=True)
            backend = SandboxManager.get_sandbox_backend(
                str(work_dir), browser_capable=True,
                thread_scope=(str(tmp_path / "threads"), "t"), owner_run_id=run.id)
            assert backend is not None
            container = SandboxManager.current_container(str(work_dir))
            assert container is not None
            session = browser.BrowserSession(
                "t", run.id, str(work_dir), str(tmp_path / "threads"),
                browser.BrowserUserRequest(
                    run.id, run.work_id, run.text, run.admission_sequence),
                work_id=run.work_id, sandbox_generation=container.id, run_service=runs)
            sessions.append(session)
            path = "/login" if turn == 1 else "/account"
            result = session.command("open", url=f"http://{origin_ip}:8000{path}")
            assert "error" not in result, result
            if turn == 1:
                old_generation = container.id
                old_page_id = result["result"]["page_id"]
                assert "Login fixture" in result["result"]["snapshot"]
                session.close()
                SandboxManager.cleanup(str(work_dir), container)
                state_file = tmp_path / "threads" / "t" / storage.FILE
                assert state_file.stat().st_mode & 0o777 == 0o600
                assert state_file.stat().st_size <= storage.MAX_BYTES
                assert all(mount.get("Source") != str(state_file.parent)
                           for mount in container.attrs["Mounts"])
                with pytest.raises(docker.errors.NotFound):
                    client.containers.get(old_generation)
            else:
                assert container.id != old_generation
                assert "Authenticated fixture" in result["result"]["snapshot"]
                assert "synthetic-local-canary" in result["result"]["snapshot"]
                assert result["result"]["page_id"] != old_page_id
                stale = session.command("observe", page_id=old_page_id)
                assert "unknown or closed page ID" in stale["error"]
                hostile = session.command(
                    "open", url=("http://fixture-public.test:8000/hostile?host="
                                 + origin_ip))["result"]
                public_page = hostile["page_id"]
                fetched = session.command(
                    "act", page_id=public_page,
                    snapshot_id=hostile["snapshot_id"], action="click",
                    target={"role": "button", "name": "Fetch internal"})["result"]
                popup = session.command(
                    "act", page_id=public_page,
                    snapshot_id=fetched["snapshot_id"], action="click",
                    target={"role": "button", "name": "Popup internal"})["result"]
                assert "Authenticated popup fixture" in popup["snapshot"]
                refreshed = session.command("observe", page_id=public_page)["result"]
                submitted = session.command(
                    "act", page_id=public_page,
                    snapshot_id=refreshed["snapshot_id"], action="click",
                    target={"role": "button", "name": "Post internal"})["result"]
                assert "Private POST reached" in submitted["snapshot"]
                history = origin.exec_run(
                    ["python", "-c", "import urllib.request;print(urllib.request.urlopen("
                     "'http://127.0.0.1:8000/events').read().decode())"])[1].decode()
                assert history.count("post-auth=") >= 2
                assert "popup-get-auth=True" in history
                session.close()
                SandboxManager.cleanup(str(work_dir), container)
    finally:
        for session in sessions:
            if not session.closed:
                try:
                    session.close()
                except Exception:
                    pass
        try:
            SandboxManager.cleanup(str(work_dir))
        except Exception:
            pass
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
