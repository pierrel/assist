#!/usr/bin/env python3
from __future__ import annotations
import atexit
import base64
import errno
import fcntl
import hashlib
import http.client
import json
import os
import select
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parents[1]
PROMPT = "ASSIST_P0_PROMPT_OWNERSHIP_CANARY_v1"
USER = "Return the fixed deterministic response."
FIFO = "/run/assist-p0/liveness.fifo"
ENV_CANARY = "P0_AUTHORITY_ENV_MUST_NOT_ENTER_RUNNER"
NODE_VERSION = "v22.23.1"
LOCK_SHA256 = "ed4d145d52056516ddbc8dd602f74f6f6d363ac62acef73f26679acd9336baa7"
MAX_FRAME = 512 * 1024
MAX_SESSION = 2 * 1024 * 1024
WORK_SECONDS = 15.0
CLEANUP_SECONDS = 6.0
SCENARIOS = ("success", "fail-closed", "fail-backpressure", "hostile", "cancel", "fresh", "authority-death")

class P0Error(RuntimeError):
    pass

class Deadline:
    def __init__(self, seconds: float):
        self.end = time.monotonic() + seconds
    def remaining(self, label: str = "operation") -> float:
        value = self.end - time.monotonic()
        if value <= 0:
            raise P0Error(f"{label} exceeded its absolute deadline")
        return value

def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def command(argv: list[str], *, timeout: float, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True, **kwargs,
        )
        try:
            returncode = process.wait(timeout=timeout)
        except BaseException as caught:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired as reap:
                raise P0Error(f"command could not be reaped after timeout: {argv!r}") from reap
            output.seek(max(0, output.seek(0, os.SEEK_END) - 65536))
            raise P0Error(f"command interrupted {argv!r}: {output.read().decode(errors='replace')}") from caught
        output.seek(max(0, output.seek(0, os.SEEK_END) - 65536))
        result = subprocess.CompletedProcess(argv, returncode, output.read())
    if result.returncode:
        detail = result.stdout[-65536:].decode(errors="replace")
        raise P0Error(f"command failed {argv!r}:\n{detail}")
    return result

def tool(name: str) -> Path:
    value = shutil.which(name)
    if value is None:
        raise P0Error(f"missing tool: {name}")
    return Path(value).resolve()

def verify_lock() -> dict[str, object]:
    raw = (ROOT / "package-lock.json").read_bytes()
    if digest(raw) != LOCK_SHA256:
        raise P0Error("package lock differs from the reviewed P0 identity")
    lock = json.loads(raw)
    bad = [
        name for name, meta in lock["packages"].items()
        if name and not meta.get("link") and (
            not str(meta.get("resolved", "")).startswith("https://registry.npmjs.org/")
            or not str(meta.get("integrity", "")).startswith("sha512-")
        )
    ]
    if bad: raise P0Error(f"package lock has an unreviewed registry entry: {bad}")
    return {"sha256": digest(raw), "packages": len(lock["packages"])}

def tree_manifest(root: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        item: dict[str, object] = {"path": relative, "mode": stat.S_IMODE(path.lstat().st_mode)}
        if path.is_symlink():
            item.update(type="symlink", target=os.readlink(path))
        elif path.is_dir():
            item["type"] = "directory"
        else:
            item.update(type="file", sha256=digest(path.read_bytes()))
        entries.append(item)
    return entries

def node_libraries(node: Path) -> list[tuple[str, Path]]:
    output = command(["ldd", str(node)], timeout=10).stdout.decode()
    paths: dict[str, Path] = {}
    for line in output.splitlines():
        tokens = line.replace("=>", " ").split()
        sources = [Path(token).resolve() for token in tokens if token.startswith("/") and Path(token).is_file()]
        if sources:
            name = Path(line.split()[0]).name if "=>" in line else sources[0].name
            paths[name] = sources[0]
    if not paths:
        raise P0Error("Node ELF closure is empty")
    return sorted(paths.items())

def set_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if not path.is_symlink():
            path.chmod(0o555 if path.is_dir() or os.access(path, os.X_OK) else 0o444)
    root.chmod(0o555)

def remove_tree(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if not path.is_symlink():
            with suppress(FileNotFoundError):
                path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)
    root.chmod(stat.S_IMODE(root.stat().st_mode) | stat.S_IWUSR)
    shutil.rmtree(root)

def build_release(run_root: Path, expected_source: dict[str, dict[str, object]]) -> tuple[Path, dict[str, object]]:
    lock = verify_lock()
    node, npm, cc = tool("node"), tool("npm"), tool("cc")
    if command([str(node), "--version"], timeout=10).stdout.decode().strip() != NODE_VERSION:
        raise P0Error("Node version changed")
    npm_cli = npm.resolve()
    source = run_root / "build-source"
    source.mkdir()
    for name in ("package.json", "package-lock.json", "tsconfig.json"):
        shutil.copy2(ROOT / name, source / name)
    shutil.copytree(ROOT / "src", source / "src")
    copied = {
        f"experiments/pi-runtime-p0/{path.relative_to(source)}":
        {"mode": stat.S_IMODE(path.stat().st_mode), "sha256": digest(path.read_bytes())}
        for path in source.rglob("*") if path.is_file() and path.name != "package-lock.json"
    }
    expected = {name: value for name, value in expected_source.items() if "/src/" in name or name.endswith(("package.json", "tsconfig.json"))}
    if copied != expected or digest((source / "package-lock.json").read_bytes()) != LOCK_SHA256:
        raise P0Error("copied build source differs from the captured source")
    home, cache = run_root / "build-home", run_root / "npm-cache"
    home.mkdir()
    cache.mkdir()
    canary = f"P0-BUILD-CANARY-{os.urandom(16).hex()}".encode()
    (home / "canary").write_bytes(canary)
    env = {
        "HOME": str(home), "PATH": f"{node.parent}:{cc.parent}", "LANG": "C", "LC_ALL": "C",
        "npm_config_cache": str(cache),
    }
    install = command(
        [str(node), str(npm_cli), "ci", "--ignore-scripts", "--no-audit", "--no-fund"],
        cwd=source, env=env, timeout=300,
    ).stdout
    tsc = source / "node_modules/typescript/bin/tsc"
    compile_log = command(
        [str(node), str(tsc), "-p", str(source / "tsconfig.json")], cwd=source, env=env, timeout=120,
    ).stdout
    launcher = source / "launch-in-cgroup"
    compile_c = command(
        [str(cc), "-O2", "-Wall", "-Wextra", "-Werror", "-o", str(launcher),
         str(source / "src/launch-in-cgroup.c")], env=env, timeout=60,
    ).stdout
    command(
        [str(node), str(npm_cli), "prune", "--omit=dev", "--ignore-scripts", "--no-audit", "--no-fund"],
        cwd=source, env=env, timeout=300,
    )
    release = run_root / "release"
    (release / "runtime/app").mkdir(parents=True)
    (release / "runtime/lib").mkdir()
    shutil.copy2(node, release / "runtime/node")
    shutil.copy2(source / "dist/runner.js", release / "runtime/app/runner.js")
    shutil.copy2(launcher, release / "runtime/launch-in-cgroup")
    shutil.copytree(source / "node_modules", release / "runtime/node_modules", symlinks=True)
    libraries = node_libraries(node)
    for name, library in libraries:
        shutil.copy2(library, release / "runtime/lib" / name)
    if any(canary in value for value in (install, compile_log, compile_c)):
        raise P0Error("build canary escaped in logs")
    if any(path.is_file() and canary in path.read_bytes() for path in release.rglob("*")):
        raise P0Error("build canary escaped into release")
    set_read_only(release / "runtime")
    manifest = tree_manifest(release)
    release_digest = digest(canonical(manifest))
    release.chmod(0o555)
    evidence = {"digest": release_digest, "lock": lock, "node": NODE_VERSION}
    return release, evidence

def strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise P0Error(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(P0Error(f"invalid number: {item}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise P0Error("invalid JSON") from error
    return value

def expected_payload() -> dict[str, object]:
    return {
        "model": "fixture-model",
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": [{"type": "text", "text": USER}]},
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True},
    }

def parse_provider_request(raw: bytes) -> tuple[bytes, dict[str, object]]:
    if len(raw) > MAX_FRAME or b"\r\n\r\n" not in raw:
        raise P0Error("provider request is malformed")
    head, body = raw.split(b"\r\n\r\n", 1)
    try:
        lines = head.decode("ascii").split("\r\n")
    except UnicodeDecodeError as error:
        raise P0Error("provider headers are not ASCII") from error
    if lines.pop(0) != "POST /v1/chat/completions HTTP/1.1":
        raise P0Error("provider method or path changed")
    allowed = {
        "host", "connection", "accept", "user-agent", "x-stainless-retry-count",
        "x-stainless-timeout", "x-stainless-lang", "x-stainless-package-version",
        "x-stainless-os", "x-stainless-arch", "x-stainless-runtime",
        "x-stainless-runtime-version", "authorization", "content-type",
        "accept-language", "sec-fetch-mode", "accept-encoding", "content-length",
    }
    headers: dict[str, str] = {}
    for line in lines:
        if ":" not in line:
            raise P0Error("malformed provider header")
        name, value = line.split(":", 1)
        name = name.lower()
        if name in headers or name not in allowed:
            raise P0Error(f"provider header rejected: {name}")
        headers[name] = value.strip()
    expected_headers = {
        "connection": "keep-alive", "accept": "application/json", "user-agent": "OpenAI/JS 6.26.0",
        "x-stainless-retry-count": "0", "x-stainless-timeout": "300", "x-stainless-lang": "js",
        "x-stainless-package-version": "6.26.0", "x-stainless-os": "Linux", "x-stainless-arch": "x64",
        "x-stainless-runtime": "node", "x-stainless-runtime-version": NODE_VERSION,
        "authorization": "Bearer local", "content-type": "application/json", "accept-language": "*",
        "sec-fetch-mode": "cors", "accept-encoding": "gzip, deflate",
    }
    try:
        content_length = int(headers.get("content-length", "-1"))
    except ValueError as error:
        raise P0Error("provider content length is invalid") from error
    host = headers.get("host", "")
    port_text = host.removeprefix("127.0.0.1:") if host.startswith("127.0.0.1:") else ""
    if (
        set(headers) != allowed
        or any(headers.get(name) != value for name, value in expected_headers.items())
        or not port_text.isdecimal() or not 1 <= int(port_text) <= 65535
        or content_length != len(body)
    ):
        raise P0Error("provider content headers changed")
    payload = strict_json(body)
    if canonical(payload) != canonical(expected_payload()):
        raise P0Error("provider payload changed")
    encoded = canonical(payload)
    return encoded, {
        "raw_sha256": digest(raw), "received_sha256": digest(body),
        "canonical_sha256": digest(encoded), "headers": sorted(headers), "host": host,
    }

class Channel:
    def __init__(self, connection: socket.socket):
        self.connection = connection
        self.buffer = bytearray()
    def read(self, deadline: Deadline) -> dict[str, Any]:
        while b"\n" not in self.buffer:
            self.connection.settimeout(deadline.remaining("control read"))
            chunk = self.connection.recv(min(65536, MAX_FRAME - len(self.buffer)))
            if not chunk:
                raise EOFError("channel closed")
            self.buffer.extend(chunk)
            if len(self.buffer) >= MAX_FRAME:
                raise P0Error("protocol frame too large")
        raw, _, rest = self.buffer.partition(b"\n")
        self.buffer = bytearray(rest)
        value = strict_json(bytes(raw).rstrip(b" "))
        if not isinstance(value, dict):
            raise P0Error("protocol frame is not an object")
        return value
    def write(self, value: dict[str, object], deadline: Deadline) -> None:
        raw = canonical(value) + b"\n"
        if len(raw) > MAX_FRAME:
            raise P0Error("outbound frame too large")
        self.connection.settimeout(deadline.remaining("control write"))
        self.connection.sendall(raw)

def expect(frame: dict[str, Any], expected: dict[str, object], label: str) -> None:
    if canonical(frame) != canonical(expected):
        raise P0Error(f"{label} changed: {frame}")

class FixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def do_POST(self) -> None:
        owner: FixtureServer = self.server.owner  # type: ignore[attr-defined]
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            self.send_error(400)
            return
        if length < 0 or length > MAX_FRAME:
            self.send_error(413)
            return
        body = self.rfile.read(length)
        with owner.lock:
            owner.requests += 1
            admitted = owner.requests == 1
        if not admitted or self.path != "/v1/chat/completions" or body != owner.expected:
            self.send_error(409 if not admitted else 400)
            return
        owner.request = body
        owner.active.set()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            chunks = owner.chunks if not owner.endless else owner.chunks[:1]
            while True:
                for chunk in chunks:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                if not owner.endless or owner.stop.wait(0.02):
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.close_connection = True
            owner.terminal.set()
    def log_message(self, _format: str, *_args: object) -> None:
        pass

class FixtureServer:
    def __init__(self, expected: bytes, *, endless: bool):
        self.expected, self.endless = expected, endless
        self.identity = os.urandom(16).hex()
        self.active, self.stop, self.terminal = threading.Event(), threading.Event(), threading.Event()
        self.lock = threading.Lock()
        self.requests = 0
        self.request: bytes | None = None
        base = {"id": "p0", "object": "chat.completion.chunk", "created": 1, "model": "fixture-model"}
        self.chunks = [
            f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'P0_OK'}, 'finish_reason': None}]})}\n\n".encode(),
            f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})}\n\ndata: [DONE]\n\n".encode(),
        ]
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
        self.server.owner = self  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
    def start(self) -> None:
        self.thread.start()
    def close(self, deadline: Deadline) -> None:
        self.stop.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(deadline.remaining("fixture thread"))
        if self.thread.is_alive():
            raise P0Error("fixture server did not stop")

def gateway(
    channel: Channel, raw: bytes, mode: str, deadline: Deadline, observer: Channel | None = None,
) -> dict[str, object]:
    canonical_body, evidence = parse_provider_request(raw)
    fixture = FixtureServer(canonical_body, endless=mode in {"cancel", "authority-death"})
    fixture.start()
    connection = http.client.HTTPConnection("127.0.0.1", fixture.server.server_port, timeout=deadline.remaining())
    response: http.client.HTTPResponse | None = None
    try:
        connection.request(
            "POST", "/v1/chat/completions", body=canonical_body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(canonical_body))},
        )
        response = connection.getresponse()
        if response.status != 200 or not fixture.active.wait(deadline.remaining("upstream activation")):
            raise P0Error("upstream did not activate")
        if connection.sock is not None:
            connection.sock.settimeout(deadline.remaining("upstream response"))
        first = response.fp.readline(MAX_FRAME) + response.fp.readline(MAX_FRAME)  # type: ignore[union-attr]
        channel.write({"request": 1, "type": "response_start"}, deadline)
        channel.write({"data": base64.b64encode(first).decode(), "request": 1, "type": "response_chunk"}, deadline)
        if mode == "authority-death":
            if observer is None:
                raise P0Error("authority-death observer is absent")
            observer.write({"type": "authority_death_ready", "upstream_active": True}, deadline)
            signal.pause()
            raise P0Error("authority death signal returned")
        if mode == "cancel":
            channel.write({"request": 1, "type": "cancel"}, deadline)
            expect(channel.read(deadline), {"request": 1, "type": "client_closed"}, "client close")
            fixture.stop.set()
            response.close()
            connection.close()
            if not fixture.terminal.wait(deadline.remaining("cancelled upstream terminal")):
                raise P0Error("cancelled upstream remained active")
            channel.write({"request": 1, "type": "cancel_ack"}, deadline)
        else:
            rest = response.read(MAX_FRAME + 1)
            if len(rest) > MAX_FRAME:
                raise P0Error("upstream response exceeded its byte budget")
            channel.write({"data": base64.b64encode(rest).decode(), "request": 1, "type": "response_chunk"}, deadline)
            channel.write({"request": 1, "type": "response_end"}, deadline)
            connection.close()
        if not fixture.terminal.wait(deadline.remaining("upstream terminal")):
            raise P0Error("upstream request did not become terminal")
        if fixture.requests != 1:
            raise P0Error("upstream one-shot admission changed")
        return {
            **evidence, "upstream_sha256": digest(fixture.request or b""),
            "upstream_instance": fixture.identity, "upstream_requests": fixture.requests,
            "cancelled": mode == "cancel",
        }
    finally:
        if mode != "authority-death":
            if response is not None:
                response.close()
            connection.close()
            fixture.close(Deadline(CLEANUP_SECONDS))

def cgroup_path(pid: int | None = None) -> Path:
    text = Path(f"/proc/{pid or 'self'}/cgroup").read_text().strip()
    if not text.startswith("0::/"):
        raise P0Error("P0 requires unified cgroup v2")
    return Path("/sys/fs/cgroup") / text[3:].lstrip("/")

def process_identity(pid: int) -> dict[str, int]:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "starttime": int(fields[19])}

def inode_identity(metadata: os.stat_result, root: Path) -> dict[str, object]:
    return {"root": root.name, "device": metadata.st_dev, "inode": metadata.st_ino}

def populated(path: Path) -> bool:
    values = dict(line.split() for line in (path / "cgroup.events").read_text().splitlines())
    return values.get("populated") == "1"
def wait_cgroup_empty(path: Path, deadline: Deadline) -> None:
    descriptor = os.open(path / "cgroup.events", os.O_RDONLY)
    try:
        os.read(descriptor, 4096)
        while populated(path):
            if not select.select([], [], [descriptor], deadline.remaining("runner cgroup empty"))[2]: raise P0Error("runner cgroup remained populated")
            os.lseek(descriptor, 0, os.SEEK_SET); os.read(descriptor, 4096)
    finally: os.close(descriptor)
def pidfd_alive(descriptor: int) -> bool:
    return not select.select([descriptor], [], [], 0)[0]

def wait_pidfd(descriptor: int, deadline: Deadline, label: str) -> None:
    if not select.select([descriptor], [], [], deadline.remaining(label))[0]:
        raise P0Error(f"{label} did not become terminal")

def read_fd_line(descriptor: int, deadline: Deadline, label: str) -> bytes:
    if not select.select([descriptor], [], [], deadline.remaining(label))[0]:
        raise P0Error(f"{label} timed out")
    raw = os.read(descriptor, MAX_FRAME)
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise P0Error(f"{label} is malformed")
    return raw

def drain_dead_peer(channel: Channel, deadline: Deadline) -> dict[str, object]:
    raw = bytearray(channel.buffer)
    channel.buffer.clear()
    while True:
        if not select.select([channel.connection], [], [], deadline.remaining("fail-stop drain"))[0]:
            raise P0Error("fail-stop control connection did not close")
        chunk = channel.connection.recv(min(65536, MAX_FRAME + 1 - len(raw)))
        if not chunk:
            break
        raw.extend(chunk)
        if len(raw) > MAX_FRAME:
            raise P0Error("fail-stop evidence exceeded its byte budget")
    if b'"type":"provider_request"' in raw:
        raise P0Error("fail-stop emitted a provider admission frame")
    return {"bytes": len(raw), "provider_frames": 0, "sha256": digest(raw)}

def launch_runner(
    scenario: str, release: Path, control_path_fd: int, session_fd: int, fifo_fd: int,
    runner_cgroup: Path, deadline: Deadline,
) -> tuple[int, int, subprocess.Popen[bytes], int]:
    status_r, status_w = os.pipe()
    cgroup_fd = os.open(runner_cgroup, os.O_PATH | os.O_DIRECTORY)
    session_path_fd = os.open(f"/proc/self/fd/{session_fd}", os.O_PATH)
    args = [
        str(release / "runtime/launch-in-cgroup"), str(cgroup_fd), "--",
        str(tool("prlimit")), f"--fsize={MAX_SESSION}:{MAX_SESSION}", "--nofile=64:64", "--core=0:0", "--",
        str(tool("bwrap")), "--unshare-all", "--unshare-user", "--new-session", "--disable-userns",
        "--cap-drop", "ALL", "--uid", "65534", "--gid", "65534",
        "--ro-bind", str(release / "runtime"), "/runtime", "--dir", "/run", "--dir", "/run/assist-p0",
        "--ro-bind-fd", str(control_path_fd), "/run/assist-p0/control.sock",
        "--dir", "/session", "--bind-fd", str(session_path_fd), "/session/session.jsonl",
        "--dir", "/workspace", "--dir", "/agent", "--proc", "/proc", "--dev", "/dev",
        "--tmpfs", "/tmp", "--dir", "/tmp/home", "--clearenv", "--setenv", "HOME", "/tmp/home",
        "--setenv", "PATH", "/runtime", "--chdir", "/tmp", "--json-status-fd", str(status_w),
        "--bind-fd", str(fifo_fd), FIFO,
    ]
    loader = next((release / "runtime/lib").glob("ld-linux-*.so.*"))
    args += [
        f"/runtime/lib/{loader.name}", "--library-path", "/runtime/lib", "/runtime/node",
        "/runtime/app/runner.js", scenario,
    ]
    inherited = (cgroup_fd, status_w, control_path_fd, session_path_fd, fifo_fd)
    launcher = subprocess.Popen(
        args, pass_fds=inherited, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, env={},
    )
    for descriptor in inherited:
        os.close(descriptor)
    try:
        value = strict_json(read_fd_line(status_r, deadline, "Bubblewrap status"))
        status_fields = {"child-pid", "cgroup-namespace", "ipc-namespace", "mnt-namespace", "net-namespace", "pid-namespace", "uts-namespace"}
        if not isinstance(value, dict) or set(value) != status_fields or not all(isinstance(item, int) for item in value.values()):
            raise P0Error(f"invalid Bubblewrap status: {value}")
        init_pid = value["child-pid"]
        if Path(f"/proc/{init_pid}/environ").read_bytes():
            raise P0Error("Bubblewrap PID 1 inherited the authority environment")
        return init_pid, os.pidfd_open(init_pid), launcher, status_r
    except BaseException:
        close_fd(status_r)
        if populated(runner_cgroup):
            (runner_cgroup / "cgroup.kill").write_text("1")
        launcher.wait(timeout=CLEANUP_SECONDS)
        raise

def accept_runner(
    listener: socket.socket, init_pid: int, init_pidfd: int, runner_cgroup: Path,
    host_canary: Path, deadline: Deadline,
) -> tuple[Channel, int, int, dict[str, object]]:
    ready, _, _ = select.select([listener, init_pidfd], [], [], deadline.remaining("runner accept"))
    if init_pidfd in ready:
        raise P0Error("runner died before connecting")
    if listener not in ready:
        raise P0Error("runner did not connect")
    connection, _ = listener.accept()
    pid, _uid, _gid = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
    node_pidfd = os.pidfd_open(pid)
    try:
        namespace_pids = next(
            line.split()[1:] for line in Path(f"/proc/{pid}/status").read_text().splitlines()
            if line.startswith("NSpid:")
        )
        host_net, runner_net = Path("/proc/self/ns/net").stat(), Path(f"/proc/{pid}/ns/net").stat()
        host_mount, runner_mount = Path("/proc/self/ns/mnt").stat(), Path(f"/proc/{pid}/ns/mnt").stat()
        routes = Path(f"/proc/{pid}/net/route").read_text().splitlines()
        core_limit = next(
            line for line in Path(f"/proc/{pid}/limits").read_text().splitlines()
            if line.startswith("Max core file size")
        ).split()[-3:-1]
        environment = sorted(filter(None, Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")))
        if (
            pid == init_pid or cgroup_path(pid) != runner_cgroup or not pidfd_alive(init_pidfd)
            or not pidfd_alive(node_pidfd) or namespace_pids[-1] != "2"
            or host_net.st_ino == runner_net.st_ino or host_mount.st_ino == runner_mount.st_ino
            or len(routes) != 1 or not host_canary.is_file() or Path(f"/proc/{pid}/root{host_canary}").exists()
            or core_limit != ["0", "0"]
            or environment != [b"HOME=/tmp/home", b"PATH=/runtime", b"PWD=/tmp"]
        ):
            raise P0Error("runner peer identity changed")
    except BaseException:
        connection.close()
        os.close(node_pidfd)
        raise
    listener_path = Path(listener.getsockname())
    listener.close()
    listener_path.unlink(missing_ok=True)
    containment = {"net_namespace": runner_net.st_ino, "mount_namespace": runner_mount.st_ino,
                   "ipv4_routes": len(routes) - 1, "host_canary_absent": True,
                   "core_limit": core_limit, "environment": [value.decode() for value in environment]}
    return Channel(connection), pid, node_pidfd, containment

def session_evidence(descriptor: int, *, completed: bool) -> dict[str, object]:
    os.fsync(descriptor)
    size = os.fstat(descriptor).st_size
    if size > MAX_SESSION:
        raise P0Error("Pi session exceeded its file-size limit")
    raw = os.pread(descriptor, size + 1, 0)
    if len(raw) != size or (raw and not raw.endswith(b"\n")):
        raise P0Error("Pi session evidence is incomplete")
    records = [strict_json(line) for line in raw.splitlines()]
    assistants = [
        entry for entry in records
        if isinstance(entry, dict) and entry.get("type") == "message"
        and isinstance(entry.get("message"), dict) and entry["message"].get("role") == "assistant"
    ]
    terminal: dict[str, object] | None = None
    if completed:
        if len(assistants) != 1:
            raise P0Error("completed session does not have exactly one assistant record")
        message = assistants[0]["message"]
        terminal = {"content": message.get("content"), "stopReason": message.get("stopReason")}
        if terminal != {"content": [{"type": "text", "text": "P0_OK"}], "stopReason": "stop"}:
            raise P0Error(f"completed assistant record changed: {terminal}")
    return {"sha256": digest(raw), "bytes": len(raw), "assistant_records": len(assistants), "terminal": terminal}

def cleanup_runner(
    runner_cgroup: Path | None, node_pidfd: int | None, init_pidfd: int | None,
    launcher: subprocess.Popen[bytes] | None,
    deadline: Deadline,
) -> dict[str, object]:
    if runner_cgroup is not None and runner_cgroup.exists() and populated(runner_cgroup):
        (runner_cgroup / "cgroup.kill").write_text("1")
    if node_pidfd is not None:
        wait_pidfd(node_pidfd, deadline, "Node cleanup")
    if init_pidfd is not None:
        wait_pidfd(init_pidfd, deadline, "Bubblewrap cleanup")
    code = launcher.wait(timeout=deadline.remaining("launcher reap")) if launcher is not None else None
    if code != 137:
        raise P0Error(f"cgroup launcher exit changed: {code}")
    if runner_cgroup is not None and runner_cgroup.exists():
        wait_cgroup_empty(runner_cgroup, deadline)
        runner_cgroup.rmdir()
    return {"node_terminal": node_pidfd is not None, "runner_cgroup_empty": runner_cgroup is not None,
            "bwrap_reaped": launcher is not None, "launcher_exit": code}

def close_fd(descriptor: int | None) -> None:
    if descriptor is not None:
        with suppress(OSError):
            os.close(descriptor)

def scenario_authority(name: str, root: Path, observer_path: Path, release: Path) -> int:
    work = Deadline(WORK_SECONDS)
    observer_socket = socket.socket(socket.AF_UNIX)
    observer_socket.connect(str(observer_path))
    observer = Channel(observer_socket)
    listener: socket.socket | None = None
    channel: Channel | None = None
    launcher: subprocess.Popen[bytes] | None = None
    runner_cgroup: Path | None = None
    session_fd: int | None = None
    fifo_read: int | None = None
    init_pidfd: int | None = None
    node_pidfd: int | None = None
    status_fd: int | None = None
    result: dict[str, object] | None = None
    cleanup_error: BaseException | None = None
    try:
        if os.environ.get("P0_SECRET_CANARY") != ENV_CANARY:
            raise P0Error("authority environment canary is absent")
        observer.write({"pid": os.getpid(), "type": "authority_online"}, work)
        expect(observer.read(work), {"type": "authority_start"}, "authority start")
        runner_cgroup = cgroup_path() / "runner"
        runner_cgroup.mkdir()
        listener = socket.socket(socket.AF_UNIX)
        control_path = root / "control.sock"
        listener.bind(str(control_path))
        listener.listen(1)
        control_identity = inode_identity(os.fstat(listener.fileno()), root)
        control_path_fd = os.open(control_path, os.O_PATH | os.O_NOFOLLOW)
        session_path = root / "session.jsonl"
        session_fd = os.open(session_path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        session_identity = inode_identity(os.fstat(session_fd), root)
        fifo_path = root / "liveness.fifo"
        fifo_fd = os.open(fifo_path, os.O_PATH | os.O_NOFOLLOW)
        fifo_read = os.open(fifo_path, os.O_RDONLY | os.O_NONBLOCK)
        host_canary = root / "host-secret-canary"
        host_canary.write_text(ENV_CANARY)
        init_pid, init_pidfd, launcher, status_fd = launch_runner(
            name, release, control_path_fd, session_fd, fifo_fd, runner_cgroup, work,
        )
        channel, node_pid, node_pidfd, containment = accept_runner(
            listener, init_pid, init_pidfd, runner_cgroup, host_canary, work,
        )
        identities = {"control": control_identity, "runner": process_identity(node_pid),
                      "session": session_identity, "containment": containment}
        listener = None
        expect(channel.read(work), {"scenario": name, "type": "hello"}, "runner hello")
        channel.write({"type": "accepted"}, work)
        first = channel.read(work)
        if name == "hostile":
            expect(first, {"type": "hostile_ready"}, "hostile readiness")
            try:
                before = os.read(fifo_read, 1)
            except BlockingIOError:
                before = None
            if not pidfd_alive(node_pidfd) or not populated(runner_cgroup) or before is not None:
                raise P0Error("hostile pre-kill evidence was not active")
            channel.connection.close()
            channel = None
            cleanup = cleanup_runner(runner_cgroup, node_pidfd, init_pidfd, launcher, Deadline(CLEANUP_SECONDS))
            if os.read(fifo_read, 1) != b"":
                raise P0Error("hostile descendant retained its FIFO after cgroup kill")
            runner_cgroup = None
            result = {"scenario": name, "status": "PASS", "causal_cgroup_kill": True, "cleanup": cleanup}
        else:
            expect(first, {"type": "runner_ready"}, "runner readiness")
            channel.write({"type": "begin"}, work)
            if name in {"fail-closed", "fail-backpressure"}:
                wait_pidfd(node_pidfd, work, "fail-stop Node")
                if launcher.wait(timeout=work.remaining("fail-stop signal")) != 137:
                    raise P0Error("fail-stop was not SIGKILL")
                if name == "fail-backpressure" and os.read(fifo_read, 1) != b"B":
                    raise P0Error("backpressure was not reached before fail-stop")
                fail_stop = drain_dead_peer(channel, work)
                channel.connection.close()
                channel = None
                cleanup = cleanup_runner(runner_cgroup, node_pidfd, init_pidfd, launcher, Deadline(CLEANUP_SECONDS))
                runner_cgroup = None
                result = {
                    "scenario": name, "status": "PASS", "requests": 0, "one_shot_admissions": 0,
                    "fail_stop": fail_stop, "cleanup": cleanup,
                }
            else:
                frame = channel.read(work)
                if (
                    set(frame) != {"raw", "request", "type"}
                    or frame.get("request") != 1 or frame.get("type") != "provider_request"
                    or not isinstance(frame.get("raw"), str)
                ):
                    raise P0Error("runner provider request frame changed")
                raw = base64.b64decode(frame["raw"], validate=True)
                mode = "cancel" if name == "cancel" else name
                provider = gateway(channel, raw, mode, work, observer if name == "authority-death" else None)
                done = channel.read(work)
                expected_counts = (2, 1) if name == "cancel" else (1, 0)
                if (
                    set(done) != {"connections", "rejected", "request", "type"}
                    or done.get("request") != 1 or done.get("type") != "scenario_done"
                    or (done.get("connections"), done.get("rejected")) != expected_counts
                ):
                    raise P0Error(f"scenario completion changed: {done}")
                channel.connection.close()
                channel = None
                cleanup = cleanup_runner(runner_cgroup, node_pidfd, init_pidfd, launcher, Deadline(CLEANUP_SECONDS))
                runner_cgroup = None
                evidence = session_evidence(session_fd, completed=name in {"success", "fresh"})
                result = {
                    "scenario": name, "status": "PASS", "requests": 1, "one_shot_admissions": 1,
                    "provider": provider, "bridge": done, "session": evidence, "cleanup": cleanup,
                }
        close_fd(node_pidfd)
        node_pidfd = None
        close_fd(init_pidfd)
        init_pidfd = None
        close_fd(session_fd)
        session_fd = None
        close_fd(fifo_read)
        fifo_read = None
        close_fd(status_fd)
        status_fd = None
        result["identities"] = identities
        result["evidence_sha256"] = digest(canonical(result))
        observer.write({"result": result, "type": "scenario_result"}, work)
        return 0
    except BaseException as error:
        try:
            if channel is not None:
                channel.connection.close()
            cleanup_runner(runner_cgroup, node_pidfd, init_pidfd, launcher, Deadline(CLEANUP_SECONDS))
        except BaseException as secondary:
            cleanup_error = secondary
        detail = f"{type(error).__name__}: {error}"
        if cleanup_error is not None:
            detail += f"; cleanup: {type(cleanup_error).__name__}: {cleanup_error}"
        with suppress(BaseException):
            observer.write({"error": detail, "evidence": str(root), "type": "scenario_error"}, Deadline(CLEANUP_SECONDS))
        return 1
    finally:
        if listener is not None:
            with suppress(OSError):
                listener.close()
        close_fd(node_pidfd)
        close_fd(init_pidfd)
        close_fd(session_fd)
        close_fd(fifo_read)
        close_fd(status_fd)
        observer_socket.close()

def unit_properties(unit: str, deadline: Deadline) -> dict[str, str]:
    names = (
        "InvocationID", "MainPID", "ControlGroup", "Type", "ExitType", "KillMode", "SendSIGKILL",
        "Restart", "TimeoutStopUSec", "RuntimeMaxUSec", "OOMPolicy", "Delegate", "MemoryMax", "TasksMax",
        "CPUQuotaPerSecUSec", "LimitCORE",
    )
    raw = command(
        ["systemctl", "--user", "show", unit, *sum((["-p", name] for name in names), [])],
        timeout=deadline.remaining("unit property query"),
    ).stdout.decode()
    properties = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
    expected = {
        "Type": "exec", "ExitType": "main", "KillMode": "control-group", "SendSIGKILL": "yes",
        "Restart": "no", "TimeoutStopUSec": "1s", "RuntimeMaxUSec": "30s", "OOMPolicy": "stop",
        "Delegate": "yes", "MemoryMax": str(2 * 1024**3), "TasksMax": "64",
        "CPUQuotaPerSecUSec": "1s", "LimitCORE": "0",
    }
    if any(properties.get(name) != value for name, value in expected.items()):
        raise P0Error(f"effective transient-unit properties changed: {properties}")
    if not properties.get("InvocationID") or not properties.get("ControlGroup"):
        raise P0Error("transient unit identity is incomplete")
    return properties

def empty_certificate(events_fd: int, cgroup: Path, properties: dict[str, str]) -> str:
    if properties.get("Delegate") != "yes":
        raise P0Error("unit delegation changed")
    try:
        os.lseek(events_fd, 0, os.SEEK_SET)
        data = os.read(events_fd, 4096).decode()
        if "populated 0" not in data:
            raise P0Error("service cgroup remained populated")
        return "populated-0"
    except OSError as error:
        if error.errno != errno.ENODEV or cgroup.exists():
            raise
        return "removed-enodev"

def run_scenario(name: str, release: Path, invocation: str) -> dict[str, object]:
    work = Deadline(25)
    unit = f"assist-pi-p0-{invocation[-8:]}-{name}.service"
    root: Path | None = None
    listener: socket.socket | None = None
    waiter: subprocess.Popen[bytes] | None = None
    connection: socket.socket | None = None
    fifo_read: int | None = None
    main_pidfd: int | None = None
    events_fd: int | None = None
    service_cgroup: Path | None = None
    result: dict[str, object] | None = None
    try:
        root = Path(tempfile.mkdtemp(prefix=f"{name}-", dir="/tmp"))
        root.chmod(0o700)
        fifo = root / "liveness.fifo"
        os.mkfifo(fifo, 0o600)
        fifo_read = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
        observer_path = root / "observer.sock"
        listener = socket.socket(socket.AF_UNIX)
        listener.bind(str(observer_path))
        listener.listen(1)
        argv = [
            "systemd-run", "--user", "--wait", "--collect", "--quiet", f"--unit={unit}", "--service-type=exec",
            f"--setenv=P0_SECRET_CANARY={ENV_CANARY}",
            "-p", "ExitType=main", "-p", "KillMode=control-group", "-p", "SendSIGKILL=yes", "-p", "Restart=no",
            "-p", "TimeoutStopSec=1s", "-p", "RuntimeMaxSec=30s", "-p", "OOMPolicy=stop", "-p", "Delegate=yes",
            "-p", "MemoryMax=2G", "-p", "TasksMax=64", "-p", "CPUQuota=100%",
            "-p", "LimitCORE=0",
            "-p", "StandardOutput=null", "-p", "StandardError=journal",
            sys.executable, str(Path(__file__).resolve()), "scenario", name, str(root), str(observer_path), str(release),
        ]
        waiter = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True,
        )
        listener.settimeout(work.remaining("authority accept"))
        connection, _ = listener.accept()
        listener.close()
        listener = None
        observer = Channel(connection)
        online = observer.read(work)
        if set(online) != {"pid", "type"} or online["type"] != "authority_online" or not isinstance(online["pid"], int):
            raise P0Error("scenario authority did not start")
        main_pidfd = os.pidfd_open(online["pid"])
        properties = unit_properties(unit, work)
        main_pid = int(properties["MainPID"])
        if main_pid != online["pid"]:
            raise P0Error("systemd MainPID does not match the authority")
        service_cgroup = Path("/sys/fs/cgroup") / properties["ControlGroup"].lstrip("/")
        events_fd = os.open(service_cgroup / "cgroup.events", os.O_RDONLY)
        observer.write({"type": "authority_start"}, work)
        frame = observer.read(work)
        authority_death = name == "authority-death"
        if authority_death:
            expect(frame, {"type": "authority_death_ready", "upstream_active": True}, "authority-death readiness")
            try:
                before = os.read(fifo_read, 1)
            except BlockingIOError:
                before = None
            if before is not None:
                raise P0Error("authority-death descendant FIFO was not active")
            signal.pidfd_send_signal(main_pidfd, signal.SIGKILL)
            try:
                observer.read(Deadline(CLEANUP_SECONDS))
            except (EOFError, OSError):
                pass
            else:
                raise P0Error("dead authority emitted a result")
        else:
            if set(frame) != {"result", "type"} or frame["type"] != "scenario_result" or not isinstance(frame["result"], dict):
                raise P0Error(f"scenario failed: {frame}")
            result = frame["result"]
            if result.get("scenario") != name or result.get("status") != "PASS":
                raise P0Error(f"scenario result changed: {result}")
        expected_codes = {0} if not authority_death else {255}
        code = waiter.wait(timeout=work.remaining("scenario service"))
        if code not in expected_codes:
            raise P0Error(f"systemd-run failed with {code}")
        terminal = Deadline(CLEANUP_SECONDS)
        wait_pidfd(main_pidfd, terminal, "authority cleanup")
        certificate = empty_certificate(events_fd, service_cgroup, properties)
        if authority_death:
            if os.read(fifo_read, 1) != b"":
                raise P0Error("authority-death descendant survived service cleanup")
            result = {
                "scenario": name, "status": "PASS", "dead_authority_report_accepted": False,
                "main_pid_terminal": True, "fifo_eof": True, "predeath_readiness": frame,
            }
        result["unit"] = {**properties, "empty_certificate": certificate}
        result["outer_evidence_sha256"] = digest(canonical(result))
    except BaseException as caught:
        raise P0Error(f"{type(caught).__name__}: {caught}; evidence: {root}") from caught
    finally:
        primary = sys.exc_info()[1]
        cleanup_error: BaseException | None = None
        cleanup = Deadline(CLEANUP_SECONDS)
        try:
            if waiter is not None and (primary is not None or waiter.poll() is None):
                command(["systemctl", "--user", "stop", unit], timeout=cleanup.remaining("unit stop"))
                waiter.wait(timeout=cleanup.remaining("systemd-run cleanup"))
        except BaseException as caught:
            cleanup_error = caught
            fallback = Deadline(35)
            if main_pidfd is not None and pidfd_alive(main_pidfd):
                with suppress(ProcessLookupError):
                    signal.pidfd_send_signal(main_pidfd, signal.SIGKILL)
                with suppress(P0Error):
                    wait_pidfd(main_pidfd, fallback, "failed unit authority cleanup")
            if waiter is not None and waiter.poll() is None:
                with suppress(subprocess.TimeoutExpired, P0Error):
                    waiter.wait(timeout=fallback.remaining("failed unit cleanup"))
            if waiter is not None and waiter.poll() is None:
                waiter.kill()
                try:
                    waiter.wait(timeout=1)
                except subprocess.TimeoutExpired as secondary:
                    cleanup_error = P0Error(f"{cleanup_error}; systemd-run could not be reaped: {secondary}")
        try:
            stale_unit_preflight()
        except BaseException as caught:
            cleanup_error = cleanup_error or caught
        if events_fd is not None and service_cgroup is not None:
            try:
                empty_certificate(events_fd, service_cgroup, {"Delegate": "yes"})
            except BaseException as caught:
                cleanup_error = cleanup_error or caught
        if listener is not None:
            with suppress(OSError):
                listener.close()
        if connection is not None:
            connection.close()
        close_fd(main_pidfd)
        close_fd(events_fd)
        close_fd(fifo_read)
        if cleanup_error is None and primary is None and root is not None:
            try:
                shutil.rmtree(root)
            except BaseException as caught:
                cleanup_error = caught
        if cleanup_error is not None:
            detail = f"cleanup: {type(cleanup_error).__name__}: {cleanup_error}"
            if primary is not None:
                detail = f"{type(primary).__name__}: {primary}; {detail}"
                raise P0Error(detail) from cleanup_error
            elif result is not None:
                result.update(status="FAIL", cleanup_error=detail, evidence_root=str(root))
                result.pop("outer_evidence_sha256", None); result["outer_evidence_sha256"] = digest(canonical(result))
            else:
                raise P0Error(detail) from cleanup_error
    if result is None: raise P0Error("scenario produced no result")
    return result

def stale_unit_preflight() -> None:
    output = command(
        ["systemctl", "--user", "list-units", "assist-pi-p0-*", "--state=active,activating,deactivating",
         "--no-legend", "--no-pager"],
        timeout=10,
    ).stdout.decode().strip()
    if output: raise P0Error(f"a stale P0 unit is active: {output}")

def line_count() -> tuple[int, list[str], dict[str, dict[str, object]]]:
    raw = command(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "experiments/pi-runtime-p0"],
        cwd=ROOT.parents[1], timeout=10,
    ).stdout.decode().splitlines()
    files = [
        value for value in sorted(raw)
        if (ROOT.parents[1] / value).is_file() and Path(value).name != "package-lock.json"
        and not any(part in {"node_modules", "dist", ".artifacts", ".build"} for part in Path(value).parts)
    ]
    contents = {value: (ROOT.parents[1] / value).read_bytes() for value in files}
    total = sum(len(data.splitlines()) for data in contents.values())
    identity = {
        value: {"mode": stat.S_IMODE((ROOT.parents[1] / value).stat().st_mode), "sha256": digest(data)}
        for value, data in contents.items()
    }
    return total, files, identity

def write_report(report: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    data = canonical(report) + b"\n"
    offset = 0
    try:
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise P0Error("report write made no progress")
            offset += written
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    os.replace(temporary, destination)

def run_all() -> tuple[Path, bool]:
    invocation = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + os.urandom(4).hex()
    run_root = ROOT / ".build" / invocation
    run_root.mkdir(parents=True)
    atexit.register(remove_tree, run_root)
    lock_fd: int | None = None
    results: list[dict[str, object]] = []
    build: dict[str, object] | None = None
    error: str | None = None
    lines: int | None = None
    files: list[str] = []
    source: dict[str, dict[str, object]] = {}
    systemd: str | None = None
    try:
        lines, files, source = line_count()
        lock_fd = os.open("/tmp/assist-pi-runtime-p0.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as caught:
            raise P0Error("another P0 invocation is active") from caught
        stale_unit_preflight()
        systemd = command(["systemctl", "--version"], timeout=5).stdout.decode().splitlines()[0]
        release, build = build_release(run_root, source)
        for name in SCENARIOS:
            outcome = run_scenario(name, release, invocation)
            results.append(outcome)
            if outcome.get("status") != "PASS":
                raise P0Error(f"scenario cleanup failed: {outcome}")
        current_lines, current_files, current_source = line_count()
        if ((current_lines, current_files, current_source) != (lines, files, source)
                or digest((ROOT / "package-lock.json").read_bytes()) != LOCK_SHA256):
            raise P0Error("counted P0 source changed during the run")
        indexed = {result["scenario"]: result for result in results}
        cancel, fresh = indexed["cancel"], indexed["fresh"]
        if any(cancel["identities"][key] == fresh["identities"][key] for key in ("control", "runner", "session")):
            raise P0Error("fresh scenario reused a prior runtime identity")
        if cancel["provider"]["upstream_instance"] == fresh["provider"]["upstream_instance"]:
            raise P0Error("fresh scenario reused a prior upstream fixture")
        if lines > 1500:
            raise P0Error(f"simplicity gate exceeded: {lines} lines")
        status, scope = "GO", "P1_ONLY"
    except BaseException as caught:
        status, scope = "STOP", "NONE"
        error = f"{type(caught).__name__}: {caught}"
    try:
        remove_tree(run_root)
    except BaseException as caught:
        status, scope = "STOP", "NONE"
        error = "; ".join(filter(None, (error, f"build cleanup: {type(caught).__name__}: {caught}")))
    finally:
        close_fd(lock_fd)
    report = {
        "status": status, "scope": scope, "invocation": invocation, "error": error,
        "build": build, "containment": {
            "kernel": os.uname().release, "cgroup": "v2",
            "systemd": systemd,
        },
        "scenarios": results, "simplicity": {"lines": lines, "files": files}, "source": source,
    }
    destination = ROOT / ".artifacts" / "runs" / invocation / "report.json"
    write_report(report, destination)
    return destination, status == "GO"

def main() -> int:
    if len(sys.argv) == 6 and sys.argv[1] == "scenario":
        return scenario_authority(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))
    if sys.argv[1:] == ["offline"]:
        destination, passed = run_all()
        print(destination)
        return 0 if passed else 1
    print("usage: pi_runtime_p0.py offline", file=sys.stderr)
    return 2
if __name__ == "__main__":
    raise SystemExit(main())
