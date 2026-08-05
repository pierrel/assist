#!/usr/bin/env python3
"""Disposable P0 build and containment driver.

This module is deliberately confined to the experiment.  It is evidence machinery,
not the future RuntimeSession, product gateway, or deployment path.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPERIMENT = Path(__file__).resolve().parents[1]
BUILD_ROOT = EXPERIMENT / ".build"
ARTIFACT_ROOT = EXPERIMENT / ".artifacts"
RUNTIME_ROOT = Path("/tmp/assist-pi-runtime-p0")
_NODE_COMMAND = os.environ.get("ASSIST_PI_P0_NODE") or shutil.which("node")
NODE = Path(_NODE_COMMAND) if _NODE_COMMAND else None
EXPECTED_NODE = "v22.23.1"
MAX_FRAME = 8 * 1024 * 1024
MAX_LOG_TAIL = 64 * 1024
PROFILE = {
    "api": "openai-completions",
    "context_window": 131072,
    "model": "Qwen_Qwen3.6-27B-Q4_K_M.gguf",
    "provider": "assist-local-qwen",
}
OWNED_PROMPT = "ASSIST_P0_PROMPT_OWNERSHIP_CANARY_v1"
ALLOWED_SESSION_TYPES = {
    "custom",
    "message",
    "model_change",
    "session",
    "thinking_level_change",
}
MAX_SESSION_BYTES = 1024 * 1024
MAX_SESSION_LINE_BYTES = 64 * 1024


class P0Error(RuntimeError):
    """A concrete P0 gate failure."""


@dataclass(frozen=True)
class ProviderBinding:
    run_id: str
    release: str
    model: str
    profile_digest: str
    request_digest: str


class ProviderAdmission:
    """Disposable capacity-one lease and generation fence used by P0 tests."""

    def __init__(self) -> None:
        self.gateway_generation = 1
        self.server_generation = 1
        self._issued: dict[str, tuple[ProviderBinding, int, int]] = {}
        self._used: set[str] = set()
        self._active: str | None = None
        self._closed = False
        self._quarantined = False

    def issue(self, binding: ProviderBinding) -> str:
        if self._closed or self._quarantined or self._active is not None:
            raise P0Error("provider admission is unavailable")
        lease = secrets.token_hex(32)
        self._issued[lease] = (
            binding,
            self.gateway_generation,
            self.server_generation,
        )
        return lease

    def admit(self, lease: str, binding: ProviderBinding) -> None:
        expected = self._issued.pop(lease, None)
        if lease in self._used or expected is None:
            raise P0Error("provider lease is unknown or already consumed")
        if self._closed or self._quarantined or self._active is not None:
            raise P0Error("provider admission is unavailable")
        if expected != (
            binding,
            self.gateway_generation,
            self.server_generation,
        ):
            raise P0Error("provider lease binding or generation changed")
        self._used.add(lease)
        self._active = lease

    def cancel(self, lease: str) -> None:
        if self._active != lease:
            raise P0Error("cannot cancel an inactive provider generation")
        self._closed = True

    def observe_terminal(self, lease: str) -> None:
        if self._active != lease:
            raise P0Error("terminal frame belongs to another generation")
        self._active = None
        self._closed = False
        self._quarantined = False

    def observe_single_slot_idle(self, is_processing: object) -> None:
        if is_processing is not False or self._active is None:
            raise P0Error("provider idle proof is not the exact inactive sole slot")
        self._active = None
        self._closed = False
        self._quarantined = False

    def gateway_restart(self) -> None:
        self.gateway_generation += 1
        self._issued.clear()
        self._quarantined = self._active is not None
        self._closed = self._active is not None

    def server_restart_complete(self) -> None:
        if self._active is None and not self._quarantined:
            raise P0Error("model restart was not required")
        self.server_generation += 1
        self._active = None
        self._closed = False
        self._quarantined = False


class ActiveWorkLedger:
    """Disposable persistent cumulative-time and single-holder proof."""

    def __init__(self, path: Path, cap_ns: int) -> None:
        self.path = path
        self.lock_path = path.with_suffix(".lock")
        self.cap_ns = cap_ns
        with self._exclusive():
            if not path.exists():
                descriptor = os.open(
                    path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                )
                try:
                    os.write(
                        descriptor,
                        _canonical({"cumulative_ns": 0, "holder": None}) + b"\n",
                    )
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)

    @contextlib.contextmanager
    def _exclusive(self) -> Iterable[None]:
        descriptor = os.open(
            self.lock_path,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _read(self) -> dict[str, object]:
        value = json.loads(self.path.read_text())
        if set(value) != {"cumulative_ns", "holder"}:
            raise P0Error("active-work ledger schema changed")
        return value

    def _write(self, value: dict[str, object]) -> None:
        descriptor, raw_temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent
        )
        temporary = Path(raw_temporary)
        try:
            os.fchmod(descriptor, 0o600)
            os.write(descriptor, _canonical(value) + b"\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, self.path)
        directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def acquire(self, holder: str) -> None:
        with self._exclusive():
            value = self._read()
            if value["holder"] is not None:
                raise P0Error("active-work holder is already occupied")
            if int(value["cumulative_ns"]) >= self.cap_ns:
                raise P0Error("cumulative active-work cap is exhausted")
            self._write({"cumulative_ns": value["cumulative_ns"], "holder": holder})

    def release_after_exit(self, holder: str, elapsed_ns: int, child_pid: int) -> None:
        if Path(f"/proc/{child_pid}").exists():
            raise P0Error("cannot release holder before child exit")
        with self._exclusive():
            value = self._read()
            if value["holder"] != holder or elapsed_ns < 0:
                raise P0Error("active-work release identity changed")
            self._write(
                {
                    "cumulative_ns": int(value["cumulative_ns"]) + elapsed_ns,
                    "holder": None,
                }
            )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 120,
) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        tail = result.stdout[-MAX_LOG_TAIL:].decode("utf-8", "replace")
        raise P0Error(f"command failed ({result.returncode}): {argv!r}\n{tail}")
    return result


def _safe_build_env(cache: Path, home: Path) -> dict[str, str]:
    return {
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "NPM_CONFIG_AUDIT": "false",
        "NPM_CONFIG_CACHE": str(cache),
        "NPM_CONFIG_FUND": "false",
        "NPM_CONFIG_IGNORE_SCRIPTS": "true",
        "NPM_CONFIG_REGISTRY": "https://registry.npmjs.org/",
        "NPM_CONFIG_UPDATE_NOTIFIER": "false",
        "PATH": os.environ["PATH"],
    }


def _verify_lock() -> dict[str, Any]:
    lock_path = EXPERIMENT / "package-lock.json"
    lock = json.loads(lock_path.read_text())
    if lock.get("lockfileVersion") != 3:
        raise P0Error("package lock must use lockfile version 3")
    packages = lock.get("packages")
    if not isinstance(packages, dict):
        raise P0Error("package lock packages must be an object")
    for path, metadata in packages.items():
        if not isinstance(metadata, dict):
            raise P0Error(f"invalid lock metadata for {path}")
        resolved = metadata.get("resolved")
        if resolved is not None and not str(resolved).startswith(
            "https://registry.npmjs.org/"
        ):
            raise P0Error(f"non-registry dependency in lock: {path}: {resolved}")
    expected = {
        "node_modules/@earendil-works/pi-agent-core": "0.83.0",
        "node_modules/@earendil-works/pi-ai": "0.83.0",
        "node_modules/@earendil-works/pi-coding-agent": "0.83.0",
        "node_modules/@earendil-works/pi-tui": "0.83.0",
    }
    for path, version in expected.items():
        actual = packages.get(path, {}).get("version")
        if actual != version:
            raise P0Error(f"{path} resolved to {actual}, expected {version}")
    return lock


def _copy_build_sources(destination: Path) -> None:
    for name in ("package.json", "package-lock.json", "tsconfig.json"):
        shutil.copy2(EXPERIMENT / name, destination / name)
    shutil.copytree(EXPERIMENT / "src", destination / "src")
    shutil.copytree(EXPERIMENT / "test", destination / "test")


_NEEDED = re.compile(r"Shared library: \[(?P<name>[^]]+)\]")
_INTERPRETER = re.compile(r"Requesting program interpreter: (?P<path>[^]]+)")


def _elf_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            with path.open("rb") as handle:
                if handle.read(4) == b"\x7fELF":
                    yield path
        except OSError:
            continue


def _readelf(path: Path, *args: str) -> str:
    result = _run(["/usr/bin/readelf", *args, str(path)], timeout=30)
    return result.stdout.decode("utf-8", "replace")


def _runtime_libraries(elfs: Iterable[Path]) -> dict[str, Path]:
    pending = list(elfs)
    seen_files: set[Path] = set()
    libraries: dict[str, Path] = {}
    while pending:
        path = pending.pop()
        resolved = path.resolve()
        if resolved in seen_files:
            continue
        seen_files.add(resolved)
        dynamic = _readelf(resolved, "-d")
        for match in _NEEDED.finditer(dynamic):
            name = match.group("name")
            candidate = Path("/usr/lib") / name
            if not candidate.exists():
                raise P0Error(f"cannot resolve ELF dependency {name} for {path}")
            libraries[name] = candidate.resolve()
            pending.append(candidate)
        program = _readelf(resolved, "-l")
        interpreter = _INTERPRETER.search(program)
        if interpreter:
            loader = Path(interpreter.group("path"))
            if not loader.exists():
                raise P0Error(f"missing ELF interpreter {loader}")
            libraries[loader.name] = loader.resolve()
            pending.append(loader)
    return libraries


def _minimal_node_bwrap(
    node: Path,
    libraries: dict[str, Path],
    build_dir: Path,
    command: list[str],
) -> list[str]:
    args = [
        "bwrap",
        "--unshare-all",
        "--unshare-user",
        "--die-with-parent",
        "--cap-drop",
        "ALL",
        "--ro-bind",
        str(node),
        "/node",
        "--dir",
        "/usr",
        "--dir",
        "/usr/lib",
    ]
    for name, source in sorted(libraries.items()):
        args.extend(["--ro-bind", str(source), f"/usr/lib/{name}"])
    args.extend(
        [
            "--symlink",
            "usr/lib",
            "/lib",
            "--symlink",
            "usr/lib",
            "/lib64",
            "--bind",
            str(build_dir),
            "/build",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/tmp/home",
            "--clearenv",
            "--setenv",
            "HOME",
            "/tmp/home",
            "--setenv",
            "PATH",
            "/node-bin",
            "--chdir",
            "/build",
            "/node",
            *command,
        ]
    )
    return args


def _copy_runtime_tree(build: Path, candidate: Path) -> None:
    if NODE is None:
        raise P0Error("Node is not available on PATH and ASSIST_PI_P0_NODE is unset")
    (candidate / "app").mkdir(parents=True)
    shutil.copytree(build / "dist" / "src", candidate / "app", dirs_exist_ok=True)
    for name in ("package.json", "package-lock.json"):
        shutil.copy2(build / name, candidate / name)
    shutil.copy2(NODE, candidate / "node")

    runtime_install = candidate / ".runtime-install"
    runtime_install.mkdir()
    for name in ("package.json", "package-lock.json"):
        shutil.copy2(build / name, runtime_install / name)
    cache = candidate.parent / "runtime-npm-cache"
    home = candidate.parent / "runtime-home"
    cache.mkdir()
    home.mkdir()
    _run(
        [
            "npm",
            "ci",
            "--ignore-scripts",
            "--omit=dev",
            "--omit=optional",
            "--no-audit",
            "--no-fund",
            "--update-notifier=false",
        ],
        cwd=runtime_install,
        env=_safe_build_env(cache, home),
        timeout=300,
    )
    shutil.move(str(runtime_install / "node_modules"), candidate / "node_modules")
    shutil.rmtree(runtime_install)

    libraries = _runtime_libraries([candidate / "node", *_elf_files(candidate / "node_modules")])
    library_root = candidate / "lib"
    library_root.mkdir()
    for name, source in sorted(libraries.items()):
        shutil.copy2(source, library_root / name)


def _tree_entries(root: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if path.is_symlink():
            entries.append(
                {"mode": mode, "path": relative, "target": os.readlink(path), "type": "symlink"}
            )
        elif path.is_file():
            entries.append(
                {
                    "mode": mode,
                    "path": relative,
                    "sha256": _sha256(path.read_bytes()),
                    "type": "file",
                }
            )
        elif path.is_dir():
            entries.append({"mode": mode, "path": relative, "type": "directory"})
        else:
            raise P0Error(f"unsupported release entry: {path}")
    return entries


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            continue
        mode = stat.S_IMODE(path.stat().st_mode)
        path.chmod(mode & ~0o222)
    root.chmod(0o555)


def _remove_tree(root: Path) -> None:
    for directory, names, files in os.walk(root):
        Path(directory).chmod(0o700)
        for name in names:
            path = Path(directory) / name
            if not path.is_symlink():
                path.chmod(0o700)
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink():
                path.chmod(0o600)
    shutil.rmtree(root)


def build_release() -> Path:
    _verify_lock()
    if NODE is None:
        raise P0Error("Node is not available on PATH and ASSIST_PI_P0_NODE is unset")
    if not NODE.is_file():
        raise P0Error(f"pinned Node binary is missing: {NODE}")
    version = _run([str(NODE), "--version"]).stdout.decode().strip()
    if version != EXPECTED_NODE:
        raise P0Error(f"Node is {version}, expected {EXPECTED_NODE}")
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="build-", dir=BUILD_ROOT))
    try:
        source = work / "source"
        source.mkdir()
        _copy_build_sources(source)
        cache = work / "npm-cache"
        home = work / "home"
        cache.mkdir()
        home.mkdir()
        fetch_log = _run(
            [
                "npm",
                "ci",
                "--ignore-scripts",
                "--omit=optional",
                "--no-audit",
                "--no-fund",
                "--update-notifier=false",
            ],
            cwd=source,
            env=_safe_build_env(cache, home),
            timeout=300,
        ).stdout
        canary = f"P0-BUILD-CANARY-{secrets.token_hex(16)}".encode()
        (home / "secret-canary").write_bytes(canary)
        node_libraries = _runtime_libraries([NODE])
        compile_command = ["/build/node_modules/typescript/bin/tsc", "-p", "/build/tsconfig.json"]
        compile_log = _run(
            _minimal_node_bwrap(NODE, node_libraries, source, compile_command),
            timeout=120,
        ).stdout
        compiled_tests = [
            f"/build/{path.relative_to(source).as_posix()}"
            for path in sorted((source / "dist" / "test").glob("*.test.js"))
        ]
        if not compiled_tests:
            raise P0Error("TypeScript build produced no tests")
        test_log = _run(
            _minimal_node_bwrap(
                NODE,
                node_libraries,
                source,
                ["--test", *compiled_tests],
            ),
            timeout=120,
        ).stdout
        candidate = work / "candidate"
        candidate.mkdir()
        _copy_runtime_tree(source, candidate)
        manifest_log = _run(
            _minimal_node_bwrap(
                NODE,
                node_libraries,
                source,
                ["/build/dist/src/emit-manifest.js"],
            ),
            timeout=30,
        ).stdout
        manifest = json.loads(manifest_log)
        if not isinstance(manifest, list):
            raise P0Error("emitted extension manifest is not an array")
        (candidate / "extension-manifest.json").write_bytes(
            _canonical(manifest) + b"\n"
        )
        runtime_libraries = _runtime_libraries(
            [candidate / "node", *_elf_files(candidate / "node_modules")]
        )
        runtime_smoke_log = _run(
            _minimal_node_bwrap(
                candidate / "node",
                runtime_libraries,
                candidate,
                [
                    "--input-type=module",
                    "--eval",
                    "await import('/build/app/sdk-fixture.js')",
                ],
            ),
            timeout=30,
        ).stdout
        if canary in fetch_log or canary in compile_log or canary in test_log:
            raise P0Error("build canary appeared in build logs")
        if canary in manifest_log or canary in runtime_smoke_log:
            raise P0Error("build canary appeared in release-generation logs")
        for path in candidate.rglob("*"):
            if path.is_file() and canary in path.read_bytes():
                raise P0Error(f"build canary appeared in release: {path}")
        entries = _tree_entries(candidate)
        identity = {
            "contents": entries,
            "node": EXPECTED_NODE,
            "profile": PROFILE,
        }
        digest = _sha256(_canonical(identity))
        (candidate / "contents.json").write_bytes(_canonical(identity) + b"\n")
        (candidate / "release.json").write_bytes(
            _canonical({"digest": digest, "node": EXPECTED_NODE, "profile": PROFILE})
            + b"\n"
        )
        destination_root = ARTIFACT_ROOT / "releases"
        destination_root.mkdir(parents=True, exist_ok=True)
        destination = destination_root / digest
        _make_read_only(candidate)
        if destination.exists():
            _remove_tree(candidate)
        else:
            candidate.chmod(0o755)
            os.replace(candidate, destination)
            destination.chmod(0o555)
        return destination
    finally:
        if work.exists():
            _remove_tree(work)


class JsonLines:
    def __init__(self, connection: socket.socket, maximum: int = MAX_FRAME):
        self._connection = connection
        self._maximum = maximum
        self._buffer = bytearray()

    def read(self) -> dict[str, Any]:
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                raw = bytes(self._buffer[:newline])
                del self._buffer[: newline + 1]
                if not raw:
                    raise P0Error("empty protocol frame")
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise P0Error("protocol frame must be an object")
                return value
            if len(self._buffer) >= self._maximum:
                raise P0Error("protocol frame exceeds bound")
            chunk = self._connection.recv(min(65536, self._maximum - len(self._buffer)))
            if not chunk:
                raise P0Error("protocol connection closed")
            self._buffer.extend(chunk)

    def write(self, value: dict[str, object]) -> None:
        payload = _canonical(value) + b"\n"
        if len(payload) > self._maximum:
            raise P0Error("outbound protocol frame exceeds bound")
        self._connection.sendall(payload)


def _exact(frame: dict[str, Any], fields: set[str]) -> None:
    actual = set(frame)
    if actual != fields:
        raise P0Error(f"frame fields {sorted(actual)} != {sorted(fields)}")


def _peer_pid(connection: socket.socket) -> int:
    raw = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
    return int.from_bytes(raw[:4], sys.byteorder, signed=True)


def _process_cgroup(pid: int) -> str:
    for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            return line[3:]
    raise P0Error(f"process {pid} has no unified cgroup")


def _is_descendant(pid: int, ancestor: int) -> bool:
    current = pid
    for _ in range(32):
        if current == ancestor:
            return True
        status = Path(f"/proc/{current}/status")
        if not status.exists():
            return False
        parent = next(
            (
                int(line.split()[1])
                for line in status.read_text().splitlines()
                if line.startswith("PPid:")
            ),
            0,
        )
        if parent <= 1 or parent == current:
            return False
        current = parent
    return False


def _namespace_pid(pid: int) -> int:
    line = next(
        line
        for line in Path(f"/proc/{pid}/status").read_text().splitlines()
        if line.startswith("NSpid:")
    )
    return int(line.split()[-1])


def _accept_peer(
    listener: socket.socket,
    expected_pid: int,
    expected_cgroup: str,
    *,
    descendant_allowed: bool = False,
) -> tuple[socket.socket, int]:
    listener.settimeout(10)
    connection, _ = listener.accept()
    peer = _peer_pid(connection)
    if peer != expected_pid and not (
        descendant_allowed and _is_descendant(peer, expected_pid)
    ):
        connection.close()
        raise P0Error(f"socket peer pid {peer} != registered runner {expected_pid}")
    if _process_cgroup(peer) != expected_cgroup:
        connection.close()
        raise P0Error("socket peer is outside the registered cgroup")
    os.pidfd_open(peer)
    return connection, peer


@dataclass
class LogTail:
    stream: Any
    tail: bytearray
    thread: threading.Thread


def _drain(stream: Any) -> LogTail:
    tail = bytearray()

    def consume() -> None:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                return
            tail.extend(chunk)
            if len(tail) > MAX_LOG_TAIL:
                del tail[: len(tail) - MAX_LOG_TAIL]

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    return LogTail(stream, tail, thread)


def _socket_listener(run_root: Path, name: str) -> tuple[socket.socket, Path, int]:
    path = run_root / name
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    path.chmod(0o600)
    listener.listen(1)
    descriptor = os.open(path, os.O_PATH | os.O_NOFOLLOW)
    return listener, path, descriptor


def _runtime_bwrap(
    release: Path,
    control_fd: int,
    provider_fd: int,
    session_fd: int,
    status_fd: int,
) -> list[str]:
    args = [
        "bwrap",
        "--unshare-all",
        "--unshare-user",
        "--die-with-parent",
        "--disable-userns",
        "--cap-drop",
        "ALL",
        "--uid",
        "65534",
        "--gid",
        "65534",
        "--ro-bind",
        str(release),
        "/runtime",
        "--dir",
        "/usr",
        "--ro-bind",
        str(release / "lib"),
        "/usr/lib",
        "--symlink",
        "usr/lib",
        "/lib",
        "--symlink",
        "usr/lib",
        "/lib64",
        "--dir",
        "/run",
        "--dir",
        "/run/assist-p0",
        "--ro-bind-fd",
        str(control_fd),
        "/run/assist-p0/control.sock",
        "--ro-bind-fd",
        str(provider_fd),
        "/run/assist-p0/provider.sock",
        "--dir",
        "/session",
        "--bind-fd",
        str(session_fd),
        "/session/session.jsonl",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/tmp/home",
        "--clearenv",
        "--setenv",
        "HOME",
        "/tmp/home",
        "--setenv",
        "PATH",
        "/runtime",
        "--chdir",
        "/tmp",
        "--json-status-fd",
        str(status_fd),
        "/runtime/node",
        "/runtime/app/topology-probe.js",
    ]
    return args


def _read_status_line(descriptor: int) -> tuple[dict[str, Any], LogTail]:
    stream = os.fdopen(descriptor, "rb", closefd=True)
    raw = stream.readline(MAX_FRAME + 1)
    if not raw or len(raw) > MAX_FRAME:
        stream.close()
        raise P0Error("missing or oversized bwrap status")
    value = json.loads(raw)
    if not isinstance(value, dict):
        stream.close()
        raise P0Error("bwrap status is not an object")
    return value, _drain(stream)


def _validate_session(descriptor: int, run_id: str) -> dict[str, object]:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise P0Error("session inode is not a single-link regular file")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise P0Error("session inode mode changed")
    os.lseek(descriptor, 0, os.SEEK_SET)
    data = os.read(descriptor, MAX_SESSION_BYTES + 1)
    if len(data) > MAX_SESSION_BYTES or not data.endswith(b"\n"):
        raise P0Error("session output is oversized or truncated")
    records = []
    for line in data.splitlines():
        if len(line) > MAX_SESSION_LINE_BYTES:
            raise P0Error("session line exceeds bound")
        value = json.loads(line)
        if not isinstance(value, dict) or value.get("type") not in ALLOWED_SESSION_TYPES:
            observed = value.get("type") if isinstance(value, dict) else type(value).__name__
            raise P0Error(f"session contains an unaudited record: {observed!r}")
        records.append(value)
    if not records or records[0].get("type") != "session":
        raise P0Error("Pi session header is missing")
    roles = [
        value.get("message", {}).get("role")
        for value in records
        if value.get("type") == "message" and isinstance(value.get("message"), dict)
    ]
    if roles != ["user", "assistant", "toolResult", "assistant"]:
        raise P0Error(f"unexpected contained SDK session roles: {roles!r}")
    ids = [value.get("id") for value in records[1:]]
    if any(not isinstance(entry_id, str) or not entry_id for entry_id in ids):
        raise P0Error("Pi session contains an invalid entry ID")
    if len(ids) != len(set(ids)):
        raise P0Error("Pi session contains duplicate entry IDs")
    os.fsync(descriptor)
    return {
        "bytes": len(data),
        "record_types": [value["type"] for value in records],
        "roles": roles,
        "sha256": _sha256(data),
    }


def open_session_candidate(path: Path) -> int:
    """Open and validate a retained P0 session without following or replacing it."""
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_NONBLOCK | os.O_NOFOLLOW,
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise P0Error("session candidate is not a regular file")
        if metadata.st_nlink != 1:
            raise P0Error("session candidate has multiple links")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise P0Error("session candidate mode is not 0600")
        if metadata.st_size > MAX_SESSION_BYTES:
            raise P0Error("session candidate exceeds the size bound")
        data = os.pread(descriptor, MAX_SESSION_BYTES + 1, 0)
        if data and not data.endswith(b"\n"):
            raise P0Error("session candidate has a truncated suffix")
        ids: set[str] = set()
        for line in data.splitlines():
            if len(line) > MAX_SESSION_LINE_BYTES:
                raise P0Error("session candidate line exceeds the size bound")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise P0Error("session candidate contains malformed JSON") from error
            if not isinstance(value, dict) or not isinstance(value.get("type"), str):
                raise P0Error("session candidate record is not typed")
            entry_id = value.get("id")
            if entry_id is not None:
                if not isinstance(entry_id, str) or not entry_id:
                    raise P0Error("session candidate entry id is invalid")
                if entry_id in ids:
                    raise P0Error("session candidate contains a duplicate entry id")
                ids.add(entry_id)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def recover_verified_suffix(
    descriptor: int,
    committed_offset: int,
    committed_prefix_sha256: str,
) -> None:
    """Remove only bytes after an exact previously committed prefix."""
    metadata = os.fstat(descriptor)
    if committed_offset < 0 or committed_offset > metadata.st_size:
        raise P0Error("committed session offset is outside the current inode")
    prefix = os.pread(descriptor, committed_offset, 0)
    if len(prefix) != committed_offset or _sha256(prefix) != committed_prefix_sha256:
        raise P0Error("committed session prefix no longer matches")
    os.ftruncate(descriptor, committed_offset)
    os.fsync(descriptor)


def _verify_cgroup(path: str) -> dict[str, str]:
    root = Path("/sys/fs/cgroup") / path.lstrip("/")
    expected_files = ("memory.max", "pids.max", "cpu.max", "memory.oom.group")
    values = {name: (root / name).read_text().strip() for name in expected_files}
    if values["memory.max"] != str(512 * 1024 * 1024):
        raise P0Error(f"unexpected memory.max: {values['memory.max']}")
    if values["pids.max"] != "32":
        raise P0Error(f"unexpected pids.max: {values['pids.max']}")
    if values["cpu.max"].split()[0] == "max":
        raise P0Error("CPU quota is not active")
    if values["memory.oom.group"] != "1":
        raise P0Error("OOM group kill is not active")
    return values


def _validate_runner_census(census: dict[str, Any]) -> dict[str, object]:
    mounts = census.get("mounts")
    if not isinstance(mounts, str):
        raise P0Error("runner mount census is missing")
    mount_points = []
    for line in mounts.splitlines():
        fields = line.split()
        if len(fields) < 6:
            raise P0Error("runner mount census is malformed")
        mount_points.append(fields[4])
    required = {
        "/",
        "/dev",
        "/dev/full",
        "/dev/null",
        "/dev/pts",
        "/dev/random",
        "/dev/tty",
        "/dev/urandom",
        "/dev/zero",
        "/proc",
        "/run/assist-p0/control.sock",
        "/run/assist-p0/provider.sock",
        "/runtime",
        "/session/session.jsonl",
        "/tmp",
        "/usr/lib",
    }
    if set(mount_points) != required:
        raise P0Error(f"runner mount set changed: {sorted(mount_points)}")
    if any(point.startswith(("/home", "/workspace")) for point in mount_points):
        raise P0Error("runner acquired a host workspace or home mount")
    descriptors = census.get("fds")
    if not isinstance(descriptors, dict):
        raise P0Error("runner FD census is missing")
    allowed_prefixes = (
        "/dev/null",
        "<closed>",
        "anon_inode:",
        "pipe:",
        "socket:",
    )
    for target in descriptors.values():
        if not isinstance(target, str) or not target.startswith(allowed_prefixes):
            raise P0Error(f"runner inherited an unaudited descriptor: {target!r}")
    limits = census.get("limits")
    if not isinstance(limits, str) or not re.search(
        r"^Max open files\s+64\s+64\s+files\s*$", limits, re.MULTILINE
    ):
        raise P0Error("runner RLIMIT_NOFILE is not exactly 64")
    return {
        "descriptor_count": len(descriptors),
        "mount_count": len(mount_points),
        "mount_points": sorted(mount_points),
    }


def _validate_sdk_census(census: object) -> dict[str, object]:
    if not isinstance(census, dict) or set(census) != {
        "active_tools",
        "events",
        "payloads",
    }:
        raise P0Error("contained SDK census schema changed")
    expected_tools = ["load_skill", "fixture_workspace_probe"]
    if census["active_tools"] != expected_tools:
        raise P0Error("contained SDK active-tool set changed")
    payloads = census["payloads"]
    if not isinstance(payloads, list) or len(payloads) != 2:
        raise P0Error("contained SDK did not capture both provider boundaries")
    for payload in payloads:
        if not isinstance(payload, dict):
            raise P0Error("contained SDK provider payload is not an object")
        messages = payload.get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or messages[0] != {"role": "system", "content": OWNED_PROMPT}
        ):
            raise P0Error("Assist does not own the exact provider-bound prompt")
        tools = payload.get("tools")
        if not isinstance(tools, list):
            raise P0Error("contained SDK provider schemas are absent")
        names = [
            tool.get("function", {}).get("name")
            for tool in tools
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        ]
        if names != expected_tools:
            raise P0Error("contained SDK provider schemas changed")
        serialized = _canonical(payload)
        if b"Current working directory" in serialized or b"expert coding assistant operating inside pi" in serialized:
            raise P0Error("Pi default prompt bytes reached the provider")
    events = census["events"]
    if not isinstance(events, list):
        raise P0Error("contained SDK event census is absent")
    phases = [event.get("phase") for event in events if isinstance(event, dict)]
    required_phases = {
        "before_agent_start",
        "before_provider_request",
        "message_end",
        "session_start",
        "tool_call",
        "tool_result",
        "turn_end",
    }
    if not required_phases.issubset(phases):
        raise P0Error("contained SDK lifecycle census is incomplete")
    return {
        "event_count": len(events),
        "payload_count": len(payloads),
        "prompt_sha256": _sha256(OWNED_PROMPT.encode()),
        "tools": expected_tools,
    }


def _fake_provider(channel: JsonLines, identity: dict[str, object]) -> dict[str, object]:
    hello = channel.read()
    identity_fields = {
        "gateway_generation",
        "lease",
        "model",
        "nonce",
        "profile_digest",
        "release",
        "request_budget_bytes",
        "response_budget_bytes",
        "run_id",
        "server_generation",
    }
    _exact(hello, {"type", "seq", *identity_fields})
    if hello != {
        "type": "provider_hello",
        "seq": 0,
        **identity,
    }:
        raise P0Error("invalid provider hello")
    channel.write({"type": "provider_ready", "seq": 0})
    request = channel.read()
    _exact(
        request,
        {
            "type",
            "seq",
            *identity_fields,
            "method",
            "path",
            "content_type",
            "body_sha256",
            "body_base64",
        },
    )
    if request["type"] != "provider_request" or request["seq"] != 1:
        raise P0Error("invalid provider request sequence")
    for field in identity_fields:
        if request[field] != identity[field]:
            raise P0Error(f"provider identity mismatch: {field}")
    if request["method"] != "POST" or request["path"] != "/v1/chat/completions":
        raise P0Error("provider destination changed")
    body = base64.b64decode(request["body_base64"], validate=True)
    if _sha256(body) != request["body_sha256"]:
        raise P0Error("provider body digest mismatch")
    payload = json.loads(body)
    if payload != {"model": "topology-fixture", "stream": True, "messages": []}:
        raise P0Error("provider body changed")
    response_body = b'data: {"fixture":"topology-ok"}\n\ndata: [DONE]\n\n'
    channel.write(
        {
            "type": "provider_response_start",
            "seq": 1,
            "status": 200,
            "content_type": "text/event-stream",
        }
    )
    channel.write(
        {
            "type": "provider_response_chunk",
            "seq": 1,
            "chunk_base64": base64.b64encode(response_body).decode(),
        }
    )
    channel.write({"type": "provider_response_end", "seq": 1})
    return {"body_sha256": _sha256(body), "response_sha256": _sha256(response_body)}


def run_topology(release: Path, *, teardown_race: bool = False) -> dict[str, object]:
    release_metadata = json.loads((release / "release.json").read_text())
    release_digest = release_metadata["digest"]
    run_id = secrets.token_hex(16)
    nonce = secrets.token_hex(32)
    lease = secrets.token_hex(32)
    profile_digest = _sha256(_canonical(PROFILE))
    provider_identity = {
        "gateway_generation": 7,
        "lease": lease,
        "model": PROFILE["model"],
        "nonce": nonce,
        "profile_digest": profile_digest,
        "release": release_digest,
        "request_budget_bytes": 4 * 1024 * 1024,
        "response_budget_bytes": 8 * 1024 * 1024,
        "run_id": run_id,
        "server_generation": 11,
    }
    RUNTIME_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    if stat.S_IMODE(RUNTIME_ROOT.stat().st_mode) != 0o700:
        raise P0Error(f"runtime root has unsafe mode: {RUNTIME_ROOT}")
    run_root = Path(tempfile.mkdtemp(prefix="run-", dir=RUNTIME_ROOT))
    run_root.chmod(0o700)
    secret_canary = f"P0-RUNTIME-CANARY-{secrets.token_hex(16)}".encode()
    (run_root / "unmounted-secret-canary").write_bytes(secret_canary)
    control_listener = provider_listener = None
    control_fd = provider_fd = session_path_fd = session_fd = status_read = status_write = -1
    process: subprocess.Popen[bytes] | None = None
    stdout_tail: LogTail | None = None
    stderr_tail: LogTail | None = None
    status_tail: LogTail | None = None
    unit = f"assist-pi-p0-{run_id[:16]}.scope"
    try:
        control_listener, control_path, control_fd = _socket_listener(run_root, "control.sock")
        provider_listener, provider_path, provider_fd = _socket_listener(run_root, "provider.sock")
        session_path = run_root / "session.jsonl"
        session_fd = os.open(
            session_path,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        session_path_fd = os.open(session_path, os.O_PATH | os.O_NOFOLLOW)
        status_read, status_write = os.pipe()
        bwrap = _runtime_bwrap(
            release, control_fd, provider_fd, session_path_fd, status_write
        )
        command = [
            "systemd-run",
            "--user",
            "--scope",
            "--quiet",
            f"--unit={unit}",
            "-p",
            "MemoryMax=512M",
            "-p",
            "CPUQuota=50%",
            "-p",
            "TasksMax=32",
            "-p",
            "OOMPolicy=kill",
            "-p",
            "RuntimeMaxSec=60s",
            "prlimit",
            "--nofile=64:64",
            "--",
            *bwrap,
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(control_fd, provider_fd, session_path_fd, status_write),
            start_new_session=True,
        )
        os.close(status_write)
        status_write = -1
        if process.stdout is None or process.stderr is None:
            raise P0Error("runner logs were not captured")
        stdout_tail = _drain(process.stdout)
        stderr_tail = _drain(process.stderr)
        try:
            status, status_tail = _read_status_line(status_read)
        except P0Error as error:
            process.wait(timeout=5)
            stdout_tail.thread.join(timeout=1)
            stderr_tail.thread.join(timeout=1)
            logs = bytes(stdout_tail.tail + stderr_tail.tail).decode("utf-8", "replace")
            raise P0Error(f"{error}; runner exit={process.returncode}; logs={logs}") from error
        status_read = -1
        child_pid = status.get("child-pid")
        if not isinstance(child_pid, int) or child_pid <= 0:
            raise P0Error(f"bwrap did not report child pid: {status!r}")
        cgroup = _process_cgroup(child_pid)
        if not cgroup.endswith(f"/{unit}"):
            raise P0Error(f"runner entered unexpected cgroup: {cgroup}")
        cgroup_limits = _verify_cgroup(cgroup)
        try:
            control_connection, runner_pid = _accept_peer(
                control_listener,
                child_pid,
                cgroup,
                descendant_allowed=True,
            )
        except (OSError, P0Error) as error:
            logs = bytes(stdout_tail.tail + stderr_tail.tail).decode("utf-8", "replace")
            raise P0Error(f"control connection failed: {error}; logs={logs}") from error
        control_listener.close()
        control_listener = None
        control_path.unlink()
        control = JsonLines(control_connection, 1024 * 1024)
        hello = control.read()
        _exact(hello, {"type", "seq", "pid", "release"})
        if hello != {
            "type": "hello",
            "seq": 0,
            "pid": _namespace_pid(runner_pid),
            "release": release_digest,
        }:
            raise P0Error(f"invalid runner hello: {hello!r}")
        control.write({"type": "challenge", "seq": 0, "mode": (
            "teardown-race" if teardown_race else "topology"
        ), **{
            key: value for key, value in provider_identity.items() if key != "release"
        }})

        try:
            provider_connection, provider_pid = _accept_peer(
                provider_listener, runner_pid, cgroup
            )
        except (OSError, P0Error) as error:
            logs = bytes(stdout_tail.tail + stderr_tail.tail).decode("utf-8", "replace")
            raise P0Error(f"provider connection failed: {error}; logs={logs}") from error
        provider_listener.close()
        provider_listener = None
        provider_path.unlink()
        if provider_pid != runner_pid:
            raise P0Error("provider connection did not come from the registered runner")
        if teardown_race:
            provider = JsonLines(provider_connection)
            provider_hello = provider.read()
            identity_fields = set(provider_identity)
            _exact(provider_hello, {"type", "seq", *identity_fields})
            if provider_hello != {"type": "provider_hello", "seq": 0, **provider_identity}:
                raise P0Error("teardown provider identity changed")
            ready = control.read()
            _exact(ready, {"type", "seq", "nonce", "run_id", "descendant_pid"})
            if (
                ready.get("type") != "race_ready"
                or ready.get("seq") != 1
                or ready.get("nonce") != nonce
                or ready.get("run_id") != run_id
            ):
                raise P0Error("teardown runner readiness changed")
            cgroup_root = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
            host_members = {
                int(value)
                for value in (cgroup_root / "cgroup.procs").read_text().splitlines()
            }
            if runner_pid not in host_members or len(host_members) < 2:
                raise P0Error("teardown race did not create a cgroup-owned descendant")
            subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "kill",
                    "--kill-whom=all",
                    "--signal=TERM",
                    unit,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                subprocess.run(
                    [
                        "systemctl",
                        "--user",
                        "kill",
                        "--kill-whom=all",
                        "--signal=KILL",
                        unit,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                process.wait(timeout=5)
            for connection in (control_connection, provider_connection):
                connection.settimeout(2)
                if connection.recv(1) != b"":
                    raise P0Error("a runner socket survived cgroup termination")
                connection.close()
            stdout_tail.thread.join(timeout=2)
            stderr_tail.thread.join(timeout=2)
            status_tail.thread.join(timeout=2)
            if any(
                tail.thread.is_alive()
                for tail in (stdout_tail, stderr_tail, status_tail)
            ):
                raise P0Error("a teardown pipe remained open after cgroup termination")
            if cgroup_root.exists():
                events = dict(
                    line.split(maxsplit=1)
                    for line in (cgroup_root / "cgroup.events").read_text().splitlines()
                )
                if events.get("populated") != "0":
                    raise P0Error("teardown cgroup remained populated")
            if os.fstat(session_fd).st_size != 0:
                raise P0Error("teardown race unexpectedly wrote the session")
            return {
                "cgroup": cgroup,
                "descendant_count": len(host_members) - 1,
                "pipes_closed": True,
                "provider_socket_closed": True,
                "session_unchanged": True,
                "status": "PASS",
            }
        provider_evidence = _fake_provider(
            JsonLines(provider_connection),
            provider_identity,
        )
        result = control.read()
        _exact(
            result,
            {
                "type",
                "seq",
                "nonce",
                "run_id",
                "release",
                "gateway_status",
                "gateway_body_sha256",
                "replay_denied",
                "workspace_model_tool_denied",
                "sdk_census",
                "direct_model_denied",
                "public_egress_denied",
                "forbidden_paths",
                "session_sha256",
                "census",
            },
        )
        if result["type"] != "topology_result" or result["seq"] != 1:
            raise P0Error("invalid topology result sequence")
        if result["nonce"] != nonce or result["run_id"] != run_id:
            raise P0Error("topology result identity mismatch")
        if result["release"] != release_digest or result["gateway_status"] != 200:
            raise P0Error("topology result release or gateway mismatch")
        if result["replay_denied"] is not True:
            raise P0Error("one-use provider lease was replayed")
        if result["workspace_model_tool_denied"] is not True:
            raise P0Error("a model tool observed the host workspace")
        sdk_evidence = _validate_sdk_census(result["sdk_census"])
        if result["direct_model_denied"] is not True or result["public_egress_denied"] is not True:
            raise P0Error("runner retained IP egress")
        forbidden = result["forbidden_paths"]
        if not isinstance(forbidden, dict) or not forbidden or set(forbidden.values()) != {True}:
            raise P0Error("runner saw a forbidden path")
        census = result["census"]
        if not isinstance(census, dict) or census.get("env") != {
            "HOME": "/tmp/home",
            "PATH": "/runtime",
            "PWD": "/tmp",
        }:
            raise P0Error("runner environment census changed")
        if census.get("uid") != 65534 or census.get("gid") != 65534:
            raise P0Error("runner namespace identity changed")
        census_evidence = _validate_runner_census(census)
        session_evidence = _validate_session(session_fd, run_id)
        control.write({"type": "accepted", "seq": 1})
        control_connection.shutdown(socket.SHUT_RDWR)
        control_connection.close()
        provider_connection.shutdown(socket.SHUT_RDWR)
        provider_connection.close()
        return_code = process.wait(timeout=10)
        if return_code != 0:
            raise P0Error(f"runner exited {return_code}")
        stdout_tail.thread.join(timeout=2)
        stderr_tail.thread.join(timeout=2)
        status_tail.thread.join(timeout=2)
        if (
            stdout_tail.thread.is_alive()
            or stderr_tail.thread.is_alive()
            or status_tail.thread.is_alive()
        ):
            raise P0Error("runner log drainer did not reach EOF")
        if stdout_tail.tail or stderr_tail.tail:
            raise P0Error(
                "runner wrote unexpected logs: "
                + bytes(stdout_tail.tail + stderr_tail.tail).decode("utf-8", "replace")
            )
        secret_scan_targets = (
            _canonical(result),
            os.pread(session_fd, MAX_SESSION_BYTES + 1, 0),
            bytes(stdout_tail.tail),
            bytes(stderr_tail.tail),
            Path(f"/proc/{process.pid}/cmdline").read_bytes()
            if Path(f"/proc/{process.pid}/cmdline").exists()
            else b"",
        )
        if any(secret_canary in target for target in secret_scan_targets):
            raise P0Error("runtime secret canary escaped its unmounted source")
        cgroup_root = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
        if cgroup_root.exists():
            events = dict(
                line.split(maxsplit=1)
                for line in (cgroup_root / "cgroup.events").read_text().splitlines()
            )
            if events.get("populated") != "0":
                raise P0Error("runner cgroup remained populated after child exit")
        return {
            "cgroup": cgroup,
            "cgroup_limits": cgroup_limits,
            "gateway": provider_evidence,
            "release": release_digest,
            "run_id": run_id,
            "runner_census": census,
            "runner_census_evidence": census_evidence,
            "sdk_census_evidence": sdk_evidence,
            "secret_scan": "PASS",
            "session": session_evidence,
            "status": "PASS",
        }
    finally:
        if process is not None and process.poll() is None:
            subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "kill",
                    "--kill-whom=all",
                    "--signal=TERM",
                    unit,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                subprocess.run(
                    [
                        "systemctl",
                        "--user",
                        "kill",
                        "--kill-whom=all",
                        "--signal=KILL",
                        unit,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                process.wait(timeout=5)
        for listener in (control_listener, provider_listener):
            if listener is not None:
                listener.close()
        for descriptor in (
            control_fd,
            provider_fd,
            session_path_fd,
            session_fd,
            status_read,
            status_write,
        ):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        shutil.rmtree(run_root, ignore_errors=True)


def _measure_simplicity(release: Path) -> dict[str, object]:
    sources = [
        path
        for path in EXPERIMENT.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".sh", ".ts"}
        and not any(part in {".artifacts", ".build", "dist", "node_modules"} for part in path.parts)
    ]
    source_lines = {
        path.relative_to(EXPERIMENT).as_posix(): len(path.read_text().splitlines())
        for path in sorted(sources)
    }
    runtime_bytes = sum(path.stat().st_size for path in release.rglob("*") if path.is_file())
    package_root = release / "node_modules"
    top_level_packages = len([path for path in package_root.iterdir() if path.is_dir()])
    return {
        "judgment": "PASS_FOR_P1_ONLY",
        "reason": (
            "The proof needs one existing Python owner, one unprivileged transient "
            "systemd/bwrap/Node runner, and one capacity-one provider gateway. It adds "
            "no custom agent loop, checkpoint scheduler, privileged helper, or product "
            "daemon in P0. P2 must extract only these measured boundaries and reopens "
            "the stop gate if it creates another lifecycle authority."
        ),
        "runtime_bytes": runtime_bytes,
        "runtime_top_level_package_directories": top_level_packages,
        "source_lines": source_lines,
        "source_lines_total": sum(source_lines.values()),
        "runtime_process_roles": [
            "existing Python orchestration/gateway owner",
            "transient systemd scope",
            "bubblewrap namespace launcher",
            "pinned Node/Pi runner",
        ],
        "privilege": "unprivileged user scope; private namespaces; all capabilities dropped",
    }


def _validate_live_report(path: Path) -> dict[str, object]:
    report = json.loads(path.read_text())
    expected_cases = {
        "cancellation",
        "compaction",
        "dependent-sequential",
        "malformed-arguments",
        "one-tool-result",
        "parallel-readonly",
        "persisted-tool-result-continuation",
        "persisted-user-continuation",
        "prompt-schema-census",
        "serialized-mutations",
        "text-thinking",
        "tool-error",
        "unknown-tool",
    }
    if report.get("status") != "PASS" or report.get("model") != PROFILE["model"]:
        raise P0Error("live Qwen report did not pass the exact pinned model")
    cases = report.get("cases")
    if not isinstance(cases, list):
        raise P0Error("live Qwen report cases are missing")
    names = {
        case.get("name")
        for case in cases
        if isinstance(case, dict) and case.get("status") == "PASS"
    }
    if names != expected_cases:
        raise P0Error("live Qwen report does not contain the exact passing matrix")
    census = next(
        case for case in cases
        if isinstance(case, dict) and case.get("name") == "prompt-schema-census"
    )
    detail = census.get("detail")
    if not isinstance(detail, dict) or detail.get("prompt") != OWNED_PROMPT:
        raise P0Error("live Qwen census lost prompt ownership")
    return {
        "case_count": len(expected_cases),
        "model": report["model"],
        "profile": report.get("profile"),
        "report_path": str(path.resolve()),
        "status": "PASS",
    }


def _write_evidence(
    release: Path,
    topology: dict[str, object],
    teardown_race: dict[str, object],
    *,
    live_qwen: dict[str, object] | None = None,
    reproducible_release: bool = False,
) -> Path:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    commit = _run(["git", "rev-parse", "HEAD"], cwd=EXPERIMENT).stdout.decode().strip()
    profile_digest = _sha256(_canonical(PROFILE))
    run_dir = ARTIFACT_ROOT / "runs" / f"{timestamp}-{commit[:12]}-{profile_digest[:12]}"
    run_dir.mkdir(parents=True)
    report = {
        "commit": commit,
        "containment": "PASS",
        "deterministic_sdk_tests": "PASS",
        "live_qwen": live_qwen or "NOT RUN",
        "next_authorized_action": (
            "publish the reviewed P0 result; P1 requires separate authorization"
            if live_qwen is not None and reproducible_release
            else "finish remaining offline P0 gates"
        ),
        "offline": "PASS" if reproducible_release else "INCOMPLETE",
        "p0_status": (
            "GO" if live_qwen is not None and reproducible_release else "INCOMPLETE"
        ),
        "profile": PROFILE,
        "release": release.name,
        "session_preflight_tests": "PASS",
        "reproducible_release": "PASS" if reproducible_release else "NOT RUN",
        "simplicity_judgment": (
            _measure_simplicity(release) if reproducible_release else "NOT RUN"
        ),
        "topology": topology,
        "teardown_race": teardown_race,
    }
    path = run_dir / "report.json"
    path.write_bytes(_canonical(report) + b"\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "topology", "offline", "final"))
    parser.add_argument("--release", type=Path)
    parser.add_argument("--live-report", type=Path)
    arguments = parser.parse_args()
    started = time.monotonic()
    try:
        if arguments.command == "final":
            raise P0Error(
                "local review invalidated this candidate's execution-control and "
                "evidence gates; see docs/2026-08-04-pi-runtime-p0.org"
            )
        if arguments.command == "build":
            print("[build] start", flush=True)
            release = build_release()
            print(f"[build] PASS {time.monotonic() - started:.1f}s {release}", flush=True)
            return 0
        if arguments.command in {"offline", "final"}:
            print("[session-preflight] start", flush=True)
            _run(
                [sys.executable, "-m", "unittest", "-v", "test_pi_runtime_p0.py"],
                cwd=EXPERIMENT / "python",
                timeout=30,
            )
            print(
                f"[session-preflight] PASS {time.monotonic() - started:.1f}s",
                flush=True,
            )
        if arguments.command == "final":
            if arguments.live_report is None:
                raise P0Error("final requires --live-report")
            print("[reproducible-release] build 1/2", flush=True)
            first_release = build_release()
            print("[reproducible-release] build 2/2", flush=True)
            second_release = build_release()
            if first_release.name != second_release.name:
                raise P0Error("two clean builds produced different release digests")
            release = first_release
            print(f"[reproducible-release] PASS digest={release.name}", flush=True)
        else:
            release = arguments.release or build_release()
        print(f"[topology] start release={release.name}", flush=True)
        topology = run_topology(release)
        print(f"[teardown-race] start release={release.name}", flush=True)
        teardown_race = run_topology(release, teardown_race=True)
        live_qwen = (
            _validate_live_report(arguments.live_report)
            if arguments.command == "final" and arguments.live_report is not None
            else None
        )
        report = _write_evidence(
            release,
            topology,
            teardown_race,
            live_qwen=live_qwen,
            reproducible_release=arguments.command == "final",
        )
        print(f"[topology] PASS {time.monotonic() - started:.1f}s", flush=True)
        complete = arguments.command == "final"
        print(f"P0 STATUS: {'GO' if complete else 'INCOMPLETE'}", flush=True)
        print(f"offline deterministic result: {'PASS' if complete else 'INCOMPLETE'}", flush=True)
        print(f"live exact-Qwen result: {'PASS' if complete else 'NOT RUN'}", flush=True)
        print("containment and secret scan: PASS (topology slice)", flush=True)
        print(
            f"simplicity judgment: {'PASS_FOR_P1_ONLY' if complete else 'NOT RUN'}",
            flush=True,
        )
        print(
            "next authorized action: "
            + (
                "publish the reviewed P0 result; P1 requires separate authorization"
                if complete
                else "finish remaining offline P0 gates"
            ),
            flush=True,
        )
        print(f"report: {report}", flush=True)
        return 0
    except (OSError, P0Error, subprocess.SubprocessError, ValueError) as error:
        print(f"P0 STATUS: STOP\n{error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
