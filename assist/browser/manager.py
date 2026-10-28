"""Host control of one bounded browser worker in a visible web turn."""
from __future__ import annotations

import io
import hashlib
import json
import logging
import os
import re
import selectors
import subprocess
import sys
import threading
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from urllib.parse import urlsplit
from uuid import uuid4

from assist.browser import authority
from assist.egress.client_map import (
    ClientRecord, browser_records, configured_directory, disarm_sandbox_browser,
    forget_client, prune_absent_browser_clients)
from assist.sandbox_manager import (
    BROWSER_NETWORK, EGRESS_NETWORK, EGRESS_PROXY_NAME, SandboxManager,
    confirm_generation_stopped)
from assist.run_service import RunService

logger = logging.getLogger(__name__)
MAX_DOWNLOAD = 20 * 1024 * 1024


class BrowserUnavailable(Exception):
    pass


@dataclass(frozen=True)
class BrowserUserRequest:
    event_id: str
    work_id: str
    text: str
    admission_sequence: int = 0


def _url_target(url: str) -> tuple[str, int]:
    if not isinstance(url, str) or len(url) > 4096:
        raise BrowserUnavailable("invalid browser URL")
    try:
        parsed = urlsplit(url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or (parsed.port is not None and not 1 <= parsed.port <= 65535)):
            raise ValueError
    except ValueError as error:
        raise BrowserUnavailable("use an HTTP(S) URL without userinfo") from error
    return parsed.hostname.lower(), parsed.port or (443 if parsed.scheme == "https" else 80)


def _owner_start() -> str:
    with open("/proc/self/stat", encoding="ascii") as stream:
        return stream.read().rsplit(") ", 1)[1].split()[19]


def _bounded_cli(argv: list[str], *, payload: bytes = b"",
                 limit: int, timeout: float = 20) -> bytes:
    """Cancel a stuck Docker CLI process and cap its output before allocation."""
    process = subprocess.Popen(argv, stdin=subprocess.PIPE if payload else None,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + timeout
    output = io.BytesIO()
    sent = 0
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        if payload:
            os.set_blocking(process.stdin.fileno(), False)
            selector.register(process.stdin, selectors.EVENT_WRITE)
        try:
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BrowserUnavailable("browser transport timed out")
                events = selector.select(remaining)
                if not events:
                    raise BrowserUnavailable("browser transport timed out")
                for key, _ in events:
                    if key.fileobj is process.stdin:
                        try:
                            sent += os.write(process.stdin.fileno(), payload[sent:sent + 8192])
                        except BlockingIOError:
                            continue
                        except BrokenPipeError as error:
                            raise BrowserUnavailable("browser command failed") from error
                        if sent == len(payload):
                            selector.unregister(process.stdin)
                            process.stdin.close()
                    else:
                        chunk = os.read(process.stdout.fileno(), 65536)
                        if not chunk:
                            selector.unregister(process.stdout)
                        elif output.tell() + len(chunk) > limit:
                            raise BrowserUnavailable("browser response exceeds limit")
                        else:
                            output.write(chunk)
            try:
                status = process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired as error:
                raise BrowserUnavailable("browser transport timed out") from error
            if status:
                raise BrowserUnavailable("browser command failed")
            return output.getvalue()
        finally:
            try:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
            finally:
                if process.stdin and not process.stdin.closed:
                    process.stdin.close()
                process.stdout.close()


def _docker_exec(container_id: str, payload: bytes, timeout: float = 20,
                 *, in_sandbox: bool = False) -> bytes:
    executable = "/opt/assist-browser/bin/python" if in_sandbox else "python"
    runtime = (["-e", "BROWSER_RUNTIME_DIR=/run/assist-browser"]
               if in_sandbox else [])
    return _bounded_cli(
        ["docker", "exec", "-i", "--user", "10001:10001", *runtime,
         container_id,
         executable,
         "/opt/assist/browser_runner.py", "call"],
        payload=payload, limit=65536, timeout=timeout)


def _docker_download(container_id: str, source: str, timeout: float = 20,
                     *, in_sandbox: bool = False) -> bytes:
    """Read bounded raw bytes from the browser's private tmpfs export."""
    prefix = ("/run/assist-browser/downloads/" if in_sandbox else "/downloads/")
    if not source.startswith(prefix):
        raise BrowserUnavailable("download source is outside browser downloads")
    executable = "/opt/assist-browser/bin/python" if in_sandbox else "python"
    runtime = (["-e", "BROWSER_DOWNLOAD_DIR=/run/assist-browser/downloads"]
               if in_sandbox else [])
    return _bounded_cli(
        ["docker", "exec", "--user", "10001:10001", *runtime, container_id,
         executable,
         "/opt/assist/browser_runner.py", "export", source],
        limit=MAX_DOWNLOAD, timeout=timeout)


def _kill_container_confirmed(container_id: str) -> None:
    """Compatibility delegate for the shared sandbox stop proof."""
    try:
        confirm_generation_stopped(container_id)
    except RuntimeError as error:
        raise BrowserUnavailable("browser generation teardown is unconfirmed") from error


@dataclass
class _ContainerIdentity:
    map_dir: str
    ip: str
    generation: str


def _sandbox_generations(threads_root: str,
                         thread_id: str | None = None) -> set[str]:
    """Scan exact-root Docker sandboxes even when their thread dir is gone."""
    root = os.path.realpath(threads_root) + os.sep
    if thread_id is not None and (thread_id in {"", ".", ".."}
                                  or any(c in thread_id for c in "/\\\0")):
        raise ValueError("invalid browser thread ID")
    work_dir = (os.path.join(root, thread_id, "domain")
                if thread_id is not None else None)
    output = _bounded_cli(
        ["docker", "ps", "--all", "--no-trunc",
         "--filter", "label=assist.sandbox=true", "--format", "{{.ID}}"],
        limit=65536, timeout=5)
    generations = output.decode("ascii").splitlines()
    if not generations:
        return set()
    mounts = _bounded_cli(
        ["docker", "inspect", "--format", "{{json .Mounts}}", *generations],
        limit=262144, timeout=5).decode("utf-8").splitlines()
    if len(mounts) != len(generations):
        raise BrowserUnavailable("sandbox generation scan is incomplete")
    found = set()
    for generation, raw in zip(generations, mounts, strict=True):
        entries = json.loads(raw)
        if not isinstance(entries, list):
            raise BrowserUnavailable("sandbox generation scan is malformed")
        if any(entry.get("Destination") == "/workspace"
               and (os.path.realpath(entry.get("Source", "")) == work_dir
                    if work_dir is not None else
                    os.path.realpath(entry.get("Source", "")).startswith(root))
               for entry in entries if isinstance(entry, dict)):
            found.add(generation)
    return found


class BrowserSession:
    def __init__(self, thread_id: str, run_id: str, work_dir: str,
                 threads_root: str, user_request: BrowserUserRequest | None,
                 *, work_id: str | None = None,
                 sandbox_generation: str | None = None,
                 run_service: RunService | None = None):
        self.thread_id = thread_id
        self.run_id = run_id
        self.work_dir = work_dir
        self.sandbox_generation = sandbox_generation
        self.threads_root = threads_root
        self.user_request = user_request
        self.work_id = work_id
        self.run_service = run_service or RunService(threads_root)
        if (user_request is not None and work_id is not None
                and user_request.work_id != work_id):
            raise BrowserUnavailable("browser request belongs to another work")
        self.token = uuid4().hex
        self.identity: _ContainerIdentity | None = None
        self.startup_uncertain = False
        self.closed = False
        self._lock = threading.RLock()

    def _start(self):
        if self.startup_uncertain:
            raise BrowserUnavailable("browser startup outcome is unconfirmed for this Run")
        sandbox = SandboxManager.current_container(self.work_dir)
        client_identity = SandboxManager._egress_client_ips.get(self.work_dir)
        if sandbox is None or client_identity is None:
            raise BrowserUnavailable("browser turn sandbox is unavailable")
        map_dir, ip, generation = client_identity
        if (sandbox.id != generation or map_dir != configured_directory()
                or (self.sandbox_generation is not None
                    and generation != self.sandbox_generation)):
            raise BrowserUnavailable("browser turn sandbox identity changed")
        with authority.fence(self.threads_root, self.thread_id) as state:
            state.add_generation(self.run_id, generation)
        request = {
            "token": self.token, "run_id": self.run_id,
            "thread_id": self.thread_id, "threads_root": self.threads_root,
            "work_dir": self.work_dir, "map_dir": map_dir, "ip": ip,
            "generation": generation,
        }
        try:
            result = json.loads(_bounded_cli(
                [sys.executable, "-m", "assist.browser.turn_runtime"],
                payload=json.dumps(request, separators=(",", ":")).encode(),
                limit=4096, timeout=20))
            if (result != {"generation": generation, "ip": ip,
                           "map_dir": map_dir}):
                raise BrowserUnavailable("browser startup returned invalid identity")
            from assist.browser import storage
            try:
                prior_state = storage.load(self.threads_root, self.thread_id)
            except (OSError, ValueError):
                # A damaged private snapshot loses continuity, not access to
                # the browser. Valid snapshots go only to the worker, never the model.
                logger.warning("browser auth state could not be restored")
                prior_state = None
            if prior_state is not None:
                command = json.dumps({"session": self.token,
                                      "operation": "load_storage",
                                      "args": {"state": prior_state}}).encode()
                loaded = json.loads(_docker_exec(
                    generation, command, timeout=10,
                    in_sandbox=True))
                if loaded.get("result") != {"loaded": True}:
                    raise BrowserUnavailable("browser auth state could not be loaded")
        except Exception as error:
            self.startup_uncertain = True
            self._stop_identity(_ContainerIdentity(map_dir, ip, generation))
            with authority.fence(self.threads_root, self.thread_id) as state:
                state.clear(self.run_id)
            raise BrowserUnavailable("browser turn startup failed") from error
        self.identity = _ContainerIdentity(map_dir, ip, generation)

    def _stop_identity(self, identity: _ContainerIdentity) -> None:
        try:
            disarm_sandbox_browser(identity.map_dir, identity.ip,
                                   identity.generation)
        except Exception:
            # A confirmed kill still closes any existing proxy tunnel and
            # prevents further browser traffic despite a failed map update.
            logger.error("browser map disarm failed; proving exact stop", exc_info=True)
        _kill_container_confirmed(identity.generation)
        try:
            forget_client(identity.map_dir, identity.ip, identity.generation)
        except Exception as error:
            raise BrowserUnavailable("browser attribution teardown is unconfirmed") from error
        with authority.fence(self.threads_root, self.thread_id) as state:
            if state.lease is not None:
                state.remove_generation(self.run_id, identity.generation)

    def _stop(self):
        identity = self.identity
        if identity is None:
            return
        self._stop_identity(identity)
        self.identity = None

    def _latest_direct_sequence(self) -> int:
        runs = self.run_service.list(self.thread_id)
        if any(run.status == "revocation_pending" for run in runs):
            raise BrowserUnavailable("newer user message revoked browser startup")
        return max((run.admission_sequence for run in runs
                    if run.user_event_id == run.id), default=0)

    def _begin_lease(self) -> None:
        with authority.fence(self.threads_root, self.thread_id) as state:
            sequence = self._latest_direct_sequence()
            if (self.user_request is not None
                    and self.user_request.admission_sequence != sequence):
                raise BrowserUnavailable("newer user message revoked browser startup")
            state.begin(self.run_id, sequence)

    def _fence_command(self) -> None:
        sequence = self._latest_direct_sequence()
        with authority.fence(self.threads_root, self.thread_id) as state:
            lease = state.lease
            if (lease is None or lease["owner_run_id"] != self.run_id
                    or lease["sequence"] != sequence
                    or self.sandbox_generation not in lease["generations"]):
                raise BrowserUnavailable("browser turn generation changed")

    def close(self):
        with self._lock:
            self.closed = True
            try:
                self._capture_storage()
            finally:
                self._stop()

    def _capture_storage(self):
        if self.identity is None:
            return
        from assist.browser import storage
        try:
            command = json.dumps({"session": self.token,
                                  "operation": "storage_state", "args": {}}).encode()
            response = json.loads(_docker_exec(
                self.identity.generation, command,
                timeout=5, in_sandbox=True))
            if "result" in response and response["result"] is not None:
                storage.save(self.threads_root, self.thread_id, self.run_id,
                             self.identity.generation, response["result"])
        except Exception:
            logger.warning("browser auth state was not saved for this turn")
            storage.discard(self.threads_root, self.thread_id, self.run_id,
                            self.identity.generation)

    def disarm_for_held_event(self) -> None:
        """Deny new browser proxy connections, then ask its worker to close."""
        with self._lock:
            self.user_request = None
            identity = self.identity
            if identity is None:
                return
            disarm_sandbox_browser(identity.map_dir, identity.ip,
                                   identity.generation)
            try:
                self._capture_storage()
            finally:
                try:
                    _docker_exec(identity.generation, json.dumps({
                        "session": self.token, "operation": "revoke", "args": {}
                    }).encode(), timeout=5, in_sandbox=True)
                except Exception:
                    logger.warning(
                        "browser worker revocation incomplete; sandbox stop owed")

    def revoke_for_promotion(self):
        """Under the thread gate, destroy old state before promotion or cancel cleanup."""
        with self._lock:
            self.user_request = None
            reset = self.identity is not None
            if reset:
                self._stop()
            return reset

    def rebind_user_request(self, request: BrowserUserRequest):
        """Bind the same tool session to one promoted direct owner event."""
        with self._lock:
            if self.closed or request.work_id != self.work_id:
                raise BrowserUnavailable("browser session cannot adopt this request")
            self.user_request = request

    def command(self, operation: str, **args):
        with BrowserManager.bounded_thread_gate(self.thread_id), self._lock:
            if self.closed:
                raise BrowserUnavailable("browser Run has ended")
            if self.startup_uncertain:
                raise BrowserUnavailable(
                    "browser startup outcome is unconfirmed for this Run")
            self._fence_command()
            if operation in {"open", "preflight"}:
                _url_target(args["url"])
                if self.identity is None:
                    self._begin_lease()
                    self._start()
            elif self.identity is None:
                raise BrowserUnavailable("open a page first")
            identity = self.identity
            command = json.dumps({"session": self.token, "operation": operation,
                                  "args": args}, ensure_ascii=False).encode()
            if len(command) > 32768:
                raise BrowserUnavailable("browser command exceeds limit")
            try:
                parsed = json.loads(_docker_exec(
                    identity.generation, command,
                    timeout=20, in_sandbox=True))
                if not isinstance(parsed, dict):
                    raise BrowserUnavailable("browser worker returned invalid data")
                return parsed
            except Exception:
                self._stop()
                raise

    def save_download(self, download_id: str, filename: str | None = None):
        with BrowserManager.bounded_thread_gate(self.thread_id), self._lock:
            return self._save_download_locked(download_id, filename)

    def _save_download_locked(self, download_id: str, filename: str | None):
        response = self.command("download_info", download_id=download_id)
        if "error" in response:
            return response
        info = response["result"]
        source = info["path"]
        name = filename or info["name"]
        if (not isinstance(name, str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]{0,179}", name)
                or name.lower() in {"agents.md", "claude.md", "skill.md"}):
            raise BrowserUnavailable("download filename must be a safe basename")
        identity = self.identity
        try:
            data = _docker_download(
                identity.generation, source, timeout=20,
                in_sandbox=True)
        except Exception:
            self._stop()
            raise
        if len(data) != info["size"]:
            raise BrowserUnavailable("download byte count changed")
        downloads = os.path.join(self.work_dir, "downloads")
        try:
            os.mkdir(downloads, 0o700)
        except FileExistsError:
            pass
        dirfd = os.open(downloads, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        temporary = f".download-{uuid4().hex}"
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                         0o600, dir_fd=dirfd)
            try:
                with os.fdopen(fd, "wb") as output:
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
                os.link(temporary, name, src_dir_fd=dirfd, dst_dir_fd=dirfd,
                        follow_symlinks=False)
            finally:
                os.unlink(temporary, dir_fd=dirfd)
        finally:
            os.close(dirfd)
        return {"path": f"/workspace/downloads/{name}", "bytes": len(data)}


class BrowserManager:
    _lock = threading.Lock()
    _sessions: dict[str, BrowserSession] = {}
    _gate_lock = threading.Lock()
    _gates = weakref.WeakValueDictionary()
    _reconcile_lock = threading.Lock()
    _reconciled_roots: set[str] = set()
    _reconcile_timers: dict[str, threading.Timer] = {}

    @classmethod
    def thread_gate(cls, thread_id: str):
        with cls._gate_lock:
            gate = cls._gates.get(thread_id)
            if gate is None:
                gate = threading.RLock()
                cls._gates[thread_id] = gate
            return gate

    @classmethod
    @contextmanager
    def bounded_thread_gate(cls, thread_id: str):
        gate = cls.thread_gate(thread_id)
        if not gate.acquire(timeout=20):
            raise BrowserUnavailable("browser thread gate timed out")
        try:
            yield
        finally:
            gate.release()

    @classmethod
    def current_session(cls, thread_id: str) -> BrowserSession | None:
        with cls._lock:
            return cls._sessions.get(thread_id)

    @classmethod
    def confirm_generation_stopped(cls, generation: str) -> None:
        _kill_container_confirmed(generation)

    @classmethod
    def reconcile_startup(cls, threads_root: str) -> bool:
        """Sweep orphans off the model slot before enabling browser tools."""
        try:
            root = os.path.realpath(threads_root)
            if root not in cls._reconciled_roots:
                if not cls._reconcile_lock.acquire(timeout=5):
                    return False
                try:
                    if root not in cls._reconciled_roots:
                        cls.reap_orphans(threads_root)
                        cls._reconciled_roots.add(root)
                finally:
                    cls._reconcile_lock.release()
            return True
        except Exception:
            return False

    @classmethod
    def schedule_reconciliation(cls, threads_root: str) -> None:
        """Keep one daemon retry wake for an incomplete startup sweep."""
        root = os.path.realpath(threads_root)
        with cls._reconcile_lock:
            if root in cls._reconciled_roots or root in cls._reconcile_timers:
                return
            timer = threading.Timer(10, cls._retry_reconciliation, args=(root,))
            timer.daemon = True
            cls._reconcile_timers[root] = timer
            timer.start()

    @classmethod
    def _retry_reconciliation(cls, root: str) -> None:
        with cls._reconcile_lock:
            cls._reconcile_timers.pop(root, None)
        if not cls.reconcile_startup(root):
            cls.schedule_reconciliation(root)

    @classmethod
    def ready_for_browser(cls, threads_root: str, thread_id: str,
                          owner_run_id: str | None = None,
                          generation: str | None = None) -> bool:
        """A browser tool needs global recovery and exact thread readiness."""
        if os.path.realpath(threads_root) not in cls._reconciled_roots:
            return False
        try:
            map_dir = configured_directory()
            if map_dir is None:
                return False
            with authority.fence(threads_root, thread_id) as state:
                lease = state.lease
                admitted = (lease is None if owner_run_id is None else
                            lease is not None
                            and lease["owner_run_id"] == owner_run_id
                            and generation in lease["generations"])
                return (state.covered and admitted
                        and not browser_records(map_dir, thread_id))
        except (OSError, RuntimeError, ValueError, TimeoutError):
            return False

    @classmethod
    def confirm_owner_stopped(cls, threads_root: str, thread_id: str,
                              owner_run_id: str | None,
                              *, before_replacement: bool = False) -> bool:
        """Positive proof; failure retains a held or cancelled reset obligation."""
        try:
            map_dir = configured_directory()
        except (OSError, RuntimeError):
            map_dir = None
        directory = os.path.join(threads_root, thread_id)
        exists = os.path.isdir(directory) and not os.path.islink(directory)
        lease = None
        covered = False
        if exists:
            with authority.fence(threads_root, thread_id) as state:
                lease, covered = state.lease, state.covered
                if lease is not None:
                    if (owner_run_id is not None
                            and owner_run_id != lease["owner_run_id"]):
                        raise BrowserUnavailable("browser owner changed during reset")
                    owner_run_id = lease["owner_run_id"]
        records = browser_records(map_dir, thread_id) if map_dir else {}
        for ip, record in records.items():
            if record.kind == "sandbox" and record.browser_armed:
                disarm_sandbox_browser(map_dir, ip, record.generation)
        generations = set(lease["generations"] if lease else [])
        generations.update(record.generation for record in records.values())
        # A held event from an active Run must leave its ordinary shell alive.
        # A replacement turn instead proves the workspace empty even when a
        # pre-journal generation survived a crashed host worker.
        if before_replacement or lease is not None or records or not exists or not covered:
            generations.update(_sandbox_generations(threads_root, thread_id))
        label_root = hashlib.sha256(
            os.path.realpath(threads_root).encode()).hexdigest()[:20]
        sidecar_scan = [
            "docker", "ps", "--all", "--no-trunc",
            "--filter", "label=assist.browser=true",
            "--filter", f"label=assist.browser-root={label_root}"]
        if owner_run_id:
            sidecar_scan.extend(["--filter", f"label=assist.browser-run={owner_run_id}"])
        if owner_run_id or lease is not None or not exists or not covered:
            sidecar_scan.extend(["--format", "{{.ID}}"])
            output = _bounded_cli(sidecar_scan, limit=65536, timeout=5)
            sidecars = output.decode("ascii").splitlines()
            if not owner_run_id and sidecars:
                raise BrowserUnavailable(
                    "unattributed browser sidecar prevents exact stop proof")
            generations.update(sidecars)
        for generation in sorted(generations):
            _kill_container_confirmed(generation)
        if map_dir is None:
            if generations:
                SandboxManager._ensure_egress_proxy_bounded()
        else:
            for ip, record in records.items():
                forget_client(map_dir, ip, record.generation)
            if browser_records(map_dir, thread_id):
                raise BrowserUnavailable("browser attribution teardown is unconfirmed")
        if exists:
            with authority.fence(threads_root, thread_id) as state:
                if owner_run_id and state.lease is not None:
                    state.clear_reconciled(owner_run_id)
                if not state.covered:
                    state.mark_covered()
        return bool(generations)

    @classmethod
    def prune_absent_clients(cls, map_dir: str):
        def running_endpoints():
            endpoints = set()
            for label, name in (("assist.browser=true", BROWSER_NETWORK),
                                ("assist.sandbox=true", EGRESS_NETWORK)):
                output = _bounded_cli(
                    ["docker", "ps", "--no-trunc", "--filter", f"label={label}",
                     "--format", "{{.ID}}"], limit=65536, timeout=5)
                generations = output.decode("ascii").splitlines()
                if not generations:
                    continue
                details = _bounded_cli(
                    ["docker", "inspect", "--format",
                     "{{json .NetworkSettings.Networks}}", *generations],
                    limit=262144, timeout=5).decode("utf-8").splitlines()
                if len(details) != len(generations):
                    raise BrowserUnavailable("browser endpoint scan is incomplete")
                for generation, item in zip(generations, details, strict=True):
                    networks = json.loads(item)
                    if not isinstance(networks, dict):
                        raise BrowserUnavailable("browser endpoint scan is malformed")
                    network = networks.get(name)
                    if network is not None:
                        if not isinstance(network, dict) or not network.get("IPAddress"):
                            raise BrowserUnavailable("browser endpoint scan is malformed")
                        endpoints.add((generation, network["IPAddress"]))
            return endpoints

        return prune_absent_browser_clients(map_dir, running_endpoints)

    @classmethod
    def register(cls, session: BrowserSession):
        with cls.bounded_thread_gate(session.thread_id):
            if not os.path.isdir(os.path.join(session.threads_root, session.thread_id)):
                raise BrowserUnavailable("browser thread no longer exists")
            if not cls.ready_for_browser(
                    session.threads_root, session.thread_id,
                    session.run_id, session.sandbox_generation):
                raise BrowserUnavailable("browser startup reconciliation is incomplete")
            with cls._lock:
                if session.thread_id in cls._sessions:
                    raise BrowserUnavailable("thread already has a browser Run")
                cls._sessions[session.thread_id] = session

    @classmethod
    def cleanup(cls, thread_id: str, expected: BrowserSession | None = None):
        with cls.bounded_thread_gate(thread_id):
            with cls._lock:
                current = cls._sessions.get(thread_id)
                if expected is not None and current is not expected:
                    return
            if current:
                if os.path.isdir(os.path.join(current.threads_root, thread_id)):
                    current.close()
                else:
                    current.closed = True
                    cls.confirm_owner_stopped(current.threads_root, thread_id,
                                              current.run_id)
                with cls._lock:
                    if cls._sessions.get(thread_id) is current:
                        cls._sessions.pop(thread_id, None)

    @classmethod
    def cleanup_all(cls):
        with cls._lock:
            sessions = list(cls._sessions.values())
        for session in sessions:
            try:
                cls.cleanup(session.thread_id, session)
            except Exception:
                logger.warning("browser session teardown remains unconfirmed",
                               exc_info=True)

    @classmethod
    def reap_orphans(cls, threads_root: str):
        """Disarm old browser access and reconcile managed turn ownership."""
        map_dir = configured_directory()
        if map_dir is not None:
            for thread_id in os.listdir(threads_root):
                directory = os.path.join(threads_root, thread_id)
                if not os.path.isdir(directory) or os.path.islink(directory):
                    continue
                for ip, record in browser_records(map_dir, thread_id).items():
                    if record.kind == "sandbox" and record.browser_armed:
                        disarm_sandbox_browser(map_dir, ip, record.generation)
        label_root = hashlib.sha256(
            os.path.realpath(threads_root).encode()).hexdigest()[:20]
        output = _bounded_cli(
            ["docker", "ps", "--no-trunc", "--filter", "label=assist.browser=true",
             "--filter", f"label=assist.browser-root={label_root}",
             "--format", "{{.ID}}"], limit=65536, timeout=5)
        for generation in output.decode("ascii").splitlines():
            labels = json.loads(_bounded_cli(
                ["docker", "inspect", "--format", "{{json .Config.Labels}}",
                 generation], limit=8192, timeout=5))
            if (isinstance(labels, dict)
                    and labels.get("assist.browser-owner-pid") == str(os.getpid())
                    and labels.get("assist.browser-owner-start") == _owner_start()):
                continue
            _kill_container_confirmed(generation)
        if map_dir is not None:
            cls.prune_absent_clients(map_dir)
        for thread_id in os.listdir(threads_root):
            directory = os.path.join(threads_root, thread_id)
            if not os.path.isdir(directory) or os.path.islink(directory):
                continue
            with authority.fence(threads_root, thread_id) as state:
                owner = ((state.lease or {}).get("owner_run_id"))
                covered = state.covered
            if owner or (map_dir is not None and browser_records(map_dir, thread_id)):
                cls.confirm_owner_stopped(threads_root, thread_id, owner)
            elif not covered:
                with authority.fence(threads_root, thread_id) as state:
                    state.mark_covered()


def browser_tools(session: BrowserSession):
    """The web main sees typed operations, never Docker or general code execution."""
    def invoke(operation, **args):
        try:
            return json.dumps(session.command(operation, **args), ensure_ascii=False)
        except BrowserUnavailable as error:
            return f"Browser unavailable: {error}"
        except Exception as error:
            logger.warning("browser tool failed: %s", type(error).__name__)
            return ("Browser operation failed; this turn's sandbox ended. "
                    "Use a new turn for further sandbox tools.")

    def browser_open(url: str, reuse_page_id: str = "") -> str:
        """Open a website, optionally navigating one observed live page ID in place."""
        return invoke("open", url=url, reuse_page_id=reuse_page_id or None)

    def browser_close(page_id: str) -> str:
        """Close one observed live page or popup, releasing its page slot."""
        return invoke("close", page_id=page_id)

    def browser_observe(page_id: str) -> str:
        """Refresh the rendered observation of one open page or popup by its page ID."""
        return invoke("observe", page_id=page_id)

    def browser_act(page_id: str, snapshot_id: str, action: str,
                    target: dict) -> str:
        """Click or fill one unambiguous target from the named page observation."""
        return invoke("act", page_id=page_id, snapshot_id=snapshot_id,
                      action=action, target=target)

    def browser_wait(page_id: str, role: str, name: str) -> str:
        """Wait up to ten seconds for a named semantic element on this page."""
        return invoke("wait", page_id=page_id, role=role, name=name)

    def browser_preflight(url: str, page_id: str = "") -> str:
        """Inspect current policy and bounded observed page origins without navigating."""
        return invoke("preflight", url=url,
                      **({"page_id": page_id} if page_id else {}))

    def browser_save_download(download_id: str, filename: str = "") -> str:
        """Save one completed download under /workspace/downloads without replacing a file."""
        try:
            return json.dumps(session.save_download(download_id, filename or None))
        except BrowserUnavailable as error:
            return f"Download not saved: {error}"
        except FileExistsError:
            return "Download not saved: that filename already exists."
        except Exception as error:
            logger.warning("browser download failed: %s", type(error).__name__)
            return "Download not saved: browser transfer failed."

    return [browser_open, browser_close, browser_observe, browser_act,
            browser_wait, browser_preflight, browser_save_download]
