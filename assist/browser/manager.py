"""Host control of one bounded browser sidecar for one visible web Run."""
from __future__ import annotations

import io
import hashlib
import ipaddress
import json
import logging
import math
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

from assist.egress.client_map import (
    ClientRecord, configured_directory, forget_client,
    prune_absent_browser_clients, record_client)
from assist.egress import runtime_state
from assist.sandbox_manager import (
    BROWSER_NETWORK, EGRESS_PROXY_NAME, SandboxManager, _load_egress_allowlist,
    _network_identity)
from assist.run_service import RunService

logger = logging.getLogger(__name__)
BROWSER_IMAGE = "assist-browser"
SECCOMP_PROFILE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "dockerfiles", "browser-seccomp.json"))
MAX_DOWNLOAD = 20 * 1024 * 1024


class BrowserUnavailable(Exception):
    pass


@dataclass(frozen=True)
class BrowserUserRequest:
    event_id: str
    work_id: str
    text: str
    admission_sequence: int = 0


def _url_host(url: str) -> str:
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
    return parsed.hostname.lower()


def _internal_host(host: str) -> bool:
    if host == "host.docker.internal":
        return True
    try:
        return not ipaddress.ip_address(host).is_global
    except ValueError:
        return False


def _user_requested_host(message: str | None, host: str) -> bool:
    """Fail closed unless a real-user event starts with an affirmative visit command."""
    if not message:
        return False
    # Reject quoted text and prohibitions before the host. False negatives
    # require a fresh exact-host request; false positives grant LAN access.
    if re.search(r"['\"`]|\b(?:not|never|avoid|without|don.t|cannot|can.t)\b",
                 message, re.I):
        return False
    pattern = (r"^\s*(?:yes,\s*)?(?:(?:please|can you|could you|would you)\s+)?"
               r"(?:visit|open|browse|go to|look at|inspect|check|show)\s+"
               r"(?:(?:me\s+)?(?:the\s+)?"
               r"(?:site|website|page|dashboard)\s+(?:(?:at|on)\s+)?)?"
               r"(?:https?://)?" + re.escape(host)
               + r"(?=$|[\s/:?#!,])")
    return bool(re.search(pattern, message, re.I))


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
                            raise BrowserUnavailable("browser sidecar command failed") from error
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
            if process.wait(timeout=max(0.1, deadline - time.monotonic())):
                raise BrowserUnavailable("browser sidecar command failed")
            return output.getvalue()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            process.stdout.close()


def _docker_exec(container_id: str, payload: bytes) -> bytes:
    return _bounded_cli(
        ["docker", "exec", "-i", container_id, "python",
         "/opt/assist/browser_runner.py", "call"],
        payload=payload, limit=65536)


def _docker_download(container_id: str, source: str) -> bytes:
    """Read bounded raw bytes from the sidecar's read-only tmpfs export."""
    if not source.startswith("/downloads/"):
        raise BrowserUnavailable("download source is outside browser downloads")
    return _bounded_cli(
        ["docker", "exec", container_id, "python",
         "/opt/assist/browser_runner.py", "export", source],
        limit=MAX_DOWNLOAD)


def _kill_container(container_id: str) -> None:
    try:
        killed = subprocess.run(["docker", "kill", container_id],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=5, check=False)
        if killed.returncode:
            logger.warning("sidecar kill rc=%s stderr=%r", killed.returncode,
                           killed.stderr[-200:])
    except (OSError, subprocess.TimeoutExpired):
        logger.warning("browser sidecar kill could not complete")


def _kill_container_confirmed(container_id: str) -> None:
    """Stop one exact sidecar; uncertain teardown cannot release user admission."""
    try:
        subprocess.run(["docker", "kill", container_id],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BrowserUnavailable("browser sidecar teardown is unconfirmed") from error
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            inspected = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", container_id],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2,
                check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise BrowserUnavailable("browser sidecar teardown is unconfirmed") from error
        if inspected.returncode == 0 and inspected.stdout.strip() == b"false":
            return
        if (inspected.returncode != 0 and
                (b"no such object" in inspected.stderr.lower()
                 or b"no such container" in inspected.stderr.lower())):
            return
        time.sleep(0.05)
    logger.warning("sidecar teardown unresolved: rc=%s stdout=%r stderr=%r",
                   inspected.returncode, inspected.stdout[-200:],
                   inspected.stderr[-200:])
    raise BrowserUnavailable("browser sidecar teardown is unconfirmed")


def _launch_sidecar_direct(request: dict, *, ensure_proxy: bool = True):
    """Docker SDK work performed only in a disposable host worker process."""
    client = SandboxManager._get_docker_client()
    BrowserManager.prune_absent_clients(request["map_dir"])
    if ensure_proxy:
        SandboxManager._ensure_egress_proxy_running_direct(client)
    network = client.networks.get(BROWSER_NETWORK)
    network_id, _ = _network_identity(network, isolated=True)
    runtime_state.assert_admissible("network", network_id)
    proxy = client.containers.get(EGRESS_PROXY_NAME)
    proxy.reload()
    runtime_state.assert_admissible("proxy", proxy.id)
    attached = proxy.attrs.get("NetworkSettings", {}).get("Networks", {})
    if attached.get(BROWSER_NETWORK, {}).get("NetworkID") != network_id:
        raise BrowserUnavailable("egress proxy is not on the browser network")
    if request["remaining"] <= 0:
        raise BrowserUnavailable("browser Run deadline expired")
    with open(SECCOMP_PROFILE, encoding="utf-8") as profile:
        seccomp = profile.read()
    container = client.containers.run(
        BROWSER_IMAGE, detach=True, network=BROWSER_NETWORK,
        name=f"assist-browser-{uuid4().hex[:16]}",
        user="10001:10001", read_only=True, remove=True,
        mem_limit="1g", pids_limit=128, nano_cpus=1_000_000_000,
        shm_size="64m", cap_drop=["ALL"],
        security_opt=[f"seccomp={seccomp}", "no-new-privileges"],
        tmpfs={
            "/run": "rw,nosuid,size=1048576,uid=10001,gid=10001",
            "/tmp": "rw,nosuid,size=33554432,uid=10001,gid=10001",
            "/home/browser": "rw,nosuid,size=134217728,uid=10001,gid=10001",
            "/downloads": "rw,nosuid,size=33554432,uid=10001,gid=10001",
        },
        environment={"BROWSER_SESSION_TOKEN": request["token"],
                     "BROWSER_TTL_SECONDS": str(request["remaining"])},
        labels={"assist.browser": "true", "assist.browser-root": request["label_root"],
                "assist.browser-run": request["run_id"],
                "assist.browser-owner-pid": str(request["owner_pid"]),
                "assist.browser-owner-start": request["owner_start"]},
    )
    container.reload()
    ip = container.attrs["NetworkSettings"]["Networks"][BROWSER_NETWORK]["IPAddress"]
    if not ip:
        raise BrowserUnavailable("browser sidecar has no network identity")
    return {"generation": container.id, "ip": ip}, container


def _launch_sidecar_bounded(request: dict, timeout: float) -> dict:
    payload = json.dumps(request, separators=(",", ":")).encode()
    output = _bounded_cli(
        [sys.executable, "-m", "assist.browser.startup_worker"],
        payload=payload, limit=4096, timeout=timeout)
    result = json.loads(output)
    if (not isinstance(result, dict) or
            not isinstance(result.get("generation"), str) or
            not isinstance(result.get("ip"), str)):
        raise BrowserUnavailable("browser startup returned invalid identity")
    return result


@dataclass
class _ContainerIdentity:
    container: object
    map_dir: str
    ip: str
    generation: str
    mode: str
    internal_host: str | None


class BrowserSession:
    def __init__(self, thread_id: str, run_id: str, work_dir: str,
                 threads_root: str, user_request: BrowserUserRequest | None,
                 *, work_id: str | None = None):
        self.thread_id = thread_id
        self.run_id = run_id
        self.work_dir = work_dir
        self.threads_root = threads_root
        self.user_request = user_request
        self.work_id = work_id
        if (user_request is not None and work_id is not None
                and user_request.work_id != work_id):
            raise BrowserUnavailable("browser request belongs to another work")
        self.token = uuid4().hex
        self.deadline = None
        self.identity: _ContainerIdentity | None = None
        self.startup_uncertain = False
        self.closed = False
        self._lock = threading.RLock()

    def _start(self, mode: str, internal_host: str | None):
        if self.startup_uncertain:
            raise BrowserUnavailable("browser startup outcome is unconfirmed for this Run")
        map_dir = configured_directory()
        if map_dir is None:
            raise BrowserUnavailable("browser client-map directory is not configured")
        now = time.monotonic()
        remaining = math.ceil(self.deadline - now)
        if remaining <= 0:
            raise BrowserUnavailable("browser Run deadline expired")
        label_root = hashlib.sha256(
            os.path.realpath(self.threads_root).encode()).hexdigest()[:20]
        request = {"token": self.token, "remaining": remaining,
                   "label_root": label_root, "run_id": self.run_id,
                   "owner_pid": os.getpid(), "owner_start": _owner_start(),
                   "map_dir": map_dir}
        try:
            result = _launch_sidecar_bounded(
                request, timeout=min(20, max(0.1, self.deadline - now)))
        except Exception:
            # Docker may finish a create after its worker is cancelled. A
            # late sidecar is proxy-denied, but this Run must not accumulate
            # another uncertain sidecar before its fixed PID 1 TTL expires.
            self.startup_uncertain = True
            raise
        generation, ip = result["generation"], result["ip"]
        try:
            if time.monotonic() >= self.deadline:
                raise BrowserUnavailable("browser Run deadline expired")
            record_client(map_dir, ip, ClientRecord(
                self.thread_id, generation, "browser", mode, internal_host))
        except Exception:
            self.startup_uncertain = True
            _kill_container(generation)
            raise
        self.identity = _ContainerIdentity(
            None, map_dir, ip, generation, mode, internal_host)

    def _stop(self):
        identity = self.identity
        if identity is None:
            return
        forgotten_error = None
        try:
            forget_client(identity.map_dir, identity.ip, identity.generation)
        except Exception as error:
            forgotten_error = error
        _kill_container_confirmed(identity.generation)
        if forgotten_error is not None:
            raise BrowserUnavailable("browser attribution teardown is unconfirmed") from forgotten_error
        self.identity = None

    def close(self):
        with self._lock:
            self.closed = True
            self._stop()

    def _ensure_mode(self, url: str):
        host = _url_host(url)
        allowlist = frozenset(_load_egress_allowlist())
        if _internal_host(host):
            if host not in allowlist:
                raise BrowserUnavailable("internal host is outside the base allowlist")
            if (self.user_request is None
                    or not _user_requested_host(self.user_request.text, host)):
                raise BrowserUnavailable(
                    "internal browsing needs a fresh user request naming this exact host")
            mode, internal_host = "internal", host
        else:
            mode, internal_host = "public", None
        if (self.identity is not None
                and (self.identity.mode, self.identity.internal_host)
                != (mode, internal_host)):
            self._stop()
        if self.identity is None:
            self._start(mode, internal_host)

    def _direct_authority_state(self, bound_sequence):
        try:
            runs = RunService(self.threads_root).list(self.thread_id)
        except Exception as error:
            raise BrowserUnavailable("direct-user authority is unavailable") from error
        if not runs and self.user_request and self.user_request.admission_sequence:
            raise BrowserUnavailable("direct-user authority is unavailable")
        direct = [run for run in runs if run.user_event_id == run.id
                  and run.origin is None and run.sender is None]
        latest = max((run.admission_sequence for run in direct), default=0)
        bound = next((run for run in direct
                      if run.admission_sequence == bound_sequence), None)
        return latest, bound

    def _fence_internal_command(self, operation, args):
        internal = (self.identity is not None and self.identity.mode == "internal")
        if operation == "open":
            internal = internal or _internal_host(_url_host(args["url"]))
        if internal:
            bound = self.user_request.admission_sequence if self.user_request else -1
            latest, source = self._direct_authority_state(bound)
            if (latest > bound or (bound > 0 and
                                   (source is None or source.status == "revocation_pending"))):
                raise BrowserUnavailable("newer user message revoked internal browsing")

    def revoke_internal(self):
        """Under the thread gate, destroy internal state before event promotion."""
        with self._lock:
            self.user_request = None
            if self.identity is None or self.identity.mode != "internal":
                return False
            self._stop()
            return True

    def rebind_user_request(self, request: BrowserUserRequest):
        """Bind the same tool session to one promoted direct owner event."""
        with self._lock:
            if self.closed or (self.identity and self.identity.mode == "internal"):
                raise BrowserUnavailable("old internal browser is not closed")
            self.user_request = request

    def command(self, operation: str, **args):
        with BrowserManager.bounded_thread_gate(self.thread_id), self._lock:
            if self.closed:
                raise BrowserUnavailable("browser Run has ended")
            self._fence_internal_command(operation, args)
            if operation == "open":
                if self.deadline is None:
                    self.deadline = time.monotonic() + 240
                self._ensure_mode(args["url"])
            elif self.identity is None:
                raise BrowserUnavailable("open a page first")
            if self.deadline is not None and time.monotonic() >= self.deadline:
                self._stop()
                raise BrowserUnavailable("browser Run deadline expired")
            identity = self.identity
            command = json.dumps({"session": self.token, "operation": operation,
                                  "args": args}, ensure_ascii=False).encode()
            if len(command) > 32768:
                raise BrowserUnavailable("browser command exceeds limit")
            try:
                parsed = json.loads(_docker_exec(identity.generation, command))
                if not isinstance(parsed, dict):
                    raise BrowserUnavailable("browser sidecar returned invalid data")
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
            data = _docker_download(identity.generation, source)
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
    def prune_absent_clients(cls, map_dir: str):
        def live_generations():
            output = _bounded_cli(
                ["docker", "ps", "--no-trunc", "--filter", "label=assist.browser=true",
                 "--format", "{{.ID}}"], limit=65536, timeout=5)
            return set(output.decode("ascii").splitlines())

        return prune_absent_browser_clients(map_dir, live_generations)

    @classmethod
    def register(cls, session: BrowserSession):
        with cls.bounded_thread_gate(session.thread_id), cls._lock:
            if not os.path.isdir(os.path.join(session.threads_root, session.thread_id)):
                raise BrowserUnavailable("browser thread no longer exists")
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
                current.close()
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
        """Reap this deployment's absent-owner sidecars with bounded Docker CLI."""
        label_root = hashlib.sha256(
            os.path.realpath(threads_root).encode()).hexdigest()[:20]
        try:
            output = _bounded_cli(
                ["docker", "ps", "--no-trunc", "--filter", "label=assist.browser=true",
                 "--filter", f"label=assist.browser-root={label_root}",
                 "--format", "{{.ID}}"], limit=65536, timeout=5)
        except Exception as error:
            logger.warning("browser orphan scan skipped: %s", error)
            return
        try:
            map_dir = configured_directory()
        except (OSError, RuntimeError):
            map_dir = None
        for generation in output.decode("ascii").splitlines():
            try:
                labels = json.loads(_bounded_cli(
                    ["docker", "inspect", "--format", "{{json .Config.Labels}}",
                     generation], limit=8192, timeout=5))
            except Exception:
                logger.warning("browser orphan labels unavailable", exc_info=True)
                continue
            try:
                pid = int(labels["assist.browser-owner-pid"])
                with open(f"/proc/{pid}/stat", encoding="ascii") as stream:
                    start = stream.read().rsplit(") ", 1)[1].split()[19]
                if start == labels.get("assist.browser-owner-start"):
                    continue
            except (ValueError, KeyError, FileNotFoundError, ProcessLookupError):
                pass
            except (OSError, TypeError):
                logger.warning("browser orphan owner check unavailable", exc_info=True)
                continue
            try:
                if map_dir is not None:
                    networks = json.loads(_bounded_cli(
                        ["docker", "inspect", "--format",
                         "{{json .NetworkSettings.Networks}}", generation],
                        limit=16384, timeout=5))
                    ip = (networks.get(BROWSER_NETWORK) or {}).get("IPAddress")
                    if ip:
                        forget_client(map_dir, ip, generation)
                _kill_container_confirmed(generation)
            except Exception:
                logger.warning("browser orphan reap failed", exc_info=True)
        if map_dir is not None:
            try:
                cls.prune_absent_clients(map_dir)
            except Exception:
                logger.warning("browser client-map prune failed", exc_info=True)


def browser_tools(session: BrowserSession):
    """The web main sees typed operations, never Docker or general code execution."""
    def invoke(operation, **args):
        try:
            return json.dumps(session.command(operation, **args), ensure_ascii=False)
        except BrowserUnavailable as error:
            return f"Browser unavailable: {error}"
        except Exception as error:
            logger.warning("browser tool failed: %s", type(error).__name__)
            return "Browser operation failed; this Run's browser state was discarded."

    def browser_open(url: str) -> str:
        """Open an HTTP(S) website in this Run's isolated browser and observe its rendered page."""
        return invoke("open", url=url)

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

    def browser_probe(host: str, port: int) -> str:
        """Probe the proxy reason for a host whose browser request already failed."""
        return invoke("probe", host=host, port=port)

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

    return [browser_open, browser_observe, browser_act,
            browser_wait, browser_probe, browser_save_download]
