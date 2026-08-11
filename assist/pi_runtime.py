"""Host-owned lifecycle for one isolated, visible Pi preview turn."""
from __future__ import annotations

import json
import os
import secrets
import shutil
import socket
import tempfile
import threading
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from requests.exceptions import ReadTimeout

from assist.model_manager import OpenAIConfig, current_model_config
from assist.pi_broker import PiToolBroker
from assist.pi_provider_relay import PiProviderRelay
from assist.sandbox import DockerSandboxBackend
from assist.sandbox_manager import SandboxManager


_IMAGE = "assist-pi-runtime"
_MAX_RESULT_BYTES = 96 * 1024
_MAX_HISTORY_MESSAGES = 32
_MAX_MESSAGE_BYTES = 32 * 1024
_MAX_TURNS = 6
_WALL_TIMEOUT_SECONDS = 900


class PiRuntimeError(RuntimeError):
    """A bounded Pi turn could not safely complete."""


@dataclass(frozen=True)
class PiRuntimeResult:
    """The visible completion payload a host may append to its transcript."""

    reply: str
    turns: int


@dataclass
class _ActivePiTurn:
    """The host-owned cancellation state for one worker lifetime."""

    stopped: threading.Event
    finished: threading.Event


class PiResultSink:
    """Receive one bounded worker result without giving it host-file write access."""

    def __init__(self, control_dir: Path, capability: str) -> None:
        self._capability = capability
        self._path = control_dir / "result.sock"
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(str(self._path))
        os.chmod(self._path, 0o600)
        self._listener.listen(1)
        self._listener.settimeout(_WALL_TIMEOUT_SECONDS)
        self._done = threading.Event()
        self._result: PiRuntimeResult | None = None
        self._error: PiRuntimeError | None = None
        self._thread = threading.Thread(target=self._collect, name="pi-result-sink", daemon=True)
        self._thread.start()

    def _collect(self) -> None:
        try:
            self._result = self._receive()
        except PiRuntimeError as error:
            self._error = error
        finally:
            self._done.set()

    def _receive(self) -> PiRuntimeResult:
        """Accept exactly one close-delimited result while the worker runs."""
        try:
            connection, _ = self._listener.accept()
        except OSError as error:
            raise PiRuntimeError("Pi worker did not produce a result") from error
        with connection:
            connection.settimeout(5)
            chunks: list[bytes] = []
            remaining = _MAX_RESULT_BYTES + 1
            try:
                while remaining:
                    chunk = connection.recv(min(65536, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
            except OSError as error:
                raise PiRuntimeError("Pi worker result is unreadable") from error
        raw = b"".join(chunks)
        if len(raw) > _MAX_RESULT_BYTES:
            raise PiRuntimeError("Pi worker result exceeds its bound")
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PiRuntimeError("Pi worker result is malformed") from error
        if not isinstance(value, dict) or set(value) != {"capability", "status", "reply", "turns"}:
            raise PiRuntimeError("Pi worker result is malformed")
        capability = value["capability"]
        if not isinstance(capability, str) or not secrets.compare_digest(capability, self._capability):
            raise PiRuntimeError("Pi worker result is unauthorized")
        if value["status"] != "completed":
            raise PiRuntimeError("Pi worker did not complete")
        reply, turns = value["reply"], value["turns"]
        if (not isinstance(reply, str) or not reply.strip() or not isinstance(turns, int)
                or isinstance(turns, bool) or not 1 <= turns <= _MAX_TURNS
                or len(reply.encode("utf-8")) > _MAX_RESULT_BYTES):
            raise PiRuntimeError("Pi worker result is invalid")
        return PiRuntimeResult(reply, turns)

    def receive(self) -> PiRuntimeResult:
        """Return the already-drained result only after the worker exited cleanly."""
        if not self._done.wait(5):
            raise PiRuntimeError("Pi worker did not produce a result")
        if self._error is not None:
            raise self._error
        if self._result is None:
            raise PiRuntimeError("Pi worker did not produce a result")
        return self._result

    def close(self) -> None:
        """Close and unlink the per-turn listener after worker reaping."""
        try:
            self._listener.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._listener.close()
        self._path.unlink(missing_ok=True)
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("Pi result receiver did not drain")


class PiRuntimeManager:
    """Start Pi only behind host-owned model and workspace authorities."""

    def __init__(self, sandbox_manager: type[SandboxManager] = SandboxManager) -> None:
        self._sandbox_manager = sandbox_manager
        self._active_lock = threading.Lock()
        self._active: dict[str, _ActivePiTurn] = {}
        self._retired: set[str] = set()

    def _register(self, turn_id: str | None) -> _ActivePiTurn | None:
        if turn_id is None:
            return None
        active = _ActivePiTurn(threading.Event(), threading.Event())
        with self._active_lock:
            if turn_id in self._retired:
                raise PiRuntimeError("Pi thread is no longer available")
            if turn_id in self._active:
                raise PiRuntimeError("Pi turn is already active")
            self._active[turn_id] = active
        return active

    def _finish(self, turn_id: str | None, active: _ActivePiTurn | None) -> None:
        if active is None:
            return
        active.finished.set()
        with self._active_lock:
            if self._active.get(turn_id) is active:
                del self._active[turn_id]

    def stop(self, turn_id: str, *, timeout: float = 20) -> None:
        """Stop one active worker and wait until its authority is torn down."""
        with self._active_lock:
            active = self._active.get(turn_id)
        if active is None:
            return
        active.stopped.set()
        if not active.finished.wait(timeout):
            raise PiRuntimeError("Pi worker did not stop within its bound")

    def retire(self, turn_id: str, *, timeout: float = 20) -> None:
        """Permanently prevent new worker registration before thread deletion."""
        with self._active_lock:
            self._retired.add(turn_id)
        self.stop(turn_id, timeout=timeout)

    @staticmethod
    def _control_dir(work_dir: str) -> Path:
        parent = Path(work_dir).parent
        try:
            return Path(tempfile.mkdtemp(prefix=".pi-turn-", dir=parent))
        except OSError as error:
            raise PiRuntimeError("Pi control directory could not be created") from error

    @staticmethod
    def _write_request(control_dir: Path, value: dict[str, object]) -> None:
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(raw) > 512 * 1024:
            raise PiRuntimeError("Pi request exceeds its bound")
        target = control_dir / "request.json"
        descriptor, temporary = tempfile.mkstemp(prefix=".request-", dir=control_dir)
        try:
            os.fchmod(descriptor, 0o600)
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("Pi request write made no progress")
                view = view[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            os.replace(temporary, target)
            temporary = ""
            directory = os.open(control_dir, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError as error:
            raise PiRuntimeError("Pi request could not be written") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

    @staticmethod
    def _relay_url(config: OpenAIConfig) -> str:
        parsed = urllib.parse.urlsplit(config.url)
        hostname = parsed.hostname
        if hostname == "localhost":
            hostname = "127.0.0.1"
        if hostname not in {"127.0.0.1", "::1"}:
            raise PiRuntimeError("Pi preview requires a local model endpoint")
        host = f"[{hostname}]" if ":" in hostname else hostname
        port = f":{parsed.port}" if parsed.port else ""
        return urllib.parse.urlunsplit((parsed.scheme, host + port, parsed.path, "", ""))

    @staticmethod
    def _history(messages: Iterable[tuple[str, str]]) -> list[dict[str, str]]:
        history = list(messages)[-_MAX_HISTORY_MESSAGES:]
        result: list[dict[str, str]] = []
        for role, text in history:
            if role not in {"user", "assistant"} or not isinstance(text, str):
                raise PiRuntimeError("Pi transcript is invalid")
            if len(text.encode("utf-8")) > _MAX_MESSAGE_BYTES:
                raise PiRuntimeError("Pi transcript exceeds its bound")
            result.append({"role": role, "text": text})
        return result

    @staticmethod
    def _reap_worker(worker: object) -> None:
        """TERM then KILL a still-running worker, and prove it has exited."""
        try:
            worker.stop(timeout=5)
        except Exception:
            pass
        try:
            worker.wait(timeout=5)
            return
        except Exception:
            pass
        worker.kill()
        worker.wait(timeout=5)

    @staticmethod
    def _attempt(errors: list[Exception], operation: Callable[[], None]) -> None:
        try:
            operation()
        except Exception as error:
            errors.append(error)

    def run(self, *, work_dir: str, timezone: str | None, prompt: str,
            history: Iterable[tuple[str, str]], system_prompt: str,
            max_turns: int = _MAX_TURNS, turn_id: str | None = None,
            admitted: Callable[[], bool] | None = None,
            should_yield: Callable[[], bool] | None = None) -> PiRuntimeResult:
        """Run one fresh Pi worker and tear down every authority it used."""
        if (not isinstance(prompt, str) or not isinstance(system_prompt, str)
                or not system_prompt.strip() or not isinstance(max_turns, int)
                or isinstance(max_turns, bool) or not 1 <= max_turns <= _MAX_TURNS):
            raise PiRuntimeError("Pi request is invalid")
        if len(prompt.encode("utf-8")) > _MAX_MESSAGE_BYTES:
            raise PiRuntimeError("Pi prompt exceeds its bound")
        config = current_model_config()
        control_dir = self._control_dir(work_dir)
        active = self._register(turn_id)
        sandbox: DockerSandboxBackend | None = None
        broker: PiToolBroker | None = None
        relay: PiProviderRelay | None = None
        result_sink: PiResultSink | None = None
        worker = None
        worker_exited = False
        try:
            broker_capability = secrets.token_urlsafe(32)
            provider_capability = secrets.token_urlsafe(32)
            result_capability = secrets.token_urlsafe(32)
            result_sink = PiResultSink(control_dir, result_capability)
            self._write_request(control_dir, {
                "version": 1, "prompt": prompt, "history": self._history(history),
                "model": config.model, "systemPrompt": system_prompt,
                "brokerCapability": broker_capability,
                "providerCapability": provider_capability,
                "resultCapability": result_capability, "maxTurns": max_turns,
            })
            sandbox = self._sandbox_manager.get_pi_sandbox_backend(work_dir, timezone)
            if sandbox is None:
                raise PiRuntimeError("Pi workspace sandbox is unavailable")
            broker = PiToolBroker(sandbox, control_dir, broker_capability)
            relay = PiProviderRelay(
                control_dir, self._relay_url(config), config.api_key,
                config.model, provider_capability)
            broker.start()
            relay.start()
            client = self._sandbox_manager._get_docker_client()
            worker = client.containers.run(
                _IMAGE, detach=True, remove=True, network_mode="none", read_only=True,
                user=f"{os.getuid()}:{os.getgid()}", cap_drop=["ALL"],
                security_opt=["no-new-privileges"], pids_limit=64,
                mem_limit="768m", nano_cpus=1_000_000_000,
                tmpfs={"/tmp": "rw,noexec,nosuid,nodev,size=16m"},
                volumes={str(control_dir): {"bind": "/run/pi", "mode": "ro"}},
                environment={}, labels={"assist.pi-runtime": "true"},
            )
            deadline = time.monotonic() + _WALL_TIMEOUT_SECONDS
            while True:
                if ((active is not None and active.stopped.is_set())
                        or (admitted is not None and not admitted())):
                    raise PiRuntimeError("Pi preview was stopped")
                if should_yield is not None and should_yield():
                    raise PiRuntimeError("Pi preview yielded to waiting work")
                if time.monotonic() >= deadline:
                    raise PiRuntimeError("Pi worker timed out")
                try:
                    status = worker.wait(timeout=1)
                    break
                except ReadTimeout:
                    continue
            worker_exited = True
            if not isinstance(status, dict) or status.get("StatusCode") != 0:
                raise PiRuntimeError("Pi worker exited unsuccessfully")
            return result_sink.receive()
        except PiRuntimeError:
            raise
        except Exception as error:
            raise PiRuntimeError("Pi worker could not complete") from error
        finally:
            teardown_errors: list[Exception] = []
            if broker is not None:
                self._attempt(teardown_errors, broker.stop_admission)
            if relay is not None:
                self._attempt(teardown_errors, relay.stop_admission)
            worker_reaped = worker is None or worker_exited
            if worker is not None and not worker_exited:
                try:
                    self._reap_worker(worker)
                    worker_reaped = True
                except Exception as error:
                    teardown_errors.append(error)
            if worker_reaped:
                if sandbox is not None:
                    self._attempt(teardown_errors, lambda: self._sandbox_manager.cleanup(
                        work_dir, expected_container=sandbox.container))
                if broker is not None:
                    self._attempt(teardown_errors, broker.close)
                if relay is not None:
                    self._attempt(teardown_errors, relay.close)
                if result_sink is not None:
                    self._attempt(teardown_errors, result_sink.close)
                self._attempt(teardown_errors, lambda: shutil.rmtree(control_dir))
            try:
                if teardown_errors:
                    raise PiRuntimeError("Pi worker teardown failed") from teardown_errors[0]
            finally:
                self._finish(turn_id, active)
