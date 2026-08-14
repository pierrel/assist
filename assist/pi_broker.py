"""Authenticated, bounded Pi-to-workspace tool broker.

The Pi worker has no workspace mount or shell.  Its four coding tools make one
request per Unix-socket connection to this Resource Access object, which alone
holds the selected turn's ``DockerSandboxBackend``.
"""
from __future__ import annotations

import base64
import hmac
import json
import os
import posixpath
import shlex
import socket
import stat
import threading
import time
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict

from assist.sandbox import DockerSandboxBackend


_VERSION = 1
_MAX_FRAME_BYTES = 512 * 1024
_MAX_PATH_BYTES = 4096
_MAX_CONTENT_BYTES = 96 * 1024
_MAX_COMMAND_BYTES = 32 * 1024
_MAX_TIMEOUT_SECONDS = 120
_MAX_REQUESTS = 32
_WORKSPACE = "/workspace"


class PiBrokerError(ValueError):
    """A worker request is unsafe, malformed, or not authorized."""


class _BrokerRequest(TypedDict, total=False):
    version: int
    id: int
    capability: str
    operation: Literal["access", "bash", "mkdir", "read", "write"]
    path: str
    mode: Literal["read", "write"]
    content: str
    command: str
    cwd: str
    timeout: int


class PiToolBroker:
    """Serve Pi coding-tool requests against exactly one Docker sandbox."""

    def __init__(self, backend: DockerSandboxBackend, control_dir: str | Path,
                 capability: str,
                 trace_start: Callable[[str], object] | None = None,
                 trace_settle: Callable[[object, bool], None] | None = None) -> None:
        self._backend = backend
        self._control_dir = Path(control_dir)
        self._socket_path = self._control_dir / "broker.sock"
        self._capability = capability
        self._trace_start = trace_start
        self._trace_settle = trace_settle
        self._listener: socket.socket | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._requests = 0
        self._active_requests = 0
        self._request_gate = threading.Condition()
        self._request_slot = threading.BoundedSemaphore(1)

    @property
    def socket_path(self) -> Path:
        """The manager-provided private Unix-socket path."""
        return self._socket_path

    def start(self) -> None:
        """Start serving only after validating the manager-owned control dir."""
        self._validate_control_dir()
        if self._listener is not None:
            raise RuntimeError("Pi tool broker is already running")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(self._socket_path))
        os.chmod(self._socket_path, 0o600)
        listener.listen(4)
        listener.settimeout(0.1)
        self._listener = listener
        self._thread = threading.Thread(target=self._serve, name="pi-tool-broker",
                                        daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Finish teardown after the manager has killed this turn's sandbox."""
        self.stop_admission()
        deadline = time.monotonic() + 5
        with self._request_gate:
            while self._active_requests:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("Pi tool broker did not drain")
                self._request_gate.wait(remaining)
        try:
            self._socket_path.unlink()
        except FileNotFoundError:
            pass

    def stop_admission(self) -> None:
        """Fence new tool calls before the manager kills the selected sandbox."""
        self._stop.set()
        if self._listener is not None:
            self._listener.close()
            self._listener = None
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None

    def _validate_control_dir(self) -> None:
        metadata = os.lstat(self._control_dir)
        if (not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid()
                or metadata.st_mode & 0o077):
            raise PiBrokerError("Pi control directory is unsafe")

    def _serve(self) -> None:
        assert self._listener is not None
        while not self._stop.is_set():
            try:
                connection, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            if not self._request_slot.acquire(blocking=False):
                connection.close()
                continue
            threading.Thread(target=self._serve_connection, args=(connection,),
                             name="pi-tool-request", daemon=True).start()

    def _serve_connection(self, connection: socket.socket) -> None:
        admitted = False
        trace_operation: object | None = None
        completed = False
        try:
            with connection:
                connection.settimeout(5)
                request_id = 0
                try:
                    raw = self._read_frame(connection)
                    request = self._parse_request(raw)
                    request_id = request["id"]
                    self._admit_request()
                    admitted = True
                    name = self._trace_name(request)
                    if name is not None:
                        trace_operation = self._start_trace(name)
                    value = self._dispatch(request)
                    completed = True
                    response: dict[str, Any] = {
                        "version": _VERSION, "id": request_id, "ok": True, "value": value,
                    }
                except Exception:
                    response = {
                        "version": _VERSION, "id": request_id, "ok": False,
                        "error": "Pi workspace operation failed",
                    }
                self._write_frame(connection, response)
        finally:
            self._settle_trace(trace_operation, completed)
            if admitted:
                self._finish_request()
            self._request_slot.release()

    @staticmethod
    def _trace_name(request: _BrokerRequest) -> str | None:
        """Map a fully admitted closed operation to a redacted display label."""
        operation = request["operation"]
        if operation == "access":
            mode = request.get("mode")
            return "read" if mode == "read" else "write" if mode == "write" else None
        return {"mkdir": "write", "read": "read", "write": "write", "bash": "bash"}[operation]

    def _start_trace(self, name: str) -> object | None:
        """Keep optional observability from changing a permitted tool operation."""
        if self._trace_start is None:
            return None
        try:
            return self._trace_start(name)
        except Exception:
            return None

    def _settle_trace(self, operation: object | None, completed: bool) -> None:
        """Contain an activity-recorder failure after the tool operation ends."""
        if operation is None or self._trace_settle is None:
            return
        try:
            self._trace_settle(operation, completed)
        except Exception:
            pass

    def _admit_request(self) -> None:
        with self._request_gate:
            if self._stop.is_set() or self._requests >= _MAX_REQUESTS:
                raise PiBrokerError("Pi tool request is unavailable")
            self._requests += 1
            self._active_requests += 1

    def _finish_request(self) -> None:
        with self._request_gate:
            if self._active_requests:
                self._active_requests -= 1
                self._request_gate.notify_all()

    @staticmethod
    def _read_frame(connection: socket.socket) -> bytes:
        chunks: list[bytes] = []
        remaining = _MAX_FRAME_BYTES + 1
        while remaining:
            chunk = connection.recv(min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > _MAX_FRAME_BYTES or not raw.endswith(b"\n") or raw.count(b"\n") != 1:
            raise PiBrokerError("Pi tool frame is invalid")
        return raw

    @staticmethod
    def _write_frame(connection: socket.socket, value: dict[str, Any]) -> None:
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8") + b"\n"
        if len(raw) > _MAX_FRAME_BYTES:
            raw = b'{"version":1,"id":0,"ok":false,"error":"Pi workspace operation failed"}\n'
        connection.sendall(raw)

    def _parse_request(self, raw: bytes) -> _BrokerRequest:
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PiBrokerError("Pi tool request is invalid") from error
        if not isinstance(value, dict):
            raise PiBrokerError("Pi tool request is invalid")
        request = value
        if (request.get("version") != _VERSION or not isinstance(request.get("id"), int)
                or not 0 < request["id"] <= 2**31
                or not isinstance(request.get("capability"), str)
                or not hmac.compare_digest(request["capability"], self._capability)
                or request.get("operation") not in {"access", "bash", "mkdir", "read", "write"}):
            raise PiBrokerError("Pi tool request is invalid")
        expected = {
            "access": {"version", "id", "capability", "operation", "path", "mode"},
            "bash": {"version", "id", "capability", "operation", "command", "cwd", "timeout"},
            "mkdir": {"version", "id", "capability", "operation", "path"},
            "read": {"version", "id", "capability", "operation", "path"},
            "write": {"version", "id", "capability", "operation", "path", "content"},
        }[request["operation"]]
        if set(request) != expected:
            raise PiBrokerError("Pi tool request has an invalid shape")
        return request  # type: ignore[return-value]

    @staticmethod
    def _workspace_path(value: object) -> str:
        if not isinstance(value, str) or "\0" in value:
            raise PiBrokerError("Pi tool path is invalid")
        try:
            if len(value.encode("utf-8")) > _MAX_PATH_BYTES:
                raise PiBrokerError("Pi tool path exceeds its bound")
        except UnicodeEncodeError as error:
            raise PiBrokerError("Pi tool path is invalid") from error
        normalized = posixpath.normpath(value)
        if normalized != _WORKSPACE and not normalized.startswith(f"{_WORKSPACE}/"):
            raise PiBrokerError("Pi tool path escapes the workspace")
        return normalized

    def _dispatch(self, request: _BrokerRequest) -> Any:
        operation = request["operation"]
        if operation == "access":
            path = self._workspace_path(request.get("path"))
            mode = request.get("mode")
            if mode not in {"read", "write"}:
                raise PiBrokerError("Pi access mode is invalid")
            self._require_success(f"test -{mode[0]} -- {shlex.quote(path)}")
            return None
        if operation == "read":
            path = self._workspace_path(request.get("path"))
            output = self._require_success(
                f"test -f -- {shlex.quote(path)} && "
                f"test $(wc -c < {shlex.quote(path)}) -le {_MAX_CONTENT_BYTES} && "
                f"cat -- {shlex.quote(path)}"
            )
            if "\ufffd" in output or len(output.encode("utf-8")) > _MAX_CONTENT_BYTES:
                raise PiBrokerError("Pi read is invalid")
            return base64.b64encode(output.encode("utf-8")).decode("ascii")
        if operation == "mkdir":
            path = self._workspace_path(request.get("path"))
            self._require_success(f"mkdir -p -- {shlex.quote(path)}")
            return None
        if operation == "write":
            path = self._workspace_path(request.get("path"))
            content = request.get("content")
            if not isinstance(content, str):
                raise PiBrokerError("Pi write is invalid")
            try:
                if len(content.encode("utf-8")) > _MAX_CONTENT_BYTES:
                    raise PiBrokerError("Pi write exceeds its bound")
            except UnicodeEncodeError as error:
                raise PiBrokerError("Pi write is invalid") from error
            result = self._backend.write(path, content)
            if getattr(result, "error", None):
                raise PiBrokerError("Pi write failed")
            return None
        if operation == "bash":
            return self._bash(request)
        raise PiBrokerError("Pi tool operation is invalid")

    def _bash(self, request: _BrokerRequest) -> dict[str, object]:
        command = request.get("command")
        if not isinstance(command, str):
            raise PiBrokerError("Pi bash command is invalid")
        try:
            if len(command.encode("utf-8")) > _MAX_COMMAND_BYTES:
                raise PiBrokerError("Pi bash command exceeds its bound")
        except UnicodeEncodeError as error:
            raise PiBrokerError("Pi bash command is invalid") from error
        if self._workspace_path(request.get("cwd")) != _WORKSPACE:
            raise PiBrokerError("Pi bash cwd is invalid")
        timeout = request.get("timeout")
        if not isinstance(timeout, int) or isinstance(timeout, bool):
            raise PiBrokerError("Pi bash timeout is invalid")
        timeout = min(_MAX_TIMEOUT_SECONDS, max(1, timeout))
        response = self._backend.execute(self._bounded_command(
            f"timeout --kill-after=5s {timeout}s bash -c {shlex.quote(command)}"))
        output = response.output
        if len(output.encode("utf-8")) > _MAX_CONTENT_BYTES:
            raise PiBrokerError("Pi bash output exceeds its bound")
        return {
            "exitCode": response.exit_code,
            "output": base64.b64encode(output.encode("utf-8")).decode("ascii"),
        }

    def _require_success(self, command: str) -> str:
        response = self._backend.execute(self._bounded_command(command))
        if response.exit_code != 0:
            raise PiBrokerError("Pi workspace operation failed")
        return response.output

    @staticmethod
    def _bounded_command(command: str) -> str:
        """Cap output inside the sandbox before docker-py can buffer it on the host."""
        return (
            "set -o pipefail; "
            f"({command}) 2>&1 | head -c {_MAX_CONTENT_BYTES}; "
            "status=${PIPESTATUS[0]}; "
            "if [ \"$status\" -eq 141 ]; then "
            "printf '\\n[Pi output truncated]\\n'; exit 1; fi; "
            "exit \"$status\""
        )
