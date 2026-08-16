"""One-turn, capability-authenticated relay from Pi to the local model server."""
from __future__ import annotations

import hmac
import http.client
import json
import os
import socket
import socketserver
import stat
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable

from assist.pi_skills import PiSkillAuthority

_MAX_REQUEST_BYTES = 2 * 1024 * 1024
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_MODEL_CALLS = 12
_MAX_TOKENS = 2048
_SOCKET_TIMEOUT_SECONDS = 5
_MODEL_RESPONSE_TIMEOUT_SECONDS = 75
_CAPABILITY_HEADER = "x-assist-pi-capability"
_PATH = "/v1/chat/completions"
_QWEN_TEMPLATE_OPTIONS = {"enable_thinking": False}


class PiProviderRelayError(ValueError):
    """The host cannot safely expose its model through this Pi relay."""


class _UnixHTTPServer(socketserver.UnixStreamServer):
    """One request at a time: M1 deliberately has no provider parallelism."""


class PiProviderRelay:
    """Forward only bounded, authenticated chat-completion requests to one model."""

    def __init__(self, control_dir: str | Path, upstream_url: str,
                 api_key: str, model: str, capability: str,
                 trace_start: Callable[[], object] | None = None,
                 trace_settle: Callable[[object, bool], None] | None = None) -> None:
        self._control_dir = Path(control_dir)
        self._socket_path = self._control_dir / "provider.sock"
        self._upstream = self._validate_upstream(upstream_url)
        self._api_key = api_key
        self._model = model
        self._capability = capability
        self._trace_start = trace_start
        self._trace_settle = trace_settle
        self._validate_control_dir()
        if self._socket_path.exists():
            raise PiProviderRelayError("Pi provider socket already exists")
        handler = type("BoundPiProviderHandler", (_ProviderHandler,), {"relay": self})
        self._server = _UnixHTTPServer(str(self._socket_path), handler)
        os.chmod(self._socket_path, 0o600)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="pi-provider-relay", daemon=True)
        self._started = False
        self._state_lock = threading.Lock()
        self._calls = 0
        self._stopping = False
        self._connections: set[http.client.HTTPConnection] = set()
        self._clients: set[socket.socket] = set()
        self._skill_authority: PiSkillAuthority | None = None

    @property
    def socket_path(self) -> Path:
        """Private Unix endpoint for the worker-namespace loopback adapter."""
        return self._socket_path

    def start(self) -> None:
        """Begin accepting requests after the manager has mounted the socket."""
        if self._started:
            raise RuntimeError("Pi provider relay is already running")
        self._started = True
        self._thread.start()

    def configure_skills(self, authority: PiSkillAuthority) -> None:
        """Attach the broker-shared authority before any provider admission."""
        if self._started or self._skill_authority is not None:
            raise RuntimeError("Pi provider relay skill authority is already configured")
        self._skill_authority = authority

    def close(self) -> None:
        """Stop model admission and close any forwarding connection in progress."""
        self.stop_admission()
        if self._started:
            shutdown = threading.Thread(target=self._server.shutdown,
                                        name="pi-provider-relay-stop", daemon=True)
            shutdown.start()
            shutdown.join(timeout=5)
            self._thread.join(timeout=5)
            if shutdown.is_alive() or self._thread.is_alive():
                raise RuntimeError("Pi provider relay did not drain")
            self._started = False
        self._server.server_close()
        self._socket_path.unlink(missing_ok=True)

    def stop_admission(self) -> None:
        """Fence new model calls and interrupt any already-forwarding call."""
        with self._state_lock:
            self._stopping = True
            connections = tuple(self._connections)
            clients = tuple(self._clients)
        for connection in connections:
            if connection.sock is not None:
                try:
                    connection.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            connection.close()
        for client in clients:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            client.close()

    @staticmethod
    def _validate_upstream(value: str) -> urllib.parse.SplitResult:
        parsed = urllib.parse.urlsplit(value)
        if (parsed.scheme not in {"http", "https"}
                or parsed.hostname not in {"127.0.0.1", "::1"}
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment or parsed.path.rstrip("/") != "/v1"):
            raise PiProviderRelayError("Pi model endpoint is not local OpenAI /v1")
        return parsed

    def _validate_control_dir(self) -> None:
        metadata = os.lstat(self._control_dir)
        if (not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid()
                or metadata.st_mode & 0o077):
            raise PiProviderRelayError("Pi control directory is unsafe")

    def register_client(self, client: socket.socket) -> None:
        """Reserve an accepted private client so preview-off can interrupt it."""
        with self._state_lock:
            if self._stopping:
                raise PiProviderRelayError("Pi provider relay is stopped")
            self._clients.add(client)

    def finish_client(self, client: socket.socket) -> None:
        """Release a client after it has finished receiving its response."""
        with self._state_lock:
            self._clients.discard(client)

    def validate_request(self, method: str, path: str, headers: object,
                         body: bytes) -> bytes:
        """Validate an inbound request before it can consume model authority."""
        if not hasattr(headers, "get"):
            raise PiProviderRelayError("Pi model request is invalid")
        if method != "POST" or path != _PATH or len(body) > _MAX_REQUEST_BYTES:
            self.clear_loader_for_capability(headers)
            raise PiProviderRelayError("Pi model request is invalid")
        content_type = headers.get("Content-Type")
        capability = headers.get(_CAPABILITY_HEADER)
        if (not isinstance(content_type, str) or not content_type.lower().startswith("application/json")
                or not isinstance(capability, str)
                or not hmac.compare_digest(capability, self._capability)):
            raise PiProviderRelayError("Pi model request is unauthorized")
        try:
            try:
                payload = json.loads(body)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise PiProviderRelayError("Pi model request is malformed") from error
            if not isinstance(payload, dict) or payload.get("model") != self._model:
                raise PiProviderRelayError("Pi model request uses an invalid model")
            if (self._skill_authority is not None
                    and not self._skill_authority.can_continue_request(payload)):
                raise PiProviderRelayError("Pi model request does not continue skill loading")
            if ("max_completion_tokens" in payload
                    or "n" in payload and payload["n"] != 1
                    or "stream" in payload and payload["stream"] is not True):
                raise PiProviderRelayError("Pi model request changes its generation policy")
            normalized = dict(payload)
            normalized["model"] = self._model
            normalized["max_tokens"] = _MAX_TOKENS
            normalized["n"] = 1
            normalized["temperature"] = 0.1
            # Qwen otherwise spends the entire bounded completion on hidden reasoning
            # and returns no visible reply. Match the established Deep Agents request.
            normalized["chat_template_kwargs"] = _QWEN_TEMPLATE_OPTIONS
            # Pi's provider adapter expects the OpenAI stream shape. The relay
            # bounds and buffers that response before handing it back to the worker,
            # so the worker cannot choose a non-streaming or unbounded mode.
            normalized["stream"] = True
            canonical = json.dumps(normalized, separators=(",", ":")).encode("utf-8")
            if len(canonical) > _MAX_REQUEST_BYTES:
                raise PiProviderRelayError("Pi model request exceeds its bound")
            with self._state_lock:
                limited = self._stopping or self._calls >= _MAX_MODEL_CALLS
                if not limited:
                    self._calls += 1
            if limited:
                raise PiProviderRelayError("Pi model turn bound exceeded")
            return canonical
        except PiProviderRelayError:
            if self._skill_authority is not None:
                self._skill_authority.clear_loader()
            raise

    def clear_loader_for_capability(self, headers: object) -> None:
        """Clear a pending continuation only for this relay's authenticated worker."""
        if self._skill_authority is None or not hasattr(headers, "get"):
            return
        capability = headers.get(_CAPABILITY_HEADER)
        if isinstance(capability, str) and hmac.compare_digest(capability, self._capability):
            self._skill_authority.clear_loader()

    def forward(self, body: bytes, trace_operation: object | None = None) -> tuple[int, str, list[bytes]]:
        """Forward one already-authorized request without forwarding its headers."""
        completed = False
        connection_type = (http.client.HTTPSConnection
                           if self._upstream.scheme == "https" else http.client.HTTPConnection)
        deadline = time.monotonic() + _MODEL_RESPONSE_TIMEOUT_SECONDS
        connection = connection_type(
            self._upstream.hostname, self._upstream.port,
            timeout=min(_SOCKET_TIMEOUT_SECONDS, self._remaining_timeout(deadline)),
        )
        sockets: list[socket.socket] = []

        def expire() -> None:
            for socket_ in sockets:
                try:
                    socket_.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

        expiry: threading.Timer | None = None
        with self._state_lock:
            if self._stopping:
                if self._skill_authority is not None:
                    self._skill_authority.clear_loader()
                raise PiProviderRelayError("Pi provider relay is stopped")
            self._connections.add(connection)
            # Holding the short connection/send operation under this gate means
            # close() cannot complete before a post-stop request is impossible.
            try:
                connection.request(
                    "POST", _PATH, body=body,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"},
                )
            except BaseException:
                self._connections.discard(connection)
                connection.close()
                if self._skill_authority is not None:
                    self._skill_authority.clear_loader()
                raise
        try:
            if connection.sock is None:
                raise PiProviderRelayError("Pi model connection is unavailable")
            sockets.append(connection.sock)
            expiry = threading.Timer(self._remaining_timeout(deadline), expire)
            expiry.daemon = True
            expiry.start()
            connection.sock.settimeout(self._remaining_timeout(deadline))
            response = connection.getresponse()
            chunks = self._read_response(response, deadline)
            completed = 200 <= response.status < 300
            if completed and self._skill_authority is not None:
                try:
                    payload = json.loads(body)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise PiProviderRelayError("Pi model request is malformed") from error
                if not self._skill_authority.continue_request(payload):
                    raise PiProviderRelayError("Pi model request does not continue skill loading")
                observed = _sole_loader_completion(b"".join(chunks))
                if observed is None:
                    self._skill_authority.clear_loader()
                else:
                    self._skill_authority.observe_loader(*observed)
            elif self._skill_authority is not None:
                self._skill_authority.clear_loader()
            content_type = "text/event-stream" if completed else "application/json"
            return response.status, content_type, chunks
        except BaseException:
            if self._skill_authority is not None:
                self._skill_authority.clear_loader()
            raise
        finally:
            with self._state_lock:
                self._connections.discard(connection)
            connection.close()
            if expiry is not None:
                expiry.cancel()
            self.settle_trace(trace_operation, completed)

    @staticmethod
    def _remaining_timeout(deadline: float) -> float:
        """Return the remaining absolute budget for one model response."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PiProviderRelayError("Pi model response timed out")
        return remaining

    @classmethod
    def _set_response_timeout(cls, response: http.client.HTTPResponse, deadline: float) -> None:
        raw = getattr(response.fp, "raw", None)
        socket_ = getattr(raw, "_sock", None)
        if socket_ is not None:
            socket_.settimeout(cls._remaining_timeout(deadline))

    @classmethod
    def _read_response(cls, response: http.client.HTTPResponse, deadline: float) -> list[bytes]:
        if 200 <= response.status < 300:
            if not _is_sse(response):
                raise PiProviderRelayError("Pi model response is not streamed")
            return cls._read_complete_sse(response, deadline)
        return cls._read_bounded_response(response, deadline)

    @classmethod
    def _read_bounded_response(cls, response: http.client.HTTPResponse,
                               deadline: float) -> list[bytes]:
        chunks: list[bytes] = []
        total = 0
        while True:
            cls._set_response_timeout(response, deadline)
            chunk = response.read(65536)
            if not chunk:
                return chunks
            total = _append_response_chunk(chunks, total, chunk)

    @classmethod
    def _read_complete_sse(cls, response: http.client.HTTPResponse,
                           deadline: float) -> list[bytes]:
        chunks: list[bytes] = []
        total = 0
        while True:
            cls._set_response_timeout(response, deadline)
            line = response.readline(_MAX_RESPONSE_BYTES + 1)
            if not line:
                raise PiProviderRelayError("Pi model stream ended before completion")
            total = _append_response_chunk(chunks, total, line)
            value = _sse_data_value(line)
            if value is None:
                continue
            if value == b"[DONE]":
                return chunks
            if not _is_openai_stream_frame(value):
                raise PiProviderRelayError("Pi model stream is malformed")

    def start_trace(self) -> object | None:
        """Keep optional activity recording from changing model admission."""
        if self._trace_start is None:
            return None
        try:
            return self._trace_start()
        except Exception:
            return None

    def settle_trace(self, operation: object | None, completed: bool) -> None:
        """Contain recorder failures after an admitted model request."""
        if operation is None or self._trace_settle is None:
            return
        try:
            self._trace_settle(operation, completed)
        except Exception:
            pass


def _sole_loader_completion(raw: bytes) -> tuple[str, str, object] | None:
    """Extract one complete OpenAI loader call from JSON or bounded SSE bytes."""
    values: list[object] = []
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if _contains_sse_data(raw):
        for line in decoded.splitlines():
            if not line.startswith("data:"):
                continue
            item = line[5:].strip()
            if item == "[DONE]":
                continue
            try:
                values.append(json.loads(item))
            except json.JSONDecodeError:
                return None
    else:
        try:
            values.append(json.loads(decoded))
        except json.JSONDecodeError:
            return None
    calls: dict[int, dict[str, object]] = {}
    for value in values:
        if not isinstance(value, dict):
            return None
        choices = value.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                return None
            message = choice.get("message")
            delta = choice.get("delta")
            container = message if isinstance(message, dict) else delta
            if not isinstance(container, dict):
                continue
            tool_calls = container.get("tool_calls")
            if tool_calls is None:
                continue
            if not isinstance(tool_calls, list):
                return None
            for item in tool_calls:
                if not isinstance(item, dict):
                    return None
                index = item.get("index", 0)
                if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                    return None
                prior = calls.setdefault(index, {"arguments": ""})
                if isinstance(item.get("id"), str):
                    prior["id"] = item["id"]
                function = item.get("function")
                if function is not None:
                    if not isinstance(function, dict):
                        return None
                    if isinstance(function.get("name"), str):
                        prior["name"] = function["name"]
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        prior["arguments"] = str(prior["arguments"]) + arguments
                    elif arguments is not None:
                        return None
    if len(calls) != 1:
        return None
    call = next(iter(calls.values()))
    if not isinstance(call.get("id"), str) or call.get("name") != "load_skill":
        return None
    try:
        arguments = json.loads(str(call.get("arguments", "")))
    except json.JSONDecodeError:
        return None
    return call["id"], "load_skill", arguments


def _append_response_chunk(chunks: list[bytes], total: int, chunk: bytes) -> int:
    total += len(chunk)
    if total > _MAX_RESPONSE_BYTES:
        raise PiProviderRelayError("Pi model response exceeds its bound")
    chunks.append(chunk)
    return total


def _is_sse(response: http.client.HTTPResponse) -> bool:
    content_type = response.getheader("Content-Type", "")
    return content_type.lower().split(";", 1)[0].strip() == "text/event-stream"


def _sse_data_value(line: bytes) -> bytes | None:
    """Return an exact complete SSE data line, if this is one."""
    if not line.endswith(b"\n"):
        raise PiProviderRelayError("Pi model stream is malformed")
    content = line[:-1]
    if content.endswith(b"\r"):
        content = content[:-1]
    if not content.startswith(b"data:"):
        return None
    return content[5:].lstrip(b" \t")


def _contains_sse_data(raw: bytes) -> bool:
    return any(line.startswith(b"data:") for line in raw.splitlines())


def _is_openai_stream_frame(value: bytes) -> bool:
    try:
        parsed = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    choices = parsed.get("choices") if isinstance(parsed, dict) else None
    return isinstance(choices, list) and bool(choices)



class _ProviderHandler(BaseHTTPRequestHandler):
    """Tiny HTTP boundary used only behind a private Unix socket."""

    protocol_version = "HTTP/1.1"
    relay: PiProviderRelay

    def setup(self) -> None:
        self.request.settimeout(_SOCKET_TIMEOUT_SECONDS)
        super().setup()

    def send_error(self, code: int, message: str | None = None,
                   explain: str | None = None) -> None:
        """Make any authenticated rejected request invalidate its continuation."""
        self.relay.clear_loader_for_capability(getattr(self, "headers", None))
        super().send_error(code, message, explain)

    def do_POST(self) -> None:
        registered = False
        try:
            try:
                self.relay.register_client(self.connection)
                registered = True
                length_text = self.headers.get("Content-Length")
                if (not isinstance(length_text, str) or not length_text.isdecimal()
                        or int(length_text) > _MAX_REQUEST_BYTES):
                    self.send_error(413)
                    return
                body = self.rfile.read(int(length_text))
                body = self.relay.validate_request(self.command, self.path, self.headers, body)
                operation = self.relay.start_trace()
                status, content_type, chunks = self.relay.forward(body, operation)
            except PiProviderRelayError:
                self.send_error(403)
                return
            except (OSError, http.client.HTTPException):
                self.send_error(502)
                return
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Connection", "close")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(chunk)
        finally:
            if registered:
                self.relay.finish_client(self.connection)
            self.close_connection = True

    def log_message(self, _format: str, *_args: Any) -> None:
        return
