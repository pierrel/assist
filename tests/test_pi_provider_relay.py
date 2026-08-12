"""Provider authority is available only through a one-turn Pi relay."""
from __future__ import annotations

import http.server
import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from assist.pi_provider_relay import PiProviderRelay, PiProviderRelayError


_adapter_path = Path(__file__).parents[1] / "assist/pi_runtime/provider_adapter.py"
_adapter_spec = importlib.util.spec_from_file_location("pi_provider_adapter", _adapter_path)
assert _adapter_spec is not None and _adapter_spec.loader is not None
provider_adapter = importlib.util.module_from_spec(_adapter_spec)
_adapter_spec.loader.exec_module(provider_adapter)


class _Upstream(http.server.BaseHTTPRequestHandler):
    requests: list[tuple[str, dict[str, str], dict[str, object]]] = []

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        type(self).requests.append((self.path, dict(self.headers), payload))
        body = b'{"id":"test","choices":[]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class _StalledUpstream(http.server.BaseHTTPRequestHandler):
    started = threading.Event()
    release = threading.Event()

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        type(self).started.set()
        type(self).release.wait(10)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def _request(path: Path, capability: str, *, model: str = "qwen",
             extra: dict[str, object] | None = None) -> bytes:
    body = json.dumps({"model": model, "messages": []} | (extra or {})).encode()
    raw = (
        b"POST /v1/chat/completions HTTP/1.1\r\n"
        b"Content-Type: application/json\r\n"
        + f"X-Assist-Pi-Capability: {capability}\r\n".encode()
        + f"Content-Length: {len(body)}\r\n\r\n".encode() + body
    )
    with socket.socket(socket.AF_UNIX) as client:
        client.connect(str(path))
        client.sendall(raw)
        client.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        while chunk := client.recv(65536):
            chunks.append(chunk)
    return b"".join(chunks)


def test_provider_relay_forwards_only_its_model_and_capability(tmp_path: Path) -> None:
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    _Upstream.requests = []
    relay = PiProviderRelay(
        control, f"http://127.0.0.1:{upstream.server_port}/v1",
        "secret", "qwen", "a" * 43,
    )
    relay.start()
    try:
        accepted = _request(relay.socket_path, "a" * 43, extra={"max_tokens": 999999})
        denied = _request(relay.socket_path, "b" * 43)
        wrong_model = _request(relay.socket_path, "a" * 43, model="other")
    finally:
        relay.close()
        upstream.shutdown()
        upstream.server_close()

    assert b" 200 " in accepted
    assert b" 403 " in denied
    assert b" 403 " in wrong_model
    assert len(_Upstream.requests) == 1
    path, headers, payload = _Upstream.requests[0]
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer secret"
    assert payload["model"] == "qwen"
    assert payload["max_tokens"] == 8192
    assert payload["n"] == 1
    assert payload["temperature"] == 0.1
    assert payload["stream"] is True


def test_provider_relay_rejects_generation_policy_override(tmp_path: Path) -> None:
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    _Upstream.requests = []
    relay = PiProviderRelay(
        control, f"http://127.0.0.1:{upstream.server_port}/v1",
        "secret", "qwen", "a" * 43,
    )
    relay.start()
    try:
        multiple = _request(relay.socket_path, "a" * 43, extra={"n": 2})
        alternate_limit = _request(
            relay.socket_path, "a" * 43, extra={"max_completion_tokens": 999999})
        streaming = _request(relay.socket_path, "a" * 43, extra={"stream": False})
        accepted_streaming = _request(relay.socket_path, "a" * 43, extra={"stream": True})
    finally:
        relay.close()
        upstream.shutdown()
        upstream.server_close()

    assert b" 403 " in multiple
    assert b" 403 " in alternate_limit
    assert b" 403 " in streaming
    assert b" 200 " in accepted_streaming
    assert _Upstream.requests[0][2]["stream"] is True


def test_provider_relay_refuses_nonlocal_or_malformed_endpoints(tmp_path: Path) -> None:
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    with pytest.raises(PiProviderRelayError):
        PiProviderRelay(control, "https://example.test/v1", "key", "qwen", "a" * 43)
    with pytest.raises(PiProviderRelayError):
        PiProviderRelay(control, "http://127.0.0.1:8000/not-v1", "key", "qwen", "a" * 43)
    with pytest.raises(PiProviderRelayError):
        PiProviderRelay(control, "http://localhost:8000/v1", "key", "qwen", "a" * 43)


def test_provider_relay_close_interrupts_a_stalled_model_response(tmp_path: Path) -> None:
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _StalledUpstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    _StalledUpstream.started.clear()
    _StalledUpstream.release.clear()
    relay = PiProviderRelay(
        control, f"http://127.0.0.1:{upstream.server_port}/v1",
        "secret", "qwen", "a" * 43,
    )
    relay.start()
    client = threading.Thread(target=_request, args=(relay.socket_path, "a" * 43), daemon=True)
    client.start()
    try:
        assert _StalledUpstream.started.wait(1)
        relay.close()
        client.join(timeout=1)
        assert not client.is_alive()
    finally:
        _StalledUpstream.release.set()
        upstream.shutdown()
        upstream.server_close()


def test_provider_adapter_listens_before_starting_worker() -> None:
    """The first provider connection cannot race the adapter's listen call."""
    adapter = Path(__file__).parents[1] / "assist/pi_runtime/provider_adapter.py"
    worker = (
        "import socket; "
        "s = socket.create_connection(('127.0.0.1', 18766), timeout=1); s.close()"
    )
    environment = dict(os.environ, ASSIST_PI_ADAPTER_PORT="18766")
    completed = subprocess.run(
        [sys.executable, str(adapter), sys.executable, "-c", worker],
        check=False, capture_output=True, text=True, timeout=5, env=environment,
    )
    assert completed.returncode == 0, completed.stderr


def test_provider_adapter_rejects_an_oversized_complete_header() -> None:
    client, adapter_side = socket.socketpair()
    try:
        client.sendall(b"POST / HTTP/1.1\r\nX: " + b"x" * (32 * 1024) + b"\r\n\r\n")
        with pytest.raises(RuntimeError, match="headers"):
            provider_adapter._read_request(adapter_side)
    finally:
        client.close()
        adapter_side.close()


def test_provider_adapter_keeps_request_and_response_bounds_separate(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A legal maximum request and response pass through the adapter together."""
    socket_path = tmp_path / "provider.sock"
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(socket_path))
    listener.listen(1)
    request_body = b"x" * (2 * 1024 * 1024)
    request = (
        b"POST /v1/chat/completions HTTP/1.1\r\n"
        + f"Content-Length: {len(request_body)}\r\n\r\n".encode() + request_body
    )
    response_body = b"y" * (2 * 1024 * 1024)
    response = (
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
        + f"Content-Length: {len(response_body)}\r\nConnection: close\r\n\r\n".encode()
        + response_body
    )
    received: list[bytes] = []

    def serve() -> None:
        upstream, _ = listener.accept()
        with upstream:
            chunks: list[bytes] = []
            while sum(map(len, chunks)) < len(request):
                chunks.append(upstream.recv(65536))
            received.append(b"".join(chunks))
            upstream.sendall(response)

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    monkeypatch.setattr(provider_adapter, "_SOCKET_PATH", str(socket_path))
    client, adapter_side = socket.socketpair()

    def relay_request() -> None:
        with adapter_side:
            provider_adapter._relay(adapter_side)

    try:
        relay = threading.Thread(target=relay_request, daemon=True)
        relay.start()
        client.sendall(request)
        chunks: list[bytes] = []
        while chunk := client.recv(65536):
            chunks.append(chunk)
        relay.join(timeout=2)
    finally:
        client.close()
        listener.close()

    assert not relay.is_alive()
    assert received == [request]
    assert b"".join(chunks) == response
