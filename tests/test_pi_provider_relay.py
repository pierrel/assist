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

from assist import pi_provider_relay
from assist.pi_provider_relay import (_MAX_REQUEST_BYTES, PiProviderRelay, PiProviderRelayError,
                                      _sole_loader_completion)
from assist.pi_skills import PiSkill, PiSkillAuthority, PiSkillCatalog


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
        body = b'data: {"id":"test","choices":[{}]}\n\ndata: [DONE]\n\n'
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
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


class _FailedUpstream(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(500)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, _format: str, *_args: object) -> None:
        return


class _KeepAliveSSEUpstream(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    sent_done = threading.Event()
    release = threading.Event()

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for part in (b': ping\n\ndata: {"choices":[{}]}\n\ndata: [DO', b'NE]\n\n'):
            self.wfile.write(f"{len(part):X}\r\n".encode() + part + b"\r\n")
        self.wfile.flush()
        type(self).sent_done.set()
        type(self).release.wait(10)
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except OSError:
            pass

    def log_message(self, _format: str, *_args: object) -> None:
        return


class _IncompleteSSEUpstream(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        body = b'data: {"choices":[{}]}\n\n'
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class _MalformedSSEUpstream(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        body = b"data: not-json\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class _ErrorSSEUpstream(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        body = b'data: {"error":"bad model response"}\n\ndata: [DONE]\n\n'
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class _UnstreamedUpstream(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        body = b'{"choices":[{}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class _HeaderInjectionUpstream(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(500)
        self.send_header("Content-Type", "application/json\r\nX-Upstream-Injected: yes")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, _format: str, *_args: object) -> None:
        return


class _SlowLineSSEUpstream(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    started = threading.Event()
    release = threading.Event()

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        part = b'data: {"choices":[{}]}'
        self.wfile.write(f"{len(part):X}\r\n".encode() + part + b"\r\n")
        self.wfile.flush()
        type(self).started.set()
        type(self).release.wait(10)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def _request(path: Path, capability: str, *, model: str = "qwen",
             extra: dict[str, object] | None = None, request_path: str = "/v1/chat/completions") -> bytes:
    body = json.dumps({"model": model, "messages": []} | (extra or {})).encode()
    raw = (
        f"POST {request_path} HTTP/1.1\r\n".encode()
        + b"Content-Type: application/json\r\n"
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


def _pending_render_authority() -> tuple[PiSkillAuthority, str]:
    authority = PiSkillAuthority(PiSkillCatalog((
        PiSkill("render", "show a file", "render rules", "a" * 64, ("map_data",)),
    )))
    authority.observe_loader("call-1", "load_skill", {"name": "render"})
    return authority, authority.load_skill("call-1", "load_skill", {"name": "render"})


def _render_continuation(result: str) -> dict[str, object]:
    return {
        "messages": [
            {"role": "assistant", "tool_calls": [{"id": "call-1", "function": {
                "name": "load_skill", "arguments": '{"name":"render"}'}}]},
            {"role": "tool", "tool_call_id": "call-1", "content": result},
        ],
        "tools": [
            {"type": "function", "function": {"name": name}}
            for name in ("read", "write", "edit", "bash", "load_skill", "map_data")
        ],
    }


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
        accepted = _request(relay.socket_path, "a" * 43, extra={
            "max_tokens": 999999, "chat_template_kwargs": {"enable_thinking": True},
        })
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
    assert payload["max_tokens"] == 2048
    assert payload["n"] == 1
    assert payload["temperature"] == 0.1
    assert payload["stream"] is True
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_provider_relay_traces_only_an_admitted_request(tmp_path: Path) -> None:
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    events: list[tuple[str, object, object]] = []
    relay = PiProviderRelay(
        control, f"http://127.0.0.1:{upstream.server_port}/v1", "secret", "qwen", "a" * 43,
        trace_start=lambda: events.append(("start", "model request", None)) or "model",
        trace_settle=lambda operation, completed: events.append(("settle", operation, completed)),
    )
    relay.start()
    try:
        denied = _request(relay.socket_path, "b" * 43)
        accepted = _request(relay.socket_path, "a" * 43)
    finally:
        relay.close()
        upstream.shutdown()
        upstream.server_close()

    assert b" 403 " in denied
    assert b" 200 " in accepted
    assert events == [("start", "model request", None), ("settle", "model", True)]


def test_provider_relay_continues_when_trace_callbacks_fail(tmp_path: Path) -> None:
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    relay = PiProviderRelay(
        control, f"http://127.0.0.1:{upstream.server_port}/v1", "secret", "qwen", "a" * 43,
        trace_start=lambda: (_ for _ in ()).throw(RuntimeError("trace failed")),
    )
    relay.start()
    try:
        accepted = _request(relay.socket_path, "a" * 43)
    finally:
        relay.close()
        upstream.shutdown()
        upstream.server_close()

    assert b" 200 " in accepted


def test_provider_relay_marks_an_upstream_error_did_not_complete(tmp_path: Path) -> None:
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FailedUpstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    outcomes: list[bool] = []
    relay = PiProviderRelay(
        control, f"http://127.0.0.1:{upstream.server_port}/v1", "secret", "qwen", "a" * 43,
        trace_start=lambda: "model", trace_settle=lambda _operation, completed: outcomes.append(completed),
    )
    relay.start()
    try:
        response = _request(relay.socket_path, "a" * 43)
    finally:
        relay.close()
        upstream.shutdown()
        upstream.server_close()

    assert b" 500 " in response
    assert outcomes == [False]


def test_provider_relay_returns_on_a_complete_sse_done_before_upstream_eof(tmp_path: Path) -> None:
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _KeepAliveSSEUpstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    _KeepAliveSSEUpstream.sent_done.clear()
    _KeepAliveSSEUpstream.release.clear()
    relay = PiProviderRelay(
        control, f"http://127.0.0.1:{upstream.server_port}/v1", "secret", "qwen", "a" * 43,
    )
    relay.start()
    response: list[bytes] = []
    client = threading.Thread(
        target=lambda: response.append(_request(relay.socket_path, "a" * 43)), daemon=True)
    client.start()
    try:
        assert _KeepAliveSSEUpstream.sent_done.wait(1)
        client.join(timeout=1)
        assert not client.is_alive()
    finally:
        _KeepAliveSSEUpstream.release.set()
        relay.close()
        upstream.shutdown()
        upstream.server_close()

    assert b" 200 " in response[0]
    assert b"Content-Type: text/event-stream" in response[0]
    assert b"data: [DONE]" in response[0]


def test_incomplete_sse_does_not_promote_a_pending_skill(tmp_path: Path) -> None:
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _IncompleteSSEUpstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    authority, result = _pending_render_authority()
    relay = PiProviderRelay(
        control, f"http://127.0.0.1:{upstream.server_port}/v1", "secret", "qwen", "a" * 43,
    )
    relay.configure_skills(authority)
    relay.start()
    try:
        rejected = _request(relay.socket_path, "a" * 43, extra=_render_continuation(result))
    finally:
        relay.close()
        upstream.shutdown()
        upstream.server_close()

    assert b" 403 " in rejected
    assert authority.active_tools == frozenset()


def test_malformed_sse_does_not_promote_a_pending_skill(tmp_path: Path) -> None:
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _MalformedSSEUpstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    authority, result = _pending_render_authority()
    relay = PiProviderRelay(
        control, f"http://127.0.0.1:{upstream.server_port}/v1", "secret", "qwen", "a" * 43,
    )
    relay.configure_skills(authority)
    relay.start()
    try:
        rejected = _request(relay.socket_path, "a" * 43, extra=_render_continuation(result))
    finally:
        relay.close()
        upstream.shutdown()
        upstream.server_close()

    assert b" 403 " in rejected
    assert authority.active_tools == frozenset()


@pytest.mark.parametrize("handler", [_ErrorSSEUpstream, _UnstreamedUpstream])
def test_invalid_success_response_does_not_promote_a_pending_skill(
        tmp_path: Path, handler: type[http.server.BaseHTTPRequestHandler]) -> None:
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    authority, result = _pending_render_authority()
    relay = PiProviderRelay(
        control, f"http://127.0.0.1:{upstream.server_port}/v1", "secret", "qwen", "a" * 43,
    )
    relay.configure_skills(authority)
    relay.start()
    try:
        rejected = _request(relay.socket_path, "a" * 43, extra=_render_continuation(result))
    finally:
        relay.close()
        upstream.shutdown()
        upstream.server_close()

    assert b" 403 " in rejected
    assert authority.active_tools == frozenset()


def test_provider_relay_does_not_forward_an_upstream_error_header(tmp_path: Path) -> None:
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _HeaderInjectionUpstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    relay = PiProviderRelay(
        control, f"http://127.0.0.1:{upstream.server_port}/v1", "secret", "qwen", "a" * 43,
    )
    relay.start()
    try:
        response = _request(relay.socket_path, "a" * 43)
    finally:
        relay.close()
        upstream.shutdown()
        upstream.server_close()

    assert b" 500 " in response
    assert b"X-Upstream-Injected" not in response


def test_provider_relay_deadline_interrupts_an_unterminated_sse_line(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _SlowLineSSEUpstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    _SlowLineSSEUpstream.started.clear()
    _SlowLineSSEUpstream.release.clear()
    monkeypatch.setattr(pi_provider_relay, "_MODEL_RESPONSE_TIMEOUT_SECONDS", 0.25)
    relay = PiProviderRelay(
        control, f"http://127.0.0.1:{upstream.server_port}/v1", "secret", "qwen", "a" * 43,
    )
    relay.start()
    response: list[bytes] = []
    client = threading.Thread(
        target=lambda: response.append(_request(relay.socket_path, "a" * 43)), daemon=True)
    client.start()
    try:
        assert _SlowLineSSEUpstream.started.wait(1)
        client.join(timeout=1)
        assert not client.is_alive()
    finally:
        _SlowLineSSEUpstream.release.set()
        relay.close()
        upstream.shutdown()
        upstream.server_close()

    assert b" 403 " in response[0]


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


def test_rejected_continuation_does_not_promote_its_declared_tool(tmp_path: Path) -> None:
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    _Upstream.requests = []
    authority, result = _pending_render_authority()
    relay = PiProviderRelay(
        control, f"http://127.0.0.1:{upstream.server_port}/v1", "secret", "qwen", "a" * 43,
    )
    relay.configure_skills(authority)
    relay.start()
    try:
        rejected = _request(relay.socket_path, "a" * 43,
                            extra=_render_continuation(result) | {"stream": False})
    finally:
        relay.close()
        upstream.shutdown()
        upstream.server_close()

    assert b" 403 " in rejected
    assert authority.active_tools == frozenset()
    assert _Upstream.requests == []


def test_invalid_authenticated_request_clears_a_pending_tool(tmp_path: Path) -> None:
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    authority, result = _pending_render_authority()
    relay = PiProviderRelay(
        control, f"http://127.0.0.1:{upstream.server_port}/v1", "secret", "qwen", "a" * 43,
    )
    relay.configure_skills(authority)
    relay.start()
    try:
        rejected = _request(relay.socket_path, "a" * 43,
                            extra=_render_continuation(result), request_path="/other")
        accepted = _request(relay.socket_path, "a" * 43,
                            extra=_render_continuation(result))
    finally:
        relay.close()
        upstream.shutdown()
        upstream.server_close()

    assert b" 403 " in rejected
    assert b" 403 " in accepted
    assert authority.active_tools == frozenset()


def test_oversized_authenticated_request_clears_a_pending_tool(tmp_path: Path) -> None:
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    authority, result = _pending_render_authority()
    relay = PiProviderRelay(
        control, f"http://127.0.0.1:{upstream.server_port}/v1", "secret", "qwen", "a" * 43,
    )
    relay.configure_skills(authority)
    relay.start()
    try:
        with socket.socket(socket.AF_UNIX) as client:
            client.connect(str(relay.socket_path))
            client.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                b"Content-Type: application/json\r\n"
                b"X-Assist-Pi-Capability: " + b"a" * 43 + b"\r\n"
                + f"Content-Length: {_MAX_REQUEST_BYTES + 1}\r\n\r\n".encode())
            rejected = client.recv(65536)
        accepted = _request(relay.socket_path, "a" * 43,
                            extra=_render_continuation(result))
    finally:
        relay.close()
        upstream.shutdown()
        upstream.server_close()

    assert b" 413 " in rejected
    assert b" 403 " in accepted
    assert authority.active_tools == frozenset()


def test_failed_continuation_does_not_promote_its_declared_tool(tmp_path: Path) -> None:
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FailedUpstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    authority, result = _pending_render_authority()
    relay = PiProviderRelay(
        control, f"http://127.0.0.1:{upstream.server_port}/v1", "secret", "qwen", "a" * 43,
    )
    relay.configure_skills(authority)
    relay.start()
    try:
        failed = _request(relay.socket_path, "a" * 43, extra=_render_continuation(result))
    finally:
        relay.close()
        upstream.shutdown()
        upstream.server_close()

    assert b" 500 " in failed
    assert authority.active_tools == frozenset()


def test_transport_failed_continuation_clears_its_pending_tool(tmp_path: Path) -> None:
    unavailable = socket.socket(socket.AF_INET)
    unavailable.bind(("127.0.0.1", 0))
    port = unavailable.getsockname()[1]
    unavailable.close()
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    authority, result = _pending_render_authority()
    relay = PiProviderRelay(
        control, f"http://127.0.0.1:{port}/v1", "secret", "qwen", "a" * 43,
    )
    relay.configure_skills(authority)
    payload = {"model": "qwen"} | _render_continuation(result)
    upstream: http.server.ThreadingHTTPServer | None = None
    try:
        canonical = relay.validate_request(
            "POST", "/v1/chat/completions",
            {"Content-Type": "application/json", "x-assist-pi-capability": "a" * 43},
            json.dumps(payload).encode(),
        )
        with pytest.raises(OSError):
            relay.forward(canonical)
        upstream = http.server.ThreadingHTTPServer(("127.0.0.1", port), _Upstream)
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        with pytest.raises(PiProviderRelayError, match="does not continue"):
            relay.forward(canonical)
    finally:
        relay.close()
        if upstream is not None:
            upstream.shutdown()
            upstream.server_close()

    assert authority.active_tools == frozenset()


def test_stopped_continuation_clears_its_pending_tool(tmp_path: Path) -> None:
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    authority, result = _pending_render_authority()
    relay = PiProviderRelay(
        control, "http://127.0.0.1:1/v1", "secret", "qwen", "a" * 43,
    )
    relay.configure_skills(authority)
    payload = {"model": "qwen"} | _render_continuation(result)
    try:
        canonical = relay.validate_request(
            "POST", "/v1/chat/completions",
            {"Content-Type": "application/json", "x-assist-pi-capability": "a" * 43},
            json.dumps(payload).encode(),
        )
        relay.stop_admission()
        with pytest.raises(PiProviderRelayError, match="stopped"):
            relay.forward(canonical)
    finally:
        relay.close()

    assert authority.active_tools == frozenset()


def test_provider_relay_refuses_nonlocal_or_malformed_endpoints(tmp_path: Path) -> None:
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    with pytest.raises(PiProviderRelayError):
        PiProviderRelay(control, "https://example.test/v1", "key", "qwen", "a" * 43)
    with pytest.raises(PiProviderRelayError):
        PiProviderRelay(control, "http://127.0.0.1:8000/not-v1", "key", "qwen", "a" * 43)
    with pytest.raises(PiProviderRelayError):
        PiProviderRelay(control, "http://localhost:8000/v1", "key", "qwen", "a" * 43)


def test_loader_completion_requires_exactly_one_complete_loader_call() -> None:
    one = json.dumps({"id": "response-1", "choices": [{"message": {"tool_calls": [{
        "id": "call-1", "function": {"name": "load_skill", "arguments": '{"name":"render"}'},
    }]}}]}).encode()
    multiple = json.dumps({"id": "response-1", "choices": [{"message": {"tool_calls": [
        {"id": "call-1", "function": {"name": "load_skill", "arguments": '{"name":"render"}'}},
        {"id": "call-2", "function": {"name": "map_data", "arguments": "{}"}},
    ]}}]}).encode()

    stream = b": ping\n\ndata: " + one + b"\n\ndata: [DONE]\n\n"
    assert _sole_loader_completion(stream) == ("call-1", "load_skill", {"name": "render"})
    assert _sole_loader_completion(multiple) is None


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
