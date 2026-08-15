"""Boundary tests for the Pi worker's one private tool socket."""
from __future__ import annotations

import base64
import json
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from assist.pi_broker import PiToolBroker
from assist.pi_skills import PiSkill, PiSkillAuthority, PiSkillCatalog


@dataclass
class _Response:
    output: str = ""
    exit_code: int = 0


class _Backend:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.writes: list[tuple[str, str]] = []

    def execute(self, command: str) -> _Response:
        self.commands.append(command)
        return _Response("hello")

    def write(self, path: str, content: str) -> _Response:
        self.writes.append((path, content))
        return _Response()


class _BlockingBackend(_Backend):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def execute(self, command: str) -> _Response:
        self.commands.append(command)
        self.started.set()
        assert self.release.wait(2)
        return _Response("hello")


def _request(path: Path, payload: dict[str, object]) -> dict[str, object]:
    with socket.socket(socket.AF_UNIX) as client:
        client.connect(str(path))
        client.sendall(json.dumps(payload).encode() + b"\n")
        client.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        while chunk := client.recv(65536):
            chunks.append(chunk)
    return json.loads(b"".join(chunks))


def _broker(tmp_path: Path) -> tuple[PiToolBroker, _Backend, str]:
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    capability = "a" * 43
    backend = _Backend()
    broker = PiToolBroker(backend, control, capability)  # type: ignore[arg-type]
    broker.start()
    return broker, backend, capability


def _skill_authority() -> PiSkillAuthority:
    return PiSkillAuthority(PiSkillCatalog((
        PiSkill("render", "render", "body", "a" * 64, ("map_data",)),
    )))


def test_broker_load_requires_the_relay_observed_tool_call(tmp_path: Path) -> None:
    broker, _backend, capability = _broker(tmp_path)
    authority = _skill_authority()
    broker.stop_admission()
    broker.close()
    # Configure before listening, as the runtime does.
    control = tmp_path / "skills-control"
    control.mkdir(mode=0o700)
    configured = PiToolBroker(_Backend(), control, capability)  # type: ignore[arg-type]
    configured.configure_skills(authority)
    configured.start()
    try:
        denied = _request(configured.socket_path, {
            "version": 1, "id": 1, "capability": capability, "operation": "load_skill",
            "tool_call_id": "call-1", "name": "render", "arguments": {"name": "render"},
        })
        authority.observe_loader("call-1", "load_skill", {"name": "render"})
        accepted = _request(configured.socket_path, {
            "version": 1, "id": 2, "capability": capability, "operation": "load_skill",
            "tool_call_id": "call-1", "name": "render", "arguments": {"name": "render"},
        })
    finally:
        configured.close()

    assert denied["ok"] is False
    assert accepted["ok"] is True
    assert isinstance(accepted["value"], str)


def test_broker_rejects_wrong_capability(tmp_path: Path) -> None:
    broker, backend, capability = _broker(tmp_path)
    try:
        response = _request(broker.socket_path, {
            "version": 1, "id": 1, "capability": capability[:-1] + "b",
            "operation": "read", "path": "/workspace/file.txt",
        })
    finally:
        broker.close()

    assert response["ok"] is False
    assert backend.commands == []


def test_broker_traces_only_admitted_fixed_labels(tmp_path: Path) -> None:
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    events: list[tuple[str, object, object]] = []
    broker = PiToolBroker(
        _Backend(), control, "a" * 43,
        trace_start=lambda name: events.append(("start", name, None)) or name,
        trace_settle=lambda operation, completed: events.append(("settle", operation, completed)),
    )  # type: ignore[arg-type]
    broker.start()
    try:
        denied = _request(broker.socket_path, {
            "version": 1, "id": 1, "capability": "b" * 43,
            "operation": "read", "path": "/workspace/file.txt",
        })
        accepted = _request(broker.socket_path, {
            "version": 1, "id": 2, "capability": "a" * 43,
            "operation": "access", "path": "/workspace/file.txt", "mode": "read",
        })
    finally:
        broker.close()

    assert denied["ok"] is False
    assert accepted["ok"] is True
    assert events == [("start", "read", None), ("settle", "read", True)]


def test_broker_continues_when_trace_callbacks_fail(tmp_path: Path) -> None:
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    broker = PiToolBroker(
        _Backend(), control, "a" * 43,
        trace_start=lambda _name: (_ for _ in ()).throw(RuntimeError("trace failed")),
    )  # type: ignore[arg-type]
    broker.start()
    try:
        response = _request(broker.socket_path, {
            "version": 1, "id": 1, "capability": "a" * 43,
            "operation": "read", "path": "/workspace/file.txt",
        })
    finally:
        broker.close()

    assert response["ok"] is True


def test_broker_rejects_multiple_frames_without_a_backend_call(tmp_path: Path) -> None:
    broker, backend, capability = _broker(tmp_path)
    try:
        with socket.socket(socket.AF_UNIX) as client:
            client.connect(str(broker.socket_path))
            client.sendall(
                json.dumps({
                    "version": 1, "id": 1, "capability": capability,
                    "operation": "read", "path": "/workspace/file.txt",
                }).encode() + b"\n{}\n")
            client.shutdown(socket.SHUT_WR)
            response = json.loads(client.recv(65536))
    finally:
        broker.close()

    assert response["ok"] is False
    assert backend.commands == []


def test_broker_reads_only_the_selected_workspace(tmp_path: Path) -> None:
    broker, backend, capability = _broker(tmp_path)
    try:
        response = _request(broker.socket_path, {
            "version": 1, "id": 1, "capability": capability,
            "operation": "read", "path": "/workspace/file.txt",
        })
        escaped = _request(broker.socket_path, {
            "version": 1, "id": 2, "capability": capability,
            "operation": "read", "path": "/workspace/../agent/secret",
        })
    finally:
        broker.close()

    assert response == {
        "version": 1, "id": 1, "ok": True,
        "value": base64.b64encode(b"hello").decode(),
    }
    assert escaped["ok"] is False
    assert len(backend.commands) == 1
    assert "/workspace/file.txt" in backend.commands[0]


def test_broker_maps_access_and_mkdir_to_the_selected_backend(tmp_path: Path) -> None:
    broker, backend, capability = _broker(tmp_path)
    try:
        access = _request(broker.socket_path, {
            "version": 1, "id": 1, "capability": capability,
            "operation": "access", "path": "/workspace/file.txt", "mode": "read",
        })
        mkdir = _request(broker.socket_path, {
            "version": 1, "id": 2, "capability": capability,
            "operation": "mkdir", "path": "/workspace/new-dir",
        })
    finally:
        broker.close()

    assert access["ok"] is True
    assert mkdir["ok"] is True
    assert "test -r -- /workspace/file.txt" in backend.commands[0]
    assert "mkdir -p -- /workspace/new-dir" in backend.commands[1]


def test_broker_clamps_bash_and_writes_through_its_backend(tmp_path: Path) -> None:
    broker, backend, capability = _broker(tmp_path)
    try:
        write = _request(broker.socket_path, {
            "version": 1, "id": 1, "capability": capability,
            "operation": "write", "path": "/workspace/file.txt", "content": "new",
        })
        bash = _request(broker.socket_path, {
            "version": 1, "id": 2, "capability": capability,
            "operation": "bash", "command": "printf ok", "cwd": "/workspace", "timeout": 999,
        })
    finally:
        broker.close()

    assert write["ok"] is True
    assert backend.writes == [("/workspace/file.txt", "new")]
    assert bash["ok"] is True
    assert "120s bash -c 'printf ok'" in backend.commands[-1]
    assert "head -c 98304" in backend.commands[-1]


def test_broker_does_not_finish_close_while_a_request_is_active(tmp_path: Path) -> None:
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    backend = _BlockingBackend()
    broker = PiToolBroker(backend, control, "a" * 43)  # type: ignore[arg-type]
    broker.start()
    reply: dict[str, object] = {}

    requester = threading.Thread(target=lambda: reply.update(_request(broker.socket_path, {
        "version": 1, "id": 1, "capability": "a" * 43,
        "operation": "read", "path": "/workspace/file.txt",
    })))
    requester.start()
    assert backend.started.wait(1)
    broker.stop_admission()
    closer = threading.Thread(target=broker.close)
    closer.start()
    time.sleep(0.05)
    assert closer.is_alive()

    backend.release.set()
    requester.join(1)
    closer.join(1)

    assert not closer.is_alive()
    assert reply["ok"] is True


def test_broker_does_not_finish_close_before_reply_delivery(tmp_path: Path) -> None:
    broker, _backend, capability = _broker(tmp_path)
    writing = threading.Event()
    release = threading.Event()
    original_write = broker._write_frame  # noqa: SLF001 - exact lifecycle seam

    def delayed_write(connection: socket.socket, value: dict[str, object]) -> None:
        writing.set()
        assert release.wait(2)
        original_write(connection, value)

    broker._write_frame = delayed_write  # type: ignore[method-assign]  # noqa: SLF001
    reply: dict[str, object] = {}
    requester = threading.Thread(target=lambda: reply.update(_request(broker.socket_path, {
        "version": 1, "id": 1, "capability": capability,
        "operation": "read", "path": "/workspace/file.txt",
    })))
    requester.start()
    assert writing.wait(1)
    broker.stop_admission()
    closer = threading.Thread(target=broker.close)
    closer.start()
    time.sleep(0.05)
    assert closer.is_alive()

    release.set()
    requester.join(1)
    closer.join(1)

    assert not closer.is_alive()
    assert reply["ok"] is True


def test_broker_response_never_exceeds_its_frame_bound() -> None:
    server, client = socket.socketpair()
    try:
        PiToolBroker._write_frame(server, {  # noqa: SLF001 - exact protocol boundary
            "version": 1, "id": 1, "ok": True, "value": "x" * (512 * 1024),
        })
        response = json.loads(client.recv(65536))
    finally:
        server.close()
        client.close()

    assert response == {
        "version": 1, "id": 0, "ok": False,
        "error": "Pi workspace operation failed",
    }
