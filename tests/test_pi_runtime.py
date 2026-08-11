"""The Pi runtime owns every per-turn authority and tears it down in order."""
from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import pytest

from assist.model_manager import OpenAIConfig
from assist import pi_runtime


class _Backend:
    container = object()


class _Worker:
    def __init__(self, control_dir: Path, events: list[str]) -> None:
        self._control_dir = control_dir
        self._events = events

    def wait(self, *, timeout: int) -> dict[str, int]:
        assert timeout == 1
        self._events.append("worker.wait")
        return {"StatusCode": 0}

class _Containers:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def run(self, _image: str, **kwargs: object) -> _Worker:
        self._events.append("worker.start")
        assert kwargs["network_mode"] == "none"
        assert kwargs["read_only"] is True
        assert kwargs["environment"] == {}
        assert kwargs["cap_drop"] == ["ALL"]
        assert kwargs["security_opt"] == ["no-new-privileges"]
        volumes = kwargs["volumes"]
        assert isinstance(volumes, dict)
        assert next(iter(volumes.values()))["mode"] == "ro"
        control = Path(next(iter(volumes)))
        return _Worker(control, self._events)


class _Docker:
    def __init__(self, events: list[str]) -> None:
        self.containers = _Containers(events)


class _SandboxManager:
    events: list[str] = []
    backend = _Backend()

    @classmethod
    def get_pi_sandbox_backend(cls, work_dir: str, timezone: str | None) -> _Backend:
        cls.events.append("sandbox.start")
        assert timezone == "America/Los_Angeles"
        return cls.backend

    @classmethod
    def _get_docker_client(cls) -> _Docker:
        return _Docker(cls.events)

    @classmethod
    def cleanup(cls, work_dir: str, expected_container: object) -> None:
        assert expected_container is cls.backend.container
        cls.events.append("sandbox.cleanup")


class _Broker:
    def __init__(self, backend: _Backend, control_dir: Path, capability: str) -> None:
        assert backend is _SandboxManager.backend
        assert len(capability) == 43
        self.events = _SandboxManager.events

    def start(self) -> None:
        self.events.append("broker.start")

    def stop_admission(self) -> None:
        self.events.append("broker.stop")

    def close(self) -> None:
        self.events.append("broker.close")


class _Relay:
    def __init__(self, control_dir: Path, upstream: str, api_key: str,
                 model: str, capability: str) -> None:
        assert upstream == "http://127.0.0.1:8000/v1"
        assert api_key == "secret"
        assert model == "qwen"
        assert len(capability) == 43
        self.events = _SandboxManager.events

    def start(self) -> None:
        self.events.append("relay.start")

    def stop_admission(self) -> None:
        self.events.append("relay.stop")

    def close(self) -> None:
        self.events.append("relay.close")


class _ResultSink:
    def __init__(self, control_dir: Path, capability: str) -> None:
        assert len(capability) == 43
        self.events = _SandboxManager.events

    def receive(self) -> pi_runtime.PiRuntimeResult:
        self.events.append("result.receive")
        return pi_runtime.PiRuntimeResult("done", 1)

    def close(self) -> None:
        self.events.append("result.close")


def test_runtime_starts_only_host_authorities_and_reaps_them(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "thread" / "workspace"
    work_dir.mkdir(parents=True)
    _SandboxManager.events = []
    monkeypatch.setattr(pi_runtime, "current_model_config", lambda: OpenAIConfig(
        "http://localhost:8000/v1", "qwen", "secret", 32768))
    monkeypatch.setattr(pi_runtime, "PiToolBroker", _Broker)
    monkeypatch.setattr(pi_runtime, "PiProviderRelay", _Relay)
    monkeypatch.setattr(pi_runtime, "PiResultSink", _ResultSink)

    result = pi_runtime.PiRuntimeManager(_SandboxManager).run(
        work_dir=str(work_dir), timezone="America/Los_Angeles", prompt="hello",
        history=[("user", "earlier")], system_prompt="be useful",
    )

    assert result == pi_runtime.PiRuntimeResult("done", 1)
    assert _SandboxManager.events == [
        "sandbox.start", "broker.start", "relay.start", "worker.start", "worker.wait",
        "result.receive", "broker.stop", "relay.stop", "sandbox.cleanup", "broker.close",
        "relay.close", "result.close",
    ]
    assert not list(work_dir.parent.glob(".pi-turn-*"))


def test_result_sink_accepts_only_a_capability_authenticated_result(tmp_path: Path) -> None:
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    sink = pi_runtime.PiResultSink(control, "a" * 43)
    with socket.socket(socket.AF_UNIX) as client:
        client.connect(str(control / "result.sock"))
        client.sendall(json.dumps({
            "capability": "a" * 43, "status": "completed", "reply": "done", "turns": 1,
        }).encode())
        client.shutdown(socket.SHUT_WR)
    try:
        assert sink.receive() == pi_runtime.PiRuntimeResult("done", 1)
    finally:
        sink.close()
    assert not (control / "result.sock").exists()

    rejected = pi_runtime.PiResultSink(control, "a" * 43)
    with socket.socket(socket.AF_UNIX) as client:
        client.connect(str(control / "result.sock"))
        client.sendall(json.dumps({
            "capability": "b" * 43, "status": "completed", "reply": "done", "turns": 1,
        }).encode())
        client.shutdown(socket.SHUT_WR)
    try:
        with pytest.raises(pi_runtime.PiRuntimeError, match="unauthorized"):
            rejected.receive()
    finally:
        rejected.close()


def test_runtime_rejects_a_nonzero_worker_status(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "thread" / "workspace"
    work_dir.mkdir(parents=True)
    _SandboxManager.events = []
    monkeypatch.setattr(pi_runtime, "current_model_config", lambda: OpenAIConfig(
        "http://localhost:8000/v1", "qwen", "secret", 32768))
    monkeypatch.setattr(pi_runtime, "PiToolBroker", _Broker)
    monkeypatch.setattr(pi_runtime, "PiProviderRelay", _Relay)
    monkeypatch.setattr(pi_runtime, "PiResultSink", _ResultSink)

    class BadWorker(_Worker):
        def wait(self, *, timeout: int) -> dict[str, int]:
            self._events.append("worker.wait")
            return {"StatusCode": 1}

        def stop(self, *, timeout: int) -> None:
            self._events.append("worker.stop")

        def kill(self) -> None:
            self._events.append("worker.kill")

    class BadContainers(_Containers):
        def run(self, _image: str, **kwargs: object) -> BadWorker:
            self._events.append("worker.start")
            return BadWorker(Path(next(iter(kwargs["volumes"]))), self._events)  # type: ignore[index]

    class BadDocker:
        def __init__(self) -> None:
            self.containers = BadContainers(_SandboxManager.events)

    monkeypatch.setattr(_SandboxManager, "_get_docker_client", classmethod(
        lambda cls: BadDocker()))

    with pytest.raises(pi_runtime.PiRuntimeError, match="exited unsuccessfully"):
        pi_runtime.PiRuntimeManager(_SandboxManager).run(
            work_dir=str(work_dir), timezone="America/Los_Angeles", prompt="hello",
            history=[], system_prompt="be useful",
        )

    assert _SandboxManager.events[-6:] == [
        "broker.stop", "relay.stop", "sandbox.cleanup", "broker.close", "relay.close",
        "result.close",
    ]


def test_stop_reaps_an_active_worker_before_returning(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "thread" / "workspace"
    work_dir.mkdir(parents=True)
    _SandboxManager.events = []
    started = threading.Event()
    monkeypatch.setattr(pi_runtime, "current_model_config", lambda: OpenAIConfig(
        "http://localhost:8000/v1", "qwen", "secret", 32768))
    monkeypatch.setattr(pi_runtime, "PiToolBroker", _Broker)
    monkeypatch.setattr(pi_runtime, "PiProviderRelay", _Relay)
    monkeypatch.setattr(pi_runtime, "PiResultSink", _ResultSink)

    class WaitingWorker(_Worker):
        stopped = False

        def wait(self, *, timeout: int) -> dict[str, int]:
            self._events.append("worker.wait")
            started.set()
            if not self.stopped:
                raise pi_runtime.ReadTimeout("still running")
            return {"StatusCode": 0}

        def stop(self, *, timeout: int) -> None:
            self._events.append("worker.stop")
            self.stopped = True

        def kill(self) -> None:
            self._events.append("worker.kill")

    class WaitingContainers(_Containers):
        def run(self, _image: str, **kwargs: object) -> WaitingWorker:
            self._events.append("worker.start")
            return WaitingWorker(Path(next(iter(kwargs["volumes"]))), self._events)  # type: ignore[index]

    class WaitingDocker:
        def __init__(self) -> None:
            self.containers = WaitingContainers(_SandboxManager.events)

    monkeypatch.setattr(_SandboxManager, "_get_docker_client", classmethod(
        lambda cls: WaitingDocker()))
    manager = pi_runtime.PiRuntimeManager(_SandboxManager)
    errors: list[Exception] = []

    def run() -> None:
        try:
            manager.run(work_dir=str(work_dir), timezone="America/Los_Angeles",
                        prompt="hello", history=[], system_prompt="be useful", turn_id="thread")
        except Exception as error:
            errors.append(error)

    runner = threading.Thread(target=run)
    runner.start()
    assert started.wait(2)
    manager.stop("thread", timeout=5)
    runner.join(5)

    assert not runner.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], pi_runtime.PiRuntimeError)
    assert "stopped" in str(errors[0])
    assert _SandboxManager.events[-8:] == [
        "broker.stop", "relay.stop", "worker.stop", "worker.wait", "sandbox.cleanup",
        "broker.close", "relay.close", "result.close",
    ]


def test_withdrawn_admission_reaps_the_worker(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "thread" / "workspace"
    work_dir.mkdir(parents=True)
    _SandboxManager.events = []
    monkeypatch.setattr(pi_runtime, "current_model_config", lambda: OpenAIConfig(
        "http://localhost:8000/v1", "qwen", "secret", 32768))
    monkeypatch.setattr(pi_runtime, "PiToolBroker", _Broker)
    monkeypatch.setattr(pi_runtime, "PiProviderRelay", _Relay)
    monkeypatch.setattr(pi_runtime, "PiResultSink", _ResultSink)

    class StoppableWorker(_Worker):
        def wait(self, *, timeout: int) -> dict[str, int]:
            self._events.append("worker.wait")
            return {"StatusCode": 0}

        def stop(self, *, timeout: int) -> None:
            self._events.append("worker.stop")

        def kill(self) -> None:
            self._events.append("worker.kill")

    class StoppableContainers(_Containers):
        def run(self, _image: str, **kwargs: object) -> StoppableWorker:
            self._events.append("worker.start")
            return StoppableWorker(Path(next(iter(kwargs["volumes"]))), self._events)  # type: ignore[index]

    class StoppableDocker:
        def __init__(self) -> None:
            self.containers = StoppableContainers(_SandboxManager.events)

    monkeypatch.setattr(_SandboxManager, "_get_docker_client", classmethod(
        lambda cls: StoppableDocker()))

    with pytest.raises(pi_runtime.PiRuntimeError, match="preview was stopped"):
        pi_runtime.PiRuntimeManager(_SandboxManager).run(
            work_dir=str(work_dir), timezone="America/Los_Angeles", prompt="hello",
            history=[], system_prompt="be useful", turn_id="thread", admitted=lambda: False)

    assert _SandboxManager.events[-8:] == [
        "broker.stop", "relay.stop", "worker.stop", "worker.wait", "sandbox.cleanup",
        "broker.close", "relay.close", "result.close",
    ]


def test_teardown_attempts_every_authority_after_a_cleanup_failure(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "thread" / "workspace"
    work_dir.mkdir(parents=True)
    _SandboxManager.events = []
    monkeypatch.setattr(pi_runtime, "current_model_config", lambda: OpenAIConfig(
        "http://localhost:8000/v1", "qwen", "secret", 32768))
    monkeypatch.setattr(pi_runtime, "PiToolBroker", _Broker)
    monkeypatch.setattr(pi_runtime, "PiProviderRelay", _Relay)
    monkeypatch.setattr(pi_runtime, "PiResultSink", _ResultSink)

    def broken_cleanup(cls: type[_SandboxManager], work_dir: str,
                       expected_container: object) -> None:
        cls.events.append("sandbox.cleanup")
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(_SandboxManager, "cleanup", classmethod(broken_cleanup))

    with pytest.raises(pi_runtime.PiRuntimeError, match="teardown failed"):
        pi_runtime.PiRuntimeManager(_SandboxManager).run(
            work_dir=str(work_dir), timezone="America/Los_Angeles", prompt="hello",
            history=[], system_prompt="be useful",
        )

    assert _SandboxManager.events[-6:] == [
        "broker.stop", "relay.stop", "sandbox.cleanup", "broker.close", "relay.close", "result.close",
    ]
    assert not list(work_dir.parent.glob(".pi-turn-*"))


def test_runtime_reaps_a_timed_out_worker_with_term_then_kill(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "thread" / "workspace"
    work_dir.mkdir(parents=True)
    _SandboxManager.events = []
    monkeypatch.setattr(pi_runtime, "current_model_config", lambda: OpenAIConfig(
        "http://localhost:8000/v1", "qwen", "secret", 32768))
    monkeypatch.setattr(pi_runtime, "PiToolBroker", _Broker)
    monkeypatch.setattr(pi_runtime, "PiProviderRelay", _Relay)
    monkeypatch.setattr(pi_runtime, "PiResultSink", _ResultSink)

    class StuckWorker(_Worker):
        waits = 0

        def wait(self, *, timeout: int) -> dict[str, int]:
            self._events.append("worker.wait")
            self.waits += 1
            if self.waits < 3:
                raise RuntimeError("still running")
            return {"StatusCode": 137}

        def stop(self, *, timeout: int) -> None:
            self._events.append("worker.stop")
            raise RuntimeError("TERM ignored")

        def kill(self) -> None:
            self._events.append("worker.kill")

    class StuckContainers(_Containers):
        def run(self, _image: str, **kwargs: object) -> StuckWorker:
            self._events.append("worker.start")
            return StuckWorker(Path(next(iter(kwargs["volumes"]))), self._events)  # type: ignore[index]

    class StuckDocker:
        def __init__(self) -> None:
            self.containers = StuckContainers(_SandboxManager.events)

    monkeypatch.setattr(_SandboxManager, "_get_docker_client", classmethod(
        lambda cls: StuckDocker()))

    with pytest.raises(pi_runtime.PiRuntimeError, match="could not complete"):
        pi_runtime.PiRuntimeManager(_SandboxManager).run(
            work_dir=str(work_dir), timezone="America/Los_Angeles", prompt="hello",
            history=[], system_prompt="be useful",
        )

    assert _SandboxManager.events[-10:] == [
        "broker.stop", "relay.stop", "worker.stop", "worker.wait", "worker.kill",
        "worker.wait", "sandbox.cleanup", "broker.close", "relay.close", "result.close",
    ]


def test_runtime_does_not_cross_cleanup_boundary_when_worker_cannot_be_reaped(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "thread" / "workspace"
    work_dir.mkdir(parents=True)
    _SandboxManager.events = []
    monkeypatch.setattr(pi_runtime, "current_model_config", lambda: OpenAIConfig(
        "http://localhost:8000/v1", "qwen", "secret", 32768))
    monkeypatch.setattr(pi_runtime, "PiToolBroker", _Broker)
    monkeypatch.setattr(pi_runtime, "PiProviderRelay", _Relay)
    monkeypatch.setattr(pi_runtime, "PiResultSink", _ResultSink)

    class UnreapedWorker(_Worker):
        def wait(self, *, timeout: int) -> dict[str, int]:
            self._events.append("worker.wait")
            raise RuntimeError("still running")

        def stop(self, *, timeout: int) -> None:
            self._events.append("worker.stop")

        def kill(self) -> None:
            self._events.append("worker.kill")

    class UnreapedContainers(_Containers):
        def run(self, _image: str, **kwargs: object) -> UnreapedWorker:
            self._events.append("worker.start")
            return UnreapedWorker(Path(next(iter(kwargs["volumes"]))), self._events)  # type: ignore[index]

    class UnreapedDocker:
        def __init__(self) -> None:
            self.containers = UnreapedContainers(_SandboxManager.events)

    monkeypatch.setattr(_SandboxManager, "_get_docker_client", classmethod(
        lambda cls: UnreapedDocker()))

    with pytest.raises(pi_runtime.PiRuntimeError, match="teardown failed"):
        pi_runtime.PiRuntimeManager(_SandboxManager).run(
            work_dir=str(work_dir), timezone="America/Los_Angeles", prompt="hello",
            history=[], system_prompt="be useful",
        )

    assert _SandboxManager.events[-6:] == [
        "broker.stop", "relay.stop", "worker.stop", "worker.wait", "worker.kill", "worker.wait",
    ]
