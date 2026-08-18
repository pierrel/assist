"""The Pi runtime owns every per-turn authority and tears it down in order."""
from __future__ import annotations

import json
import hashlib
import socket
import threading
from pathlib import Path

import pytest
from requests.exceptions import ConnectionError as RequestsConnectionError
from urllib3.exceptions import ReadTimeoutError

from assist.model_manager import OpenAIConfig
from assist import pi_runtime
from assist.pi_conversation import PiMessage
from assist.pi_skills import PiLoadedSkill, PiSkillCatalog
from assist.thread_queue import DEFAULT_HOLD_TIMEOUT_S


class _Backend:
    container = object()


def _loaded(name: str, run_id: str, body: str, tools: tuple[str, ...] = ()) -> PiLoadedSkill:
    return PiLoadedSkill(name, body, hashlib.sha256(body.encode()).hexdigest(), tools, run_id)


def test_worker_wait_keeps_running_on_dockers_wrapped_read_timeout() -> None:
    """docker-py wraps its timed wait in ConnectionError on this host."""
    class _TimedWorker:
        def wait(self, *, timeout: int) -> dict[str, int]:
            assert timeout == 1
            try:
                raise ReadTimeoutError(None, None, "timed out")
            except ReadTimeoutError:
                raise RequestsConnectionError("docker wait timed out")

    assert pi_runtime.PiRuntimeManager._wait_worker_once(_TimedWorker()) is None


def test_runtime_uses_the_shared_visible_thread_hold_bound() -> None:
    assert pi_runtime._WALL_TIMEOUT_SECONDS == DEFAULT_HOLD_TIMEOUT_S


def test_request_context_evicts_whole_runs_and_records_before_crossing_its_wire_bound() -> None:
    records = (
        _loaded("edit-files", "run-1", "a" * (96 * 1024)),
        _loaded("org-format", "run-2", "b" * (96 * 1024)),
        _loaded("pdf", "run-3", "c" * (96 * 1024)),
        _loaded("regexp", "run-4", "d" * (96 * 1024)),
        _loaded("render", "run-5", "e" * (96 * 1024), ("map_data",)),
    )
    history = [PiMessage(f"run-{number}", role, "x" * (32 * 1024))
               for number in range(1, 17) for role in ("user", "assistant")]
    fixed = {
        "version": 1, "prompt": "current", "model": "qwen", "contextWindow": 32768,
        "brokerCapability": "a" * 43, "providerCapability": "b" * 43,
        "resultCapability": "c" * 43, "maxTurns": 12,
    }

    selected_history, selected_skills, evicted, prompt = pi_runtime.PiRuntimeManager._request_context(
        fixed, "system", PiSkillCatalog(()), history, records, ())
    serialized = json.dumps({**fixed, "history": selected_history, "systemPrompt": prompt,
                             "skillCatalog": [],
                             "retainedSkills": [skill.manifest() for skill in selected_skills]},
                            ensure_ascii=False, separators=(",", ":")).encode()

    assert len(serialized) <= 512 * 1024
    assert {message["role"] for message in selected_history} <= {"user", "assistant"}
    assert {message["text"] for message in selected_history} == {"x" * (32 * 1024)}
    assert evicted


class _Worker:
    def __init__(self, control_dir: Path, events: list[str]) -> None:
        self._control_dir = control_dir
        self._events = events

    def wait(self, *, timeout: int) -> dict[str, int]:
        assert timeout == 1
        self._events.append("worker.wait")
        return {"StatusCode": 0}

class _Containers:
    request: dict[str, object] | None = None

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
        type(self).request = json.loads((control / "request.json").read_text())
        assert type(self).request["contextWindow"] == 32768
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
                 model: str, capability: str, context_len: int) -> None:
        assert upstream == "http://127.0.0.1:8000/v1"
        assert api_key == "secret"
        assert model == "qwen"
        assert len(capability) == 43
        assert context_len == 32768
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


def test_runtime_writes_retained_skill_body_and_loaded_name_to_the_worker_request(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "thread" / "workspace"
    work_dir.mkdir(parents=True)
    _SandboxManager.events = []
    _Containers.request = None
    monkeypatch.setattr(pi_runtime, "current_model_config", lambda: OpenAIConfig(
        "http://localhost:8000/v1", "qwen", "secret", 32768))
    monkeypatch.setattr(pi_runtime, "PiToolBroker", _Broker)
    monkeypatch.setattr(pi_runtime, "PiProviderRelay", _Relay)
    monkeypatch.setattr(pi_runtime, "PiResultSink", _ResultSink)
    retained = _loaded("render", "run-1", "render rules", ("map_data",))

    pi_runtime.PiRuntimeManager(_SandboxManager).run(
        work_dir=str(work_dir), timezone="America/Los_Angeles", prompt="continue",
        history=[PiMessage("run-1", "user", "map it"), PiMessage("run-1", "assistant", "done")],
        system_prompt="be useful", retained_skills=(retained,))

    assert _Containers.request is not None
    assert _Containers.request["retainedSkills"] == [retained.manifest()]
    assert "Already loaded for this conversation: `render`" in _Containers.request["systemPrompt"]


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


def test_result_sink_exposes_only_allowlisted_worker_failure_codes(tmp_path: Path) -> None:
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    sink = pi_runtime.PiResultSink(control, "a" * 43)
    with socket.socket(socket.AF_UNIX) as client:
        client.connect(str(control / "result.sock"))
        client.sendall(json.dumps({
            "capability": "a" * 43, "status": "failed", "code": "turn-bound-exceeded",
            "phase": "prompt", "model": {
                "finish": "length", "sawText": True, "sawThinking": False,
                "completedToolCalls": 0,
            },
        }).encode())
        client.shutdown(socket.SHUT_WR)
    try:
        with pytest.raises(pi_runtime.PiRuntimeError,
                           match=r"turn-bound-exceeded.*model=length,text=yes"):
            sink.receive()
    finally:
        sink.close()

    rejected = pi_runtime.PiResultSink(control, "a" * 43)
    with socket.socket(socket.AF_UNIX) as client:
        client.connect(str(control / "result.sock"))
        client.sendall(json.dumps({
            "capability": "a" * 43, "status": "failed", "code": "host-path:/secret",
            "phase": "prompt", "model": {
                "finish": "none", "sawText": False, "sawThinking": False,
                "completedToolCalls": 0,
            },
        }).encode())
        client.shutdown(socket.SHUT_WR)
    try:
        with pytest.raises(pi_runtime.PiRuntimeError, match="malformed"):
            rejected.receive()
    finally:
        rejected.close()


def test_result_sink_rejects_untrusted_model_diagnostics(tmp_path: Path) -> None:
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    sink = pi_runtime.PiResultSink(control, "a" * 43)
    with socket.socket(socket.AF_UNIX) as client:
        client.connect(str(control / "result.sock"))
        client.sendall(json.dumps({
            "capability": "a" * 43, "status": "failed", "code": "worker-failed",
            "phase": "reply", "model": {
                "finish": "length", "sawText": "yes", "sawThinking": False,
                "completedToolCalls": 0,
            },
        }).encode())
        client.shutdown(socket.SHUT_WR)
    try:
        with pytest.raises(pi_runtime.PiRuntimeError, match="malformed"):
            sink.receive()
    finally:
        sink.close()


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


def test_runtime_preserves_an_allowlisted_worker_failure_code(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "thread" / "workspace"
    work_dir.mkdir(parents=True)
    _SandboxManager.events = []
    monkeypatch.setattr(pi_runtime, "current_model_config", lambda: OpenAIConfig(
        "http://localhost:8000/v1", "qwen", "secret", 32768))
    monkeypatch.setattr(pi_runtime, "PiToolBroker", _Broker)
    monkeypatch.setattr(pi_runtime, "PiProviderRelay", _Relay)

    class FailedResultSink(_ResultSink):
        def receive(self) -> pi_runtime.PiRuntimeResult:
            self.events.append("result.receive")
            raise pi_runtime.PiRuntimeError("Pi worker failed: turn-bound-exceeded")

    class BadWorker(_Worker):
        def wait(self, *, timeout: int) -> dict[str, int]:
            self._events.append("worker.wait")
            return {"StatusCode": 1}

    class BadContainers(_Containers):
        def run(self, _image: str, **kwargs: object) -> BadWorker:
            self._events.append("worker.start")
            return BadWorker(Path(next(iter(kwargs["volumes"]))), self._events)  # type: ignore[index]

    class BadDocker:
        def __init__(self) -> None:
            self.containers = BadContainers(_SandboxManager.events)

    monkeypatch.setattr(pi_runtime, "PiResultSink", FailedResultSink)
    monkeypatch.setattr(_SandboxManager, "_get_docker_client", classmethod(
        lambda cls: BadDocker()))

    with pytest.raises(pi_runtime.PiRuntimeError, match="turn-bound-exceeded"):
        pi_runtime.PiRuntimeManager(_SandboxManager).run(
            work_dir=str(work_dir), timezone="America/Los_Angeles", prompt="hello",
            history=[], system_prompt="be useful",
        )


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


def test_teardown_failure_does_not_mask_the_primary_worker_failure(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
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

    class BadWorker(_Worker):
        def wait(self, *, timeout: int) -> dict[str, int]:
            self._events.append("worker.wait")
            return {"StatusCode": 1}

    class BadContainers(_Containers):
        def run(self, _image: str, **kwargs: object) -> BadWorker:
            self._events.append("worker.start")
            return BadWorker(Path(next(iter(kwargs["volumes"]))), self._events)  # type: ignore[index]

    class BadDocker:
        def __init__(self) -> None:
            self.containers = BadContainers(_SandboxManager.events)

    monkeypatch.setattr(_SandboxManager, "cleanup", classmethod(broken_cleanup))
    monkeypatch.setattr(_SandboxManager, "_get_docker_client", classmethod(
        lambda cls: BadDocker()))

    with pytest.raises(pi_runtime.PiRuntimeError, match="exited unsuccessfully"):
        pi_runtime.PiRuntimeManager(_SandboxManager).run(
            work_dir=str(work_dir), timezone="America/Los_Angeles", prompt="hello",
            history=[], system_prompt="be useful")

    assert "cleanup failed" in caplog.text


def test_committed_result_stays_successful_when_later_teardown_fails(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
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
    committed: list[pi_runtime.PiRuntimeResult] = []
    result = pi_runtime.PiRuntimeManager(_SandboxManager).run(
        work_dir=str(work_dir), timezone="America/Los_Angeles", prompt="hello",
        history=[], system_prompt="be useful", commit=committed.append)

    assert result == pi_runtime.PiRuntimeResult("done", 1)
    assert committed == [result]
    assert "cleanup failed" in caplog.text


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

    with pytest.raises(pi_runtime.PiRuntimeError, match="could not complete"):
        pi_runtime.PiRuntimeManager(_SandboxManager).run(
            work_dir=str(work_dir), timezone="America/Los_Angeles", prompt="hello",
            history=[], system_prompt="be useful",
        )

    assert _SandboxManager.events[-6:] == [
        "broker.stop", "relay.stop", "worker.stop", "worker.wait", "worker.kill", "worker.wait",
    ]
