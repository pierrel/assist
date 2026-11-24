"""Native release proof and retirement share one generation critical section."""
import threading
from types import SimpleNamespace

import pytest

from assist.hermes_runtime import HermesRuntimeManager
from assist.thread_queue import ThreadAffinityQueue


def test_paused_proof_serializes_retirement_before_competitor_admission(tmp_path):
    inspecting = threading.Event()
    finish_inspection = threading.Event()
    retiring = threading.Event()
    removed = threading.Event()
    errors = []
    def parked():
        inspecting.set()
        assert finish_inspection.wait(2)
        if removed.is_set():
            raise RuntimeError("generation disappeared during paused proof")
        return True
    class Relay:
        parked = property(lambda self: parked())
        connections = {}
        def stop_admission(self): pass
        def close(self): pass
    container = SimpleNamespace()
    manager = HermesRuntimeManager(sandbox_manager=SimpleNamespace(
        cleanup_verified=lambda *_args: removed.set()))
    control = tmp_path / "control"
    control.mkdir()
    manager.turns["native"] = SimpleNamespace(work_id="work", work_dir="workspace", paused=True,
        backend=SimpleNamespace(container=container), control_dir=control,
        relay=Relay())
    queue = ThreadAffinityQueue()
    def release():
        try:
            with queue.acquire("native", before_release=lambda: manager.release_slice("native")):
                pass
        except Exception as error:
            errors.append(error)
    def retire():
        retiring.set()
        manager.retire("native")
    first = threading.Thread(target=release)
    second = threading.Thread(target=retire)
    first.start()
    try:
        assert inspecting.wait(2)
        second.start()
        assert retiring.wait(2)
        assert not removed.wait(0.1), "retirement overtook the in-flight paused proof"
    finally:
        finish_inspection.set()
        first.join(2)
        if second.ident is not None:
            second.join(2)
    assert not first.is_alive() and not second.is_alive()
    assert errors == [] and removed.is_set()
    assert queue.current_handle() is None
    with queue.acquire("competitor", wait_timeout_s=0.01):
        pass


@pytest.mark.parametrize("failure", ["relay", "map"])
def test_retirement_retry_keeps_verified_container_stop_after_cleanup_failure(
        tmp_path, monkeypatch, failure):
    from assist.sandbox_manager import SandboxManager
    from assist import sandbox_expiry

    work_dir = str(tmp_path / "workspace")
    control = tmp_path / "control"
    control.mkdir()
    stopped = []
    container = SimpleNamespace(id="private-generation", kill=lambda: stopped.append("kill"),
                                wait=lambda **_kwargs: stopped.append("wait"))
    monkeypatch.setattr(SandboxManager, "_containers", {work_dir: container})
    monkeypatch.setattr(SandboxManager, "_generation_owners", {})
    monkeypatch.setattr(SandboxManager, "_egress_client_ips", {})
    monkeypatch.setattr(sandbox_expiry, "cancel", lambda _generation: None)
    drain_attempts = []

    def close():
        drain_attempts.append(True)
        if failure == "relay" and len(drain_attempts) == 1:
            raise RuntimeError("fixture cleanup unconfirmed")

    map_path = tmp_path / "maps" / "client-map.json"
    if failure == "map":
        map_path.parent.mkdir()
        map_path.write_text("[]")
        monkeypatch.setattr(SandboxManager, "_egress_client_ips", {
            work_dir: (str(map_path.parent), "172.30.0.2", container.id)})

    runtime = HermesRuntimeManager()
    runtime.turns["native"] = SimpleNamespace(
        backend=SimpleNamespace(container=container), work_dir=work_dir,
        control_dir=control, relay=SimpleNamespace(stop_admission=lambda: None, close=close))
    with pytest.raises(RuntimeError if failure == "relay" else ValueError,
                       match="cleanup unconfirmed" if failure == "relay" else "invalid proxy client map"):
        runtime.retire("native")
    assert "native" in runtime.turns
    assert (SandboxManager.current_container(work_dir) is container) == (failure == "map")
    if failure == "map":
        map_path.write_text("{}")
    runtime.retire("native")
    assert "native" not in runtime.turns
    assert stopped == ["kill", "wait"] * (2 if failure == "map" else 1)
    assert len(drain_attempts) == (1 if failure == "map" else 2)
    assert not control.exists()


def test_model_revocation_keeps_cleanup_owner_and_rejects_other_work():
    closed = []
    runtime = HermesRuntimeManager()
    turn = SimpleNamespace(work_id="original", relay=SimpleNamespace(
        close=lambda *, drain_egress=True: closed.append(drain_egress)))
    runtime.turns["native"] = turn
    assert not runtime.revoke_model("native", "unrelated")
    assert not closed and runtime.turns["native"] is turn
    assert runtime.revoke_model("native", "original")
    assert closed == [False] and runtime.turns["native"] is turn
