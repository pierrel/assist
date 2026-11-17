"""Native release proof and retirement share one generation critical section."""
import threading
from types import SimpleNamespace

from assist.hermes_runtime import HermesRuntimeManager
from assist.thread_queue import ThreadAffinityQueue


def test_paused_proof_serializes_retirement_before_competitor_admission(tmp_path):
    inspecting = threading.Event()
    finish_inspection = threading.Event()
    retiring = threading.Event()
    removed = threading.Event()
    errors = []
    def reload():
        inspecting.set()
        assert finish_inspection.wait(2)
        if removed.is_set():
            raise RuntimeError("generation disappeared during paused proof")
    container = SimpleNamespace(reload=reload, attrs={"State": {"Paused": True}})
    manager = HermesRuntimeManager(sandbox_manager=SimpleNamespace(
        cleanup_verified=lambda *_args: removed.set()))
    control = tmp_path / "control"
    control.mkdir()
    manager.turns["native"] = SimpleNamespace(work_id="work", work_dir="workspace", paused=True,
        backend=SimpleNamespace(container=container), control_dir=control,
        relay=SimpleNamespace(parked=True, connections={}, stop_admission=lambda: None, close=lambda: None))
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
