import json
import threading

import pytest

from assist.backlog import PendingMessage
from assist.run_service import InvalidRunTransition, RunNotFound, RunService


@pytest.fixture
def service(tmp_path):
    (tmp_path / "t1").mkdir()
    (tmp_path / "t2").mkdir()
    return RunService(str(tmp_path))


def test_create_is_durable_acceptance_commit(service, tmp_path):
    run = service.create("t1", "general-agent", "hello", rider={"tz": "UTC"})

    stored = json.loads((tmp_path / "t1" / "runs.json").read_text())
    assert stored[0]["id"] == run.id
    assert stored[0]["status"] == "pending"
    assert stored[0]["multitask_strategy"] == "enqueue"
    assert service.get("t1", run.id).rider == {"tz": "UTC"}


def test_child_shape_is_validated(service):
    with pytest.raises(ValueError, match="requires parent"):
        service.create("t1", "research-agent", "go", mode="child")
    with pytest.raises(ValueError, match="cannot have parent"):
        service.create("t1", "general-agent", "go", parent_thread_id="parent")

    child = service.create(
        "t2", "research-agent", "go", mode="child",
        parent_thread_id="t1", parent_run_id="parent-run",
        dispatch_key="parent-work:tool-call",
    )
    assert child.parent_run_id == "parent-run"
    assert child.dispatch_key == "parent-work:tool-call"


def test_claim_and_same_status_are_idempotent(service):
    run = service.create("t1", "general-agent", "hello")
    claimed = service.claim("t1", run.id)
    assert claimed.status == "running"
    assert service.claim("t1", run.id) == claimed


def test_same_status_can_record_consumption_once(service):
    run = service.create("t1", "general-agent", "hello")
    consumed = service.transition("t1", run.id, "success", consumed_by="head-run")
    assert consumed.consumed_by == "head-run"
    assert service.transition(
        "t1", run.id, "success", consumed_by="head-run") == consumed


def test_interrupted_protocol_run_cannot_be_restarted(service):
    run = service.create("t1", "general-agent", "hello")
    service.claim("t1", run.id)
    interrupted = service.transition("t1", run.id, "interrupted")
    with pytest.raises(InvalidRunTransition):
        service.transition("t1", interrupted.id, "running")

    resumed = service.create(
        "t1", "general-agent", None, work_id=run.work_id, resume=True,
        pending_text="hello", active_ms=12,
    )
    assert resumed.id != interrupted.id
    assert resumed.work_id == interrupted.work_id


def test_transitions_are_validated(service):
    run = service.create("t1", "general-agent", "hello")
    with pytest.raises(ValueError, match="active_ms cannot be negative"):
        service.transition("t1", run.id, "pending", active_ms=-1)
    service.claim("t1", run.id)
    with pytest.raises(ValueError, match="active_ms cannot be negative"):
        service.transition("t1", run.id, "success", active_ms=-1)
    done = service.transition("t1", run.id, "success")
    assert done.status == "success"
    with pytest.raises(InvalidRunTransition):
        service.cancel("t1", run.id)


def test_cancel_is_idempotent_for_cancelled_run(service):
    run = service.create("t1", "general-agent", "hello")
    cancelled = service.cancel("t1", run.id)
    assert service.cancel("t1", run.id) == cancelled


def test_missing_run_raises(service):
    with pytest.raises(RunNotFound):
        service.get("t1", "missing")


def test_peek_does_not_take_store_lock(service):
    run = service.create("t1", "general-agent", "hello")
    service._lock.acquire()
    try:
        result = []
        reader = threading.Thread(target=lambda: result.extend(service.peek("t1")))
        reader.start()
        reader.join(timeout=0.5)
        assert not reader.is_alive()
        assert result == [run]
    finally:
        service._lock.release()


def test_peek_degrades_on_corrupt_file(service, tmp_path):
    (tmp_path / "t1" / "runs.json").write_text("{")
    assert service.peek("t1") == []


def test_import_legacy_is_idempotent_and_preserves_execution_fields(service):
    record = PendingMessage(
        thread_id="t1", text="later", sender="+1555", rider={"tz": "UTC"},
        enqueued_at="2026-07-24T00:00:00+00:00", origin="continuation",
        id="legacy-id",
    )

    first = service.import_legacy("t1", [record])
    second = service.import_legacy("t1", [record])

    assert first == second
    assert len(service.list("t1")) == 1
    assert first[0].id == "legacy-id"
    assert first[0].sender == "+1555"
    assert first[0].origin == "continuation"


def test_import_legacy_rejects_id_collision(service):
    service.create("t1", "general-agent", "new", run_id="same")
    with pytest.raises(ValueError, match="conflicts"):
        service.import_legacy(
            "t1", [PendingMessage(thread_id="t1", text="old", id="same")])


def test_concurrent_creates_do_not_drop_runs(service):
    threads = [threading.Thread(
        target=service.create, args=("t1", "general-agent", str(i)))
        for i in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(service.list("t1")) == 20


def test_dispatch_key_create_is_atomic(service):
    created = []

    def create():
        created.append(service.create(
            "t1", "context-agent", "inspect", mode="child",
            parent_thread_id="parent", parent_run_id="run",
            dispatch_key="work:tool"))

    threads = [threading.Thread(target=create) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len({run.id for run in created}) == 1
    assert len(service.list("t1")) == 1


def test_scan_children_does_not_parse_visible_run_histories(service, tmp_path):
    service.create("t1", "general-agent", "visible")
    child = service.create(
        "t2", "context-agent", "hidden", mode="child",
        parent_thread_id="t1", parent_run_id="parent-run",
        dispatch_key="work:tool", hidden=True)

    assert service.scan_children() == [child]
