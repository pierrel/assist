import json
import os
import threading

import pytest

from assist.backlog import PendingMessage
import assist.run_service as run_service_module
from assist.record_store import PerThreadJsonStore
from assist.run_service import (InvalidRunTransition, RunNotFound,
                                RunService, RunStoreUnavailable)


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
    assert "delegate_user_urls" not in stored[0]
    assert service.get("t1", run.id).rider == {"tz": "UTC"}


def test_visible_run_keeps_private_location_across_a_successor(service, tmp_path):
    location = {"lat": 37.7749, "lon": -122.4194,
                "observed_at": "2026-08-15T20:00:00+00:00"}
    first = service.create("t1", "general-agent", "walk from here", location=location)
    service.claim("t1", first.id)
    service.transition("t1", first.id, "interrupted")
    successor = service.create(
        "t1", "general-agent", None, work_id=first.work_id, resume=True,
        pending_text=first.text, location=first.location,
    )

    stored = json.loads((tmp_path / "t1" / "runs.json").read_text())
    assert stored[-1]["location"] == location
    assert successor.location == location


def test_delegate_user_urls_round_trip_only_on_the_child_slice(service, tmp_path):
    child = service.create(
        "sub-t3", "delegate-agent", "read https://owner.example/page",
        mode="child", parent_thread_id="t1", parent_run_id="parent-run",
        dispatch_key="parent-work:delegate",
        delegate_user_urls=("https://owner.example/page",),
    )

    stored = json.loads((tmp_path / "sub-t3" / "runs.json").read_text())
    assert stored[0]["delegate_user_urls"] == ["https://owner.example/page"]
    assert service.get(child.thread_id, child.id).delegate_user_urls == (
        "https://owner.example/page",)


def test_child_shape_is_validated(service, tmp_path):
    with pytest.raises(ValueError, match="requires parent"):
        service.create("t1", "research-agent", "go", mode="child")
    with pytest.raises(ValueError, match="cannot have parent"):
        service.create("t1", "general-agent", "go", parent_thread_id="parent")
    child = service.create(
        "sub-t3", "research-agent", "go", mode="child",
        parent_thread_id="t1", parent_run_id="parent-run",
        dispatch_key="parent-work:tool-call",
    )
    assert child.parent_run_id == "parent-run"
    assert child.dispatch_key == "parent-work:tool-call"

    (tmp_path / "sub-visible").mkdir()
    (tmp_path / "sub-visible" / "description.txt").write_text("visible")
    with pytest.raises(ValueError, match="visible thread"):
        service.create(
            "sub-visible", "research-agent", "go", mode="child",
            parent_thread_id="t1", parent_run_id="parent-run",
            dispatch_key="work:visible-collision")


def test_claim_and_same_status_are_idempotent(service):
    run = service.create("t1", "general-agent", "hello")
    claimed = service.claim("t1", run.id)
    assert claimed.status == "running"
    assert service.claim("t1", run.id) == claimed


def test_initialization_failure_writes_head_and_followers_once(service, monkeypatch):
    head = service.create("t1", "general-agent", "first", work_id="head-work")
    follower = service.create("t1", "general-agent", "later", work_id="follower-work")
    writes = []
    write = service._write

    def record_write(tid, runs):
        writes.append((tid, [run.status for run in runs]))
        write(tid, runs)

    monkeypatch.setattr(service, "_write", record_write)

    failed = service.fail_initialization("t1", head.id, "setup failed")

    assert [run.id for run in failed] == [head.id, follower.id]
    assert [run.status for run in service.list("t1")] == ["error", "error"]
    assert writes == [("t1", ["error", "error"])]


def test_list_with_revision_sanitizes_an_unreadable_run_store(service, tmp_path):
    (tmp_path / "t1" / "runs.json").write_text("not json")

    with pytest.raises(RunStoreUnavailable, match="Run store is unavailable"):
        service.list_with_revision("t1")


def test_non_utf8_run_store_is_unavailable_to_every_phone_projection(service, tmp_path):
    (tmp_path / "t1" / "runs.json").write_bytes(b"[\xff]")

    with pytest.raises(RunStoreUnavailable, match="Run store is unavailable"):
        service.list_with_revision("t1")


@pytest.mark.parametrize("stored", ["[true]", "{}", "[null]"])
def test_malformed_run_store_is_unavailable_to_every_durable_operation(
        service, tmp_path, stored
):
    (tmp_path / "t1" / "runs.json").write_text(stored)

    for operation in (
            lambda: service.list("t1"),
            lambda: service.get("t1", "run-a"),
            lambda: service.create("t1", "general-agent", "hello"),
            lambda: service.list_with_revision("t1"),
    ):
        with pytest.raises(RunStoreUnavailable, match="Run store is unavailable"):
            operation()


@pytest.mark.parametrize(
    "field,value",
    [
        ("work_id", "w" * 257),
        ("rider", ["not", "a", "map"]),
        ("resume_decision", "not a map"),
        ("text", ["not", "text"]),
    ],
)
def test_semantically_malformed_run_record_is_unavailable(field, value, service, tmp_path):
    run = service.create("t1", "general-agent", "hello")
    stored = json.loads((tmp_path / "t1" / "runs.json").read_text())
    stored[0][field] = value
    (tmp_path / "t1" / "runs.json").write_text(json.dumps(stored))

    with pytest.raises(RunStoreUnavailable, match="Run store is unavailable"):
        service.get("t1", run.id)


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


def test_cancel_logical_writes_its_pending_receipt_with_one_atomic_snapshot(service, monkeypatch):
    predecessor = service.create("t1", "general-agent", "hello", work_id="work-a")
    service.claim("t1", predecessor.id)
    service.transition("t1", predecessor.id, "interrupted")
    successor = service.create("t1", "general-agent", "hello", work_id="work-a")
    other = service.create("t1", "general-agent", "later", work_id="work-b")
    writes = []
    write = service._write

    def record_write(tid, runs):
        writes.append((tid, [(run.id, run.status, run.cancel_cleanup) for run in runs]))
        write(tid, runs)

    monkeypatch.setattr(service, "_write", record_write)
    snapshot = service.cancel_logical("t1", predecessor.id)

    assert [(run.id, run.status, run.cancel_cleanup) for run in snapshot] == [
        (predecessor.id, "cancelled", "pending"),
        (successor.id, "cancelled", None),
        (other.id, "pending", None),
    ]
    assert writes == [("t1", [(predecessor.id, "cancelled", "pending"),
                               (successor.id, "cancelled", None),
                               (other.id, "pending", None)])]
    assert service.complete_cancel_cleanup("t1", predecessor.id).cancel_cleanup == "complete"
    assert service.complete_cancel_cleanup("t1", predecessor.id).cancel_cleanup == "complete"


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


def test_observation_token_detects_same_size_overwrite_with_restored_mtime(service, tmp_path):
    service.create("t1", "general-agent", "hello")
    path = tmp_path / "t1" / "runs.json"
    before = service.observation_token("t1")
    metadata = path.stat()
    path.write_text(path.read_text().replace("hello", "world"))
    os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))

    assert service.observation_token("t1") != before


def test_observation_token_detects_replace_with_restored_mtime(service, tmp_path):
    service.create("t1", "general-agent", "hello")
    path = tmp_path / "t1" / "runs.json"
    before = service.observation_token("t1")
    metadata = path.stat()
    replacement = path.with_suffix(".replacement")
    replacement.write_text(path.read_text().replace("hello", "world"))
    os.utime(replacement, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    os.replace(replacement, path)

    assert service.observation_token("t1") != before


def test_snapshot_parse_keeps_one_open_file_coherent_across_a_replace(
        service, tmp_path, monkeypatch
):
    service.create("t1", "general-agent", "hello")
    path = tmp_path / "t1" / "runs.json"
    original_load = run_service_module.json.load
    replacements = 0

    def replace_before_read(stream):
        nonlocal replacements
        replacements += 1
        if replacements == 1:
            replacement = path.with_suffix(".replacement")
            replacement.write_text(path.read_text().replace("hello", "world"))
            os.replace(replacement, path)
        return original_load(stream)

    monkeypatch.setattr(run_service_module.json, "load", replace_before_read)

    runs, _revision, token = service.list_with_revision_and_token("t1")

    assert replacements == 2
    assert runs[0].text == "world"
    assert token == service.observation_token("t1")


def test_snapshot_parse_retries_one_in_place_change_then_returns_a_coherent_token(
        service, tmp_path, monkeypatch
):
    service.create("t1", "general-agent", "hello")
    path = tmp_path / "t1" / "runs.json"
    original_load = run_service_module.json.load
    loads = 0

    def change_once(stream):
        nonlocal loads
        loads += 1
        if loads == 1:
            path.write_text(path.read_text().replace("hello", "world"))
        return original_load(stream)

    monkeypatch.setattr(run_service_module.json, "load", change_once)

    runs, _revision, token = service.list_with_revision_and_token("t1")

    assert loads == 2
    assert runs[0].text == "world"
    assert token == service.observation_token("t1")


def test_run_observation_refuses_symlink_and_nonregular_store_files(service, tmp_path):
    path = tmp_path / "t1" / "runs.json"
    target = tmp_path / "target.json"
    target.write_text("[]")
    path.symlink_to(target)

    with pytest.raises(RunStoreUnavailable, match="Run store is unavailable"):
        service.observation_token("t1")
    with pytest.raises(RunStoreUnavailable, match="Run store is unavailable"):
        service.list_with_revision_and_token("t1")

    path.unlink()
    path.mkdir()
    with pytest.raises(RunStoreUnavailable, match="Run store is unavailable"):
        service.observation_token("t1")


def test_revision_advances_only_after_a_successful_durable_write(service):
    assert service.revision("t1") == 0

    run = service.create("t1", "general-agent", "hello")

    assert service.revision("t1") == 1
    assert service.transition("t1", run.id, "pending") == run
    assert service.revision("t1") == 1
    assert service.claim("t1", run.id).status == "running"
    assert service.revision("t1") == 2


def test_failed_write_does_not_advance_revision(service, monkeypatch):
    monkeypatch.setattr(
        PerThreadJsonStore, "_write",
        lambda *_args: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        service.create("t1", "general-agent", "hello")

    assert service.revision("t1") == 0


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
            "sub-t3", "context-agent", "inspect", mode="child",
            parent_thread_id="parent", parent_run_id="run",
            dispatch_key="work:tool"))

    threads = [threading.Thread(target=create) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len({run.id for run in created}) == 1
    assert len(service.list("sub-t3")) == 1


def test_scan_children_does_not_parse_visible_run_histories(service, tmp_path):
    service.create("t1", "general-agent", "visible")
    child = service.create(
        "sub-t3", "context-agent", "hidden", mode="child",
        parent_thread_id="t1", parent_run_id="parent-run",
        dispatch_key="work:tool")

    assert service.scan_children() == [child]


def test_child_admission_does_not_parse_visible_run_histories(service, tmp_path):
    (tmp_path / "t1" / "runs.json").write_text("not json")

    child = service.create(
        "sub-t3", "research-agent", "hidden", mode="child",
        parent_thread_id="parent", parent_run_id="parent-run",
        dispatch_key="work:background", origin="background")

    assert child.status == "pending"


def test_child_admission_recovers_an_abandoned_empty_directory(service, tmp_path):
    (tmp_path / "sub-abandoned").mkdir()

    child = service.create(
        "sub-abandoned", "context-agent", "inspect", mode="child",
        parent_thread_id="parent", parent_run_id="parent-run",
        dispatch_key="work:abandoned")

    assert child.status == "pending"
    assert (tmp_path / "sub-abandoned" / ".subagent").is_file()


def test_failed_child_persistence_removes_new_hidden_directory(
        service, tmp_path, monkeypatch):
    monkeypatch.setattr(
        service, "_write", lambda *_: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(OSError, match="disk full"):
        service.create(
            "sub-failed", "context-agent", "inspect", mode="child",
            parent_thread_id="parent", parent_run_id="parent-run",
            dispatch_key="work:failed")

    assert not (tmp_path / "sub-failed").exists()
