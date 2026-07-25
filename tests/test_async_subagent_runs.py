from concurrent.futures import ThreadPoolExecutor
import os

import pytest
from fastapi import HTTPException

from assist.run_service import InvalidRunTransition
from manage.web import threads
from manage.web.protocol_service import SERVICE


def _root(monkeypatch, tmp_path):
    monkeypatch.setattr(threads.MANAGER, "root_dir", str(tmp_path))
    submitted = []
    monkeypatch.setattr(
        threads._RESUME_SCHEDULER, "submit",
        lambda run_id, tid: submitted.append((run_id, tid)))
    return submitted


def test_child_dispatch_is_idempotent(monkeypatch, tmp_path):
    submitted = _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    parent = threads._create_run("parent", "question")

    first = threads._dispatch_child(parent, "work:call", "context-agent", "look")
    second = threads._dispatch_child(parent, "work:call", "context-agent", "look")

    assert second == first
    assert set(submitted) == {(first["run_id"], first["thread_id"])}
    child = threads._runs().get(first["thread_id"], first["run_id"])
    assert len(threads._runs().list(first["thread_id"])) == 1
    assert child.mode == "child"
    assert child.parent_run_id == parent.id


def test_required_child_completion_creates_resume_invocation(monkeypatch, tmp_path):
    submitted = _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    parent = threads._create_run("parent", "question")
    threads._runs().claim("parent", parent.id)
    parent = threads._runs().transition(
        "parent", parent.id, "interrupted", active_ms=321)
    identity = threads._dispatch_child(
        parent, "work:required", "context-agent", "inspect files")
    child = threads._runs().get(identity["thread_id"], identity["run_id"])

    class Chat:
        def message(self, text):
            assert text == "inspect files"
            return "child result"

    monkeypatch.setattr(threads, "_get_sandbox_backend", lambda tid: None)
    monkeypatch.setattr(threads.MANAGER, "get", lambda *args, **kwargs: Chat())
    monkeypatch.setattr(threads.SandboxManager, "cleanup", lambda path: None)
    threads._execute_child_run(child)

    successors = [run for run in threads._runs().list("parent")
                  if run.id != parent.id]
    assert len(successors) == 1
    assert successors[0].work_id == parent.work_id
    assert successors[0].active_ms == 321
    assert successors[0].resume_value == "child result"
    assert submitted[-1] == (successors[0].id, "parent")


def test_cleanup_failure_does_not_prevent_child_handoff(monkeypatch, tmp_path):
    submitted = _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    parent = threads._create_run("parent", "question")
    identity = threads._dispatch_child(
        parent, "work:required", "context-agent", "inspect files")
    child = threads._runs().get(identity["thread_id"], identity["run_id"])

    class Chat:
        def message(self, text):
            return "child result"

    monkeypatch.setattr(threads, "_get_sandbox_backend", lambda tid: None)
    monkeypatch.setattr(threads.MANAGER, "get", lambda *args, **kwargs: Chat())
    monkeypatch.setattr(
        threads.SandboxManager, "cleanup",
        lambda path: (_ for _ in ()).throw(RuntimeError("cleanup failed")))

    threads._execute_child_run(child)

    successor = threads._runs().list("parent")[-1]
    assert successor.resume_value == "child result"
    assert submitted[-1] == (successor.id, "parent")


def test_background_child_posts_follow_up_without_resuming_parent(
        monkeypatch, tmp_path):
    submitted = _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    parent = threads._create_run("parent", "question")
    identity = threads._dispatch_child(
        parent, "work:background", "background-research-agent", "research")
    child = threads._runs().get(identity["thread_id"], identity["run_id"])

    class Chat:
        def message(self, text):
            return "findings"

    monkeypatch.setattr(threads, "_get_sandbox_backend", lambda tid: None)
    monkeypatch.setattr(threads.MANAGER, "get", lambda *args, **kwargs: Chat())
    monkeypatch.setattr(threads.SandboxManager, "cleanup", lambda path: None)
    threads._execute_child_run(child)

    follow_up = threads._runs().list("parent")[-1]
    assert follow_up.origin == "continuation"
    assert "findings" in follow_up.text
    assert follow_up.resume_value is None
    assert submitted[-1] == (follow_up.id, "parent")


def test_recovery_scans_hidden_child_and_reconciles_success_once(
        monkeypatch, tmp_path):
    submitted = _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    parent = threads._create_run("parent", "question")
    threads._runs().claim("parent", parent.id)
    threads._runs().transition("parent", parent.id, "interrupted")
    identity = threads._dispatch_child(
        parent, "work:recover", "context-agent", "inspect")
    child = threads._runs().get(identity["thread_id"], identity["run_id"])
    threads._runs().claim(child.thread_id, child.id)
    threads._runs().transition(child.thread_id, child.id, "success", result="done")
    submitted.clear()

    threads.queue_recovery_runs()
    assert submitted == [(parent.id, "parent")]
    threads._recover_run(threads._runs().get("parent", parent.id))
    threads.queue_recovery_runs()

    successors = [run for run in threads._runs().list("parent")
                  if run.dispatch_key == f"child-resume:{child.id}"]
    assert len(successors) == 1
    assert successors[0].resume_value == "done"
    assert (successors[0].id, "parent") in submitted


def test_recovered_success_stays_success_when_handoff_must_retry(
        monkeypatch, tmp_path):
    _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    parent = threads._create_run("parent", "question")
    identity = threads._dispatch_child(
        parent, "work:recover", "context-agent", "inspect")
    child = threads._runs().get(identity["thread_id"], identity["run_id"])
    child = threads._runs().claim(child.thread_id, child.id)

    class Snapshot:
        next = ()
        interrupts = ()

    class Message:
        type = "ai"
        content = "recovered result"

    class Chat:
        runconfig = {}
        agent = type("Agent", (), {"get_state": lambda self, config: Snapshot()})()

        def get_raw_messages(self):
            return [Message()]

    monkeypatch.setattr(threads.MANAGER, "get", lambda *args, **kwargs: Chat())
    monkeypatch.setattr(
        threads, "_complete_child_handoff",
        lambda run, result: (_ for _ in ()).throw(RuntimeError("retry later")))

    with pytest.raises(RuntimeError, match="retry later"):
        threads._recover_child_run(child)

    recovered = threads._runs().get(child.thread_id, child.id)
    assert recovered.status == "success"
    assert recovered.result == "recovered result"


def test_parent_delete_race_discards_completed_child(monkeypatch, tmp_path):
    _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    parent = threads._create_run("parent", "question")
    identity = threads._dispatch_child(
        parent, "work:required", "context-agent", "inspect")
    child = threads._runs().get(identity["thread_id"], identity["run_id"])
    real_create = threads._create_run

    def delete_parent_then_create(*args, **kwargs):
        threads.MANAGER.hard_delete("parent")
        return real_create(*args, **kwargs)

    monkeypatch.setattr(threads, "_create_run", delete_parent_then_create)

    assert threads._complete_child_handoff(child, "result") is None
    assert not os.path.isdir(threads.MANAGER.thread_dir(child.thread_id))


def test_concurrent_required_children_admit_only_one(monkeypatch, tmp_path):
    _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    parent = threads._create_run("parent", "question")

    def dispatch(key):
        return threads._dispatch_child(
            parent, key, "context-agent", f"inspect {key}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(dispatch, key) for key in ("work:a", "work:b")]
    results = []
    errors = []
    for future in futures:
        try:
            results.append(future.result())
        except InvalidRunTransition as exc:
            errors.append(exc)

    assert len(results) == 1
    assert len(errors) == 1


def test_concurrent_background_children_cannot_cross_chain_cap(
        monkeypatch, tmp_path):
    _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    parent = threads._create_run("parent", "question")
    monkeypatch.setattr(
        threads, "_continuation_chain_len", lambda tid: threads.CHAIN_CAP - 1)

    def dispatch(key):
        return threads._dispatch_child(
            parent, key, "background-research-agent", f"research {key}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(dispatch, key) for key in ("work:a", "work:b")]
    results = []
    errors = []
    for future in futures:
        try:
            results.append(future.result())
        except InvalidRunTransition as exc:
            errors.append(exc)

    assert len(results) == 1
    assert len(errors) == 1


def test_each_child_task_has_an_isolated_checkpoint(monkeypatch, tmp_path):
    _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    parent = threads._create_run("parent", "question")

    background = threads._dispatch_child(
        parent, "work:bg", "background-research-agent", "research")
    required = threads._dispatch_child(
        parent, "work:ctx", "context-agent", "inspect")

    assert background["thread_id"] != required["thread_id"]


def test_child_mapping_is_stable_across_parent_successor(monkeypatch, tmp_path):
    _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    parent = threads._create_run("parent", "question")
    first = threads._dispatch_child(
        parent, "work:task-call", "context-agent", "inspect")
    successor = threads._create_run(
        "parent", None, work_id=parent.work_id, resume_value="result")

    replay = threads._dispatch_child(
        successor, "work:task-call", "context-agent", "inspect")

    assert replay == first


def test_required_child_is_reaped_only_after_parent_consumes_result(
        monkeypatch, tmp_path):
    _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    parent = threads._create_run("parent", "question")
    identity = threads._dispatch_child(
        parent, "work:task-call", "context-agent", "inspect")
    child = threads._runs().get(identity["thread_id"], identity["run_id"])
    successor = threads._complete_child_handoff(child, "result")

    assert successor is not None
    assert os.path.isdir(threads.MANAGER.thread_dir(child.thread_id))
    threads._cleanup_consumed_child(successor)
    assert not os.path.isdir(threads.MANAGER.thread_dir(child.thread_id))


def test_startup_ignores_interrupted_invocation_with_terminal_successor(
        monkeypatch, tmp_path):
    submitted = _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    parent = threads._create_run("parent", "question")
    threads._runs().claim("parent", parent.id)
    threads._runs().transition("parent", parent.id, "interrupted")
    successor = threads._create_run(
        "parent", None, work_id=parent.work_id, resume_value="result")
    threads._runs().claim("parent", successor.id)
    threads._runs().transition("parent", successor.id, "success")

    threads.queue_recovery_runs()

    assert submitted == []


def test_recovery_preserves_child_mapping_until_interrupt_replay(
        monkeypatch, tmp_path):
    submitted = _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    parent = threads._create_run("parent", "question")
    parent = threads._runs().claim("parent", parent.id)
    identity = threads._dispatch_child(
        parent, "work:crash", "context-agent", "inspect")
    child = threads._runs().get(identity["thread_id"], identity["run_id"])

    class Chat:
        def pending_child(self):
            return None

    monkeypatch.setattr(threads.MANAGER, "get", lambda *args, **kwargs: Chat())
    monkeypatch.setattr(threads, "_recovery_decision", lambda tid, text: "redispatch")
    submitted.clear()

    threads._recover_run(parent)

    assert threads._runs().get(child.thread_id, child.id).status == "pending"
    successors = [run for run in threads._runs().list("parent")
                  if run.id != parent.id]
    assert len(successors) == 1
    assert submitted == [(successors[0].id, "parent")]


def test_pending_follower_waits_for_parent_resume_terminal(monkeypatch, tmp_path):
    submitted = _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    parent = threads._create_run("parent", "question")
    threads._runs().claim("parent", parent.id)
    threads._runs().transition("parent", parent.id, "interrupted")
    threads._set_status("parent", "paused", pending_message="question")
    follower = threads._create_run("parent", "newer message")
    child_tid = threads.MANAGER.reserve("child", hidden=True)
    child = threads._create_run(
        child_tid, "inspect", assistant_id="context-agent", mode="child",
        parent_thread_id="parent", parent_run_id=parent.id,
        dispatch_key="work:child", work_id=parent.work_id,
        origin="required-child")
    threads._runs().claim(child.thread_id, child.id)
    threads._runs().transition(child.thread_id, child.id, "success", result="result")

    successor = threads._complete_child_handoff(child, "result")

    assert successor is not None
    assert submitted == [(successor.id, "parent")]
    assert threads._runs().get("parent", follower.id).status == "pending"


def test_protocol_enqueue_waits_behind_interrupted_parent(monkeypatch, tmp_path):
    submitted = _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    parent = threads._create_run("parent", "question")
    threads._runs().claim("parent", parent.id)
    threads._runs().transition("parent", parent.id, "interrupted")
    threads._set_status("parent", "paused", pending_message="question")

    follower = SERVICE.create_run("parent", "general-agent", "newer")

    assert follower.status == "pending"
    assert submitted == []
    with pytest.raises(HTTPException) as caught:
        SERVICE.create_run(
            "parent", "general-agent", "replace", multitask_strategy="interrupt")
    assert caught.value.status_code == 409


def test_protocol_terminal_cancel_is_idempotent(monkeypatch, tmp_path):
    _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    run = threads._create_run("parent", "question")
    threads._runs().claim("parent", run.id)
    done = threads._runs().transition("parent", run.id, "success")

    assert SERVICE.cancel_run("parent", run.id) == done


def test_protocol_cancel_loses_claim_race_truthfully(monkeypatch, tmp_path):
    _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    run = threads._create_run("parent", "question")
    real_cancel = threads._runs().cancel_pending

    def claim_then_cancel(tid, run_id):
        threads._runs().claim(tid, run_id)
        return real_cancel(tid, run_id)

    monkeypatch.setattr(threads._runs(), "cancel_pending", claim_then_cancel)

    with pytest.raises(HTTPException) as caught:
        SERVICE.cancel_run("parent", run.id)
    assert caught.value.status_code == 409
    assert threads._runs().get("parent", run.id).status == "running"


def test_terminal_exit_releases_only_oldest_follower(monkeypatch, tmp_path):
    submitted = _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    first = threads._create_run("parent", "one")
    second = threads._create_run("parent", "two")
    third = threads._create_run("parent", "three")

    threads._dispatch_pending_after("parent", first.id)

    assert submitted == [(second.id, "parent")]
    assert third.status == "pending"


def test_specialized_protocol_run_is_not_injected_into_general_agent(
        monkeypatch, tmp_path):
    _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    threads._create_run(
        "parent", "read only", assistant_id="context-agent")

    assert threads._pending_run_records("parent") == []


def test_protocol_admission_caps_pending_runs(monkeypatch, tmp_path):
    _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    threads._create_run("parent", "one")
    monkeypatch.setattr(SERVICE, "MAX_PENDING_PER_THREAD", 1)

    with pytest.raises(HTTPException) as caught:
        SERVICE.create_run("parent", "general-agent", "two")
    assert caught.value.status_code == 429
    assert len(threads._runs().list("parent")) == 1
