import os
import threading

import pytest
from types import SimpleNamespace
from fastapi.testclient import TestClient

from manage.web import threads
import manage.web as web
from assist.async_subagents import (
    AsyncTaskContext, async_task_context, async_task_tools,
    configure_async_subagent_app,
)
from assist.middleware.url_provenance import DELEGATE_USER_URLS_KEY
from manage.web.protocol_service import SERVICE


TOOLS = {tool.name: tool for tool in async_task_tools}


def _capture_error(errors, operation, *args):
    try:
        operation(*args)
    except BaseException as exc:
        errors.append(exc)


def _root(monkeypatch, tmp_path):
    monkeypatch.setattr(threads.MANAGER, "root_dir", str(tmp_path))
    submitted = []
    monkeypatch.setattr(
        threads._RESUME_SCHEDULER, "submit",
        lambda run_id, tid, **kwargs: submitted.append((run_id, tid, kwargs)))
    return submitted


def _parent_and_metadata(monkeypatch, tmp_path):
    submitted = _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    parent = threads._create_run("parent", "question")
    metadata = {
        "parent_thread_id": "parent",
        "parent_run_id": parent.id,
        "dispatch_key": f"{parent.work_id}:tool-1",
    }
    SERVICE.create_thread("sub-stable", metadata)
    return submitted, parent, metadata


def test_agent_protocol_start_is_idempotent(monkeypatch, tmp_path):
    submitted, parent, metadata = _parent_and_metadata(monkeypatch, tmp_path)

    first = SERVICE.create_run(
        "sub-stable", "context-agent", "inspect files", metadata=metadata)
    second = SERVICE.create_run(
        "sub-stable", "context-agent", "inspect files", metadata=metadata)

    assert second.id == first.id
    assert len(threads._runs().list("sub-stable")) == 1
    assert first.mode == "child"
    assert first.parent_run_id == parent.id
    assert first.work_id == "sub-stable"
    assert {item[:2] for item in submitted} == {(first.id, "sub-stable")}


def test_task_thread_replay_does_not_rewrite_its_identity(monkeypatch, tmp_path):
    _, _, metadata = _parent_and_metadata(monkeypatch, tmp_path)
    marker = os.path.join(threads.MANAGER.thread_dir("sub-stable"), ".subagent")
    before = os.stat(marker)

    SERVICE.create_thread("sub-stable", metadata)

    after = os.stat(marker)
    assert (after.st_dev, after.st_ino, after.st_mtime_ns) == (
        before.st_dev, before.st_ino, before.st_mtime_ns)


def test_task_thread_replay_rejects_different_or_corrupt_identity(
        monkeypatch, tmp_path):
    _, _, metadata = _parent_and_metadata(monkeypatch, tmp_path)
    marker = os.path.join(threads.MANAGER.thread_dir("sub-stable"), ".subagent")

    with pytest.raises(Exception, match="Task metadata conflict"):
        SERVICE.create_thread(
            "sub-stable", {**metadata, "dispatch_key": "different"})
    with open(marker, "w") as stream:
        stream.write("{")
    with pytest.raises(Exception, match="Task metadata conflict"):
        SERVICE.create_thread("sub-stable", metadata)

    with open(marker) as stream:
        assert stream.read() == "{"


def test_task_thread_marker_failure_never_publishes_partial_identity(
        monkeypatch, tmp_path):
    _, parent, metadata = _parent_and_metadata(monkeypatch, tmp_path)
    threads.MANAGER.hard_delete("sub-stable")
    monkeypatch.setattr(
        "assist.thread_manager.json.dump",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(OSError, match="disk full"):
        SERVICE.create_thread("sub-new", {
            **metadata, "parent_run_id": parent.id,
        })

    assert not os.path.exists(threads.MANAGER.thread_dir("sub-new"))


def test_recovery_hides_and_removes_abandoned_task_staging_directory(
        monkeypatch, tmp_path):
    _root(monkeypatch, tmp_path)
    staging = tmp_path / ".subagent-abandoned"
    staging.mkdir()

    assert staging.name not in threads.MANAGER.list()
    threads.queue_recovery_runs()

    assert not staging.exists()


def test_parent_deletion_cannot_leave_a_new_orphan_task(monkeypatch, tmp_path):
    _, _, metadata = _parent_and_metadata(monkeypatch, tmp_path)
    entered = threading.Event()
    release = threading.Event()
    original_rename = os.rename

    def pause_publication(source, target):
        entered.set()
        assert release.wait(timeout=2)
        original_rename(source, target)

    monkeypatch.setattr("assist.thread_manager.os.rename", pause_publication)
    errors = []
    creator = threading.Thread(
        target=lambda: _capture_error(
            errors, SERVICE.create_thread, "sub-race", metadata))
    deleter = threading.Thread(
        target=lambda: _capture_error(
            errors, threads._delete_thread_and_children, "parent"))

    creator.start()
    assert entered.wait(timeout=2)
    deleter.start()
    assert deleter.is_alive()
    release.set()
    creator.join(timeout=2)
    deleter.join(timeout=2)

    assert not creator.is_alive()
    assert not deleter.is_alive()
    assert errors == []
    assert not os.path.exists(threads.MANAGER.thread_dir("parent"))
    assert not os.path.exists(threads.MANAGER.thread_dir("sub-race"))


def test_five_tools_round_trip_through_real_private_asgi(monkeypatch, tmp_path):
    submitted = _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    parent = threads._create_run("parent", "question")
    configure_async_subagent_app(web._agent_app)
    runtime = lambda call_id: SimpleNamespace(tool_call_id=call_id)

    with async_task_context(
            AsyncTaskContext("parent", parent.id, parent.work_id)):
        launched = TOOLS["start_async_task"].func(
            "inspect files", "context-agent", runtime("start"))
        task_id = launched.split("task_id: ", 1)[1].split(".", 1)[0]
        assert task_id in TOOLS["list_async_tasks"].func(runtime("list"))
        assert '"status": "pending"' in TOOLS["check_async_task"].func(
            task_id, runtime("check"))
        assert "Task updated" in TOOLS["update_async_task"].func(
            task_id, "inspect org files", runtime("update"))
        assert "Task cancelled" in TOOLS["cancel_async_task"].func(
            task_id, runtime("cancel"))

    assert submitted


def test_terminal_task_creates_one_ordinary_wake_and_retains_result(
        monkeypatch, tmp_path):
    submitted, _, metadata = _parent_and_metadata(monkeypatch, tmp_path)
    child = SERVICE.create_run(
        "sub-stable", "context-agent", "inspect files", metadata=metadata)
    threads._runs().claim(child.thread_id, child.id)
    child = threads._runs().transition(
        child.thread_id, child.id, "success", result="child result")

    first = threads._complete_child_handoff(child)
    second = threads._complete_child_handoff(child)

    assert second.id == first.id
    wakes = [run for run in threads._runs().list("parent")
             if run.origin == "task-completion"]
    assert len(wakes) == 1
    assert wakes[0].resume is False
    assert "Call check_async_task" in wakes[0].text
    assert "child result" not in wakes[0].text
    assert os.path.isdir(threads.MANAGER.thread_dir("sub-stable"))
    task = SERVICE.get_thread("sub-stable")["values"]["async_task"]
    assert task["status"] == "success"
    assert task["result"] == "child result"
    assert (wakes[0].id, "parent", {}) in submitted


def test_parent_thread_lists_its_tasks(monkeypatch, tmp_path):
    _, _, metadata = _parent_and_metadata(monkeypatch, tmp_path)
    child = SERVICE.create_run(
        "sub-stable", "research-agent", "research", metadata=metadata)

    values = SERVICE.get_thread("parent")["values"]
    assert values["async_tasks"] == [{
        "task_id": "sub-stable",
        "agent_name": "research-agent",
        "description": "research",
        "status": "pending",
        "run_id": child.id,
        "work_id": "sub-stable",
        "parent_thread_id": "parent",
        "result": None,
        "error": None,
        "created_at": child.created_at,
        "updated_at": child.updated_at,
    }]


def test_update_replaces_pending_slice_but_keeps_task_identity(
        monkeypatch, tmp_path):
    _, _, metadata = _parent_and_metadata(monkeypatch, tmp_path)
    first = SERVICE.create_run(
        "sub-stable", "context-agent", "inspect", metadata=metadata)
    update = {**metadata, "dispatch_key": "parent-work:update-tool"}

    second = SERVICE.create_run(
        "sub-stable", "context-agent", "inspect org only",
        multitask_strategy="interrupt", metadata=update)
    replay = SERVICE.create_run(
        "sub-stable", "context-agent", "inspect org only",
        multitask_strategy="interrupt", metadata=update)

    assert threads._runs().get("sub-stable", first.id).status == "cancelled"
    assert replay.id == second.id
    assert second.work_id == first.work_id == "sub-stable"
    assert SERVICE.get_thread("sub-stable")["values"]["async_task"][
        "description"] == "inspect org only"


def test_update_queues_behind_running_slice_without_preemption(
        monkeypatch, tmp_path):
    submitted, _, metadata = _parent_and_metadata(monkeypatch, tmp_path)
    first = SERVICE.create_run(
        "sub-stable", "context-agent", "inspect", metadata=metadata)
    threads._runs().claim(first.thread_id, first.id)
    submitted.clear()

    update = SERVICE.create_run(
        "sub-stable", "context-agent", "inspect org only",
        multitask_strategy="interrupt",
        metadata={**metadata, "dispatch_key": "update"})

    assert update.status == "pending"
    assert SERVICE.get_thread("sub-stable")["values"]["async_task"][
        "status"] == "running"
    assert submitted == []
    first = threads._runs().transition(
        first.thread_id, first.id, "success", result="obsolete")
    assert threads._complete_child_handoff(first) is None
    threads._dispatch_pending_after(first.thread_id, first.id)
    assert submitted == [(update.id, update.thread_id, {"user_priority": False})]


def test_interrupted_child_recovery_creates_one_resume(monkeypatch, tmp_path):
    submitted, _, metadata = _parent_and_metadata(monkeypatch, tmp_path)
    child = SERVICE.create_run(
        "sub-stable", "context-agent", "inspect", metadata=metadata)
    child = threads._runs().claim(child.thread_id, child.id)
    child = threads._runs().transition(child.thread_id, child.id, "interrupted",
                                       active_ms=25)
    submitted.clear()

    threads._recover_interrupted_child(child)
    threads._recover_interrupted_child(child)

    resumes = [run for run in threads._runs().list(child.thread_id)
               if run.dispatch_key == f"task-resume:{child.id}"]
    assert len(resumes) == 1
    assert resumes[0].resume is True
    assert resumes[0].active_ms == 25

    resumed = threads._runs().claim(child.thread_id, resumes[0].id)
    resumed = threads._runs().transition(
        child.thread_id, resumed.id, "success", result="done")
    task = SERVICE.get_thread(child.thread_id)["values"]["async_task"]
    assert task["status"] == "success"
    assert task["result"] == "done"
    with pytest.raises(Exception, match="Task already completed"):
        SERVICE.create_run(
            child.thread_id, "context-agent", "too late",
            multitask_strategy="interrupt",
            metadata={
                "parent_thread_id": child.parent_thread_id,
                "parent_run_id": child.parent_run_id,
                "dispatch_key": "late-update",
            })


def test_cancel_running_task_requests_a_boundary_stop(monkeypatch, tmp_path):
    _, parent, metadata = _parent_and_metadata(monkeypatch, tmp_path)
    child = SERVICE.create_run(
        "sub-stable", "context-agent", "inspect", metadata=metadata)
    assert SERVICE.cancel_run("sub-stable", child.id).status == "cancelled"
    assert SERVICE.cancel_run("sub-stable", child.id).status == "cancelled"

    other_meta = {**metadata, "dispatch_key": "other"}
    other = SERVICE.create_run(
        "sub-stable", "context-agent", "again", metadata=other_meta)
    threads._runs().claim("sub-stable", other.id)
    marker = SERVICE.cancel_run("sub-stable", other.id)
    assert marker.status == "pending"
    assert marker.multitask_strategy == "cancel"
    assert threads._runs().get("sub-stable", other.id).status == "running"
    assert SERVICE.get_thread("sub-stable")["values"]["async_task"][
        "status"] == "running"
    configure_async_subagent_app(web._agent_app)
    with async_task_context(AsyncTaskContext(
            "parent", parent.id, parent.work_id)):
        response = TOOLS["cancel_async_task"].func(
            "sub-stable", SimpleNamespace(tool_call_id="cancel"))
    assert response.startswith("Cancellation requested:")


def test_cancel_interrupted_task_reports_immediate_cancellation(
        monkeypatch, tmp_path):
    _, parent, metadata = _parent_and_metadata(monkeypatch, tmp_path)
    child = SERVICE.create_run(
        "sub-stable", "context-agent", "inspect", metadata=metadata)
    child = threads._runs().claim(child.thread_id, child.id)
    child = threads._runs().transition(child.thread_id, child.id, "interrupted")
    threads._recover_interrupted_child(child)
    configure_async_subagent_app(web._agent_app)

    with async_task_context(AsyncTaskContext(
            "parent", parent.id, parent.work_id)):
        response = TOOLS["cancel_async_task"].func(
            child.thread_id, SimpleNamespace(tool_call_id="cancel"))

    assert response.startswith("Task cancelled:")
    assert SERVICE.get_thread(child.thread_id)["values"]["async_task"][
        "status"] == "cancelled"


def test_cancel_queued_update_cancels_the_running_logical_task(
        monkeypatch, tmp_path):
    _, _, metadata = _parent_and_metadata(monkeypatch, tmp_path)
    active = SERVICE.create_run(
        "sub-stable", "context-agent", "inspect", metadata=metadata)
    threads._runs().claim(active.thread_id, active.id)
    update = SERVICE.create_run(
        "sub-stable", "context-agent", "inspect org",
        multitask_strategy="interrupt",
        metadata={**metadata, "dispatch_key": "update"})

    marker = SERVICE.cancel_run(update.thread_id, update.id)

    assert marker.multitask_strategy == "cancel"
    assert threads._runs().get(update.thread_id, update.id).status == "cancelled"
    assert threads._runs().get(active.thread_id, active.id).status == "running"


def test_active_task_limit_is_parent_scoped(monkeypatch, tmp_path):
    _, parent, metadata = _parent_and_metadata(monkeypatch, tmp_path)
    SERVICE.create_run(
        "sub-stable", "context-agent", "inspect", metadata=metadata)
    monkeypatch.setattr(SERVICE, "MAX_ACTIVE_TASKS_PER_PARENT", 1)

    with pytest.raises(Exception, match="Active task limit reached"):
        SERVICE.create_thread("sub-second", {
            "parent_thread_id": "parent",
            "parent_run_id": parent.id,
            "dispatch_key": "second",
        })


def test_retention_never_prunes_result_before_completion_wake_is_consumed(
        monkeypatch, tmp_path):
    _, parent, metadata = _parent_and_metadata(monkeypatch, tmp_path)
    child = SERVICE.create_run(
        "sub-stable", "context-agent", "inspect", metadata=metadata)
    threads._runs().claim(child.thread_id, child.id)
    child = threads._runs().transition(
        child.thread_id, child.id, "success", result="needed result")
    wake = threads._complete_child_handoff(child)
    monkeypatch.setattr(SERVICE, "MAX_RETAINED_TASKS_PER_PARENT", 1)

    SERVICE.create_thread("sub-next", {
        "parent_thread_id": "parent",
        "parent_run_id": parent.id,
        "dispatch_key": "next",
    })

    assert os.path.isdir(threads.MANAGER.thread_dir(child.thread_id))
    threads._runs().claim("parent", wake.id)
    threads._runs().transition("parent", wake.id, "interrupted")
    unrelated = threads._create_run("parent", "new user turn")
    threads._runs().claim("parent", unrelated.id)
    threads._runs().transition("parent", unrelated.id, "success")
    SERVICE.create_thread("sub-during-pause", {
        "parent_thread_id": "parent",
        "parent_run_id": parent.id,
        "dispatch_key": "during-pause",
    })
    assert os.path.isdir(threads.MANAGER.thread_dir(child.thread_id))
    resumed_wake = threads._create_run(
        "parent", None, work_id=wake.work_id, resume=True)
    threads._runs().claim("parent", resumed_wake.id)
    threads._runs().transition("parent", resumed_wake.id, "success")
    SERVICE.create_thread("sub-after-consume", {
        "parent_thread_id": "parent",
        "parent_run_id": parent.id,
        "dispatch_key": "after-consume",
    })
    assert not os.path.exists(threads.MANAGER.thread_dir(child.thread_id))


def test_task_description_survives_fair_resume_generation(monkeypatch, tmp_path):
    _, _, metadata = _parent_and_metadata(monkeypatch, tmp_path)
    original = SERVICE.create_run(
        "sub-stable", "context-agent", "original", metadata=metadata)
    update = SERVICE.create_run(
        "sub-stable", "context-agent", "updated intent",
        multitask_strategy="interrupt",
        metadata={**metadata, "dispatch_key": "update"})
    update = threads._runs().claim(update.thread_id, update.id)
    update = threads._runs().transition(update.thread_id, update.id, "interrupted")
    threads._recover_interrupted_child(update)

    task = SERVICE.get_thread(original.thread_id)["values"]["async_task"]
    assert task["description"] == "updated intent"


def test_delegate_user_url_seeds_are_frozen_per_instruction_slice(
        monkeypatch, tmp_path):
    _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    first_user = threads._create_run(
        "parent", "Use https://owner.example/first, not the unrelated "
        "https://private.example/history for this work")
    metadata = {
        "parent_thread_id": "parent",
        "parent_run_id": first_user.id,
        "dispatch_key": f"{first_user.work_id}:delegate",
    }
    SERVICE.create_thread("sub-delegate", metadata)
    original = SERVICE.create_run(
        "sub-delegate", "delegate-agent",
        "The model brief repeats https://owner.example/first", metadata=metadata)
    later_user = threads._create_run(
        "parent", "Later use https://owner.example/later too")

    original_cfg = threads._delegate_configurable(original)
    assert original_cfg[DELEGATE_USER_URLS_KEY] == (
        "https://owner.example/first",)

    resume = threads._create_run(
        original.thread_id, None, assistant_id="delegate-agent", mode="child",
        parent_thread_id="parent", parent_run_id=original.parent_run_id,
        dispatch_key="resume", work_id=original.work_id, resume=True,
        delegate_user_urls=original.delegate_user_urls)
    resume_cfg = threads._delegate_configurable(resume)
    assert resume_cfg[DELEGATE_USER_URLS_KEY] == (
        "https://owner.example/first",)

    update = SERVICE.create_run(
        original.thread_id, "delegate-agent",
        "Updated model brief uses https://owner.example/later",
        multitask_strategy="interrupt", metadata={
            "parent_thread_id": "parent",
            "parent_run_id": later_user.id,
            "dispatch_key": "update",
        })
    update_cfg = threads._delegate_configurable(update)
    assert update_cfg[DELEGATE_USER_URLS_KEY] == (
        "https://owner.example/later",)


def test_delegate_admission_includes_owner_interjection_already_accepted(
        monkeypatch, tmp_path):
    _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    active = threads._create_run("parent", "Start the work")
    threads._create_run(
        "parent", "Use https://user:secret@owner.example/interjected#private")
    metadata = {
        "parent_thread_id": "parent",
        "parent_run_id": active.id,
        "dispatch_key": f"{active.work_id}:delegate",
    }
    SERVICE.create_thread("sub-delegate", metadata)

    child = SERVICE.create_run(
        "sub-delegate", "delegate-agent",
        "Read https://owner.example/interjected, not the invented "
        "https://model.example/guess", metadata=metadata)

    assert child.delegate_user_urls == ("https://owner.example/interjected",)


def test_delegate_admission_rejects_untrusted_sms_url(monkeypatch, tmp_path):
    _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    inbound = threads._create_run(
        "parent", "Use https://owner.example/from-sms", sender="+15551234567")
    metadata = {
        "parent_thread_id": "parent",
        "parent_run_id": inbound.id,
        "dispatch_key": f"{inbound.work_id}:delegate",
    }
    SERVICE.create_thread("sub-delegate", metadata)

    child = SERVICE.create_run(
        "sub-delegate", "delegate-agent",
        "Read https://owner.example/from-sms", metadata=metadata)

    assert child.delegate_user_urls == ()


def test_delegate_admission_rejects_brief_credentials_absent_from_owner(
        monkeypatch, tmp_path):
    _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    owner = threads._create_run(
        "parent", "Use https://owner.example/interjected")
    metadata = {
        "parent_thread_id": "parent",
        "parent_run_id": owner.id,
        "dispatch_key": f"{owner.work_id}:delegate",
    }
    SERVICE.create_thread("sub-delegate", metadata)

    child = SERVICE.create_run(
        "sub-delegate", "delegate-agent",
        "Read https://parent-secret:@owner.example/interjected", metadata=metadata)

    assert child.delegate_user_urls == ()


def test_recovery_honors_persisted_cancel_before_resuming(
        monkeypatch, tmp_path):
    submitted, _, metadata = _parent_and_metadata(monkeypatch, tmp_path)
    active = SERVICE.create_run(
        "sub-stable", "context-agent", "inspect", metadata=metadata)
    active = threads._runs().claim(active.thread_id, active.id)
    marker = SERVICE.cancel_run(active.thread_id, active.id)
    submitted.clear()
    monkeypatch.setattr(
        threads.MANAGER, "get",
        lambda *args, **kwargs: pytest.fail("cancelled graph was resumed"))

    threads._recover_child_run(active)

    assert threads._runs().get(active.thread_id, active.id).status == "cancelled"
    assert submitted == [(marker.id, marker.thread_id, {})]


def test_repeated_active_cancel_keeps_the_pending_stop_marker(
        monkeypatch, tmp_path):
    _, _, metadata = _parent_and_metadata(monkeypatch, tmp_path)
    active = SERVICE.create_run(
        "sub-stable", "context-agent", "inspect", metadata=metadata)
    active = threads._runs().claim(active.thread_id, active.id)

    first = SERVICE.cancel_run(active.thread_id, active.id)
    second = SERVICE.cancel_run(active.thread_id, first.id)

    assert second.id == first.id
    assert threads._runs().get(first.thread_id, first.id).status == "pending"


def test_parent_deletion_removes_hidden_tasks(monkeypatch, tmp_path):
    _, _, metadata = _parent_and_metadata(monkeypatch, tmp_path)
    SERVICE.create_run(
        "sub-stable", "context-agent", "inspect", metadata=metadata)
    monkeypatch.setattr(threads.SandboxManager, "cleanup", lambda path: None)

    threads._delete_thread_and_children("parent")

    assert not os.path.exists(threads.MANAGER.thread_dir("parent"))
    assert not os.path.exists(threads.MANAGER.thread_dir("sub-stable"))


def test_parent_deletion_handles_child_history_once(monkeypatch, tmp_path):
    _, _, metadata = _parent_and_metadata(monkeypatch, tmp_path)
    first = SERVICE.create_run(
        "sub-stable", "context-agent", "inspect", metadata=metadata)
    threads._runs().cancel_pending(first.thread_id, first.id)
    SERVICE.create_run(
        "sub-stable", "context-agent", "inspect org",
        metadata={**metadata, "dispatch_key": "next"})

    threads._delete_thread_and_children("parent")

    assert not os.path.exists(threads.MANAGER.thread_dir("parent"))
    assert not os.path.exists(threads.MANAGER.thread_dir("sub-stable"))


def test_parent_deletion_removes_reserved_child_with_no_run(monkeypatch, tmp_path):
    _, _, metadata = _parent_and_metadata(monkeypatch, tmp_path)
    SERVICE.create_thread("sub-empty", metadata)

    threads._delete_thread_and_children("parent")

    assert not os.path.exists(threads.MANAGER.thread_dir("sub-empty"))


def test_completion_input_renders_as_agent_task_note(monkeypatch, tmp_path):
    _root(monkeypatch, tmp_path)
    threads.MANAGER.reserve("parent")
    marker = threads._TASK_COMPLETION_RIDER
    monkeypatch.setattr(
        threads.MANAGER, "get", lambda *args, **kwargs:
        type("Chat", (), {
            "get_messages": lambda self: [
                {"role": "user", "content": marker + "Task ID: sub-123 Status: success"},
                {"role": "assistant", "content": "I used the result."},
            ],
            "pending_reply": lambda self: None,
        })())
    threads._set_status("parent", "ready")

    page = TestClient(web.app).get("/thread/parent").text

    assert "assistant (task)" in page
    assert "Task ID: sub-123 Status: success" in page
    assert '<div class="role">user</div>' not in page
