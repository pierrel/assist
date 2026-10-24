"""Parent browser holds resume after hidden child execution releases ownership."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from assist.browser import authority, manager as browser
from assist.sandbox_manager import SandboxManager
from assist.thread_queue import ThreadAffinityQueue, ThreadPauseRequested
from manage.web import phone_api, threads, thread_git_lifecycle
from manage.web.protocol_service import SERVICE


@pytest.mark.parametrize("bound", [False, True])
@pytest.mark.parametrize("outcome", ["success", "error", "cancel", "pause", "cleanup_failure"])
def test_parent_hold_wakes_after_child_scope(monkeypatch, tmp_path, bound, outcome):
    monkeypatch.setattr(threads.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(threads, "THREAD_QUEUE", ThreadAffinityQueue())
    monkeypatch.setattr(SandboxManager, "_containers", {})
    monkeypatch.setattr(SandboxManager, "_generation_owners", {})
    monkeypatch.setattr(SandboxManager, "_egress_client_ips", {})
    monkeypatch.setattr(threads, "_get_status", lambda _tid: {})
    monkeypatch.setattr(threads, "_is_pi_thread", lambda _tid: False)
    monkeypatch.setattr(threads, "EGRESS_STORE", None)
    monkeypatch.setattr(threads, "configured_directory", lambda: None)
    monkeypatch.setattr(browser, "configured_directory", lambda: None)
    monkeypatch.setattr(browser, "_bounded_cli", lambda *_args, **_kwargs: b"")
    monkeypatch.setattr(browser, "_sandbox_generations", lambda *_args: set())
    monkeypatch.setattr(browser.BrowserManager, "current_session", lambda _tid: None)
    submitted, drains, retries = [], [], []
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit",
                        lambda run_id, tid, **kwargs: submitted.append((run_id, tid, kwargs)))
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "promote", lambda _tid: None)
    monkeypatch.setattr(threads, "_schedule_browser_revocation_retry", retries.append)
    threads.MANAGER.reserve("parent")
    parent = threads._create_run("parent", "question")
    threads._runs().claim("parent", parent.id)
    threads._runs().transition("parent", parent.id, "success", result="started task")
    metadata = {"parent_thread_id": "parent", "parent_run_id": parent.id,
                "dispatch_key": "start-task"}
    SERVICE.create_thread("sub-task", metadata)
    child = SERVICE.create_run("sub-task", "context-agent", "inspect files", metadata=metadata)
    submitted.clear()
    work_dir = threads.MANAGER.thread_default_working_dir("parent")
    if bound:
        # Keep the real Git checkout lock and verified teardown; publication
        # itself is unrelated to the reset notification being tested.
        from pathlib import Path
        Path(work_dir, ".git").mkdir(parents=True)
        monkeypatch.setattr(thread_git_lifecycle, "ThreadGit", lambda *_args: object())
        monkeypatch.setattr(threads.GitLifecycle, "resume", lambda _self: None)
        monkeypatch.setattr(threads.GitLifecycle, "finish_child", lambda *_args: None)
        monkeypatch.setattr(threads.GitLifecycle, "_commit", lambda *_args: None)
    entered, release = Event(), Event()

    def stop(*_args, **_kwargs):
        if outcome == "cleanup_failure":
            raise RuntimeError("stop unconfirmed")

    container = SimpleNamespace(id="child-generation", kill=stop, wait=stop)
    monkeypatch.setattr("assist.sandbox_manager.confirm_generation_stopped", stop)

    def sandbox(*_args, **_kwargs):
        with authority.fence(str(tmp_path), "parent") as state:
            state.begin(child.id, 0)
            state.add_generation(child.id, container.id)
        SandboxManager._containers[work_dir] = container
        SandboxManager._generation_owners[container.id] = (str(tmp_path), "parent", child.id)
        return SimpleNamespace(container=container)

    def message(_text):
        entered.set()
        assert release.wait(5)
        if outcome == "error":
            raise RuntimeError("task failed")
        if outcome in {"cancel", "pause"}:
            raise ThreadPauseRequested()
        return "task result"

    def wake(tid):
        if entered.is_set() and release.is_set():
            assert threads.THREAD_QUEUE.peek_holder() is None
            if bound:
                from assist.git_sync import _workspace_lock
                with _workspace_lock(threads.MANAGER.thread_dir("parent")):
                    pass
        drains.append(threads._drain_held_browser_events(tid))

    monkeypatch.setattr(threads, "_queue_browser_revocation", wake)
    monkeypatch.setattr(threads, "_get_sandbox_backend", sandbox)
    monkeypatch.setattr(threads.MANAGER, "get",
                        lambda *_args, **_kwargs: SimpleNamespace(message=message))
    with ThreadPoolExecutor(max_workers=1) as pool:
        executing = pool.submit(threads._execute_child_run, child)
        try:
            assert entered.wait(5)
            held, busy, replay = phone_api._submit_existing("parent", "follow up", "follow-up")
            assert busy and not replay
            assert held.status == "revocation_pending"
            assert held.browser_reset_run_id == child.id
            assert drains == [False]
            assert retries == []
            assert submitted == []
            if outcome == "cancel":
                SERVICE.cancel_run(child.thread_id, child.id)
        finally:
            release.set()
        executing.result(timeout=5)

    if outcome == "cleanup_failure":
        assert drains == [False, False]
        assert threads._runs().get("parent", held.id).status == "revocation_pending"
        assert SandboxManager.current_container(work_dir) is container
        with authority.fence(str(tmp_path), "parent") as state:
            assert state.lease["generations"] == [container.id]
        return

    assert drains == [False, True]
    assert threads._runs().get("parent", held.id).status == "pending"
    assert SandboxManager.current_container(work_dir) is None
    with authority.fence(str(tmp_path), "parent") as state:
        assert state.covered and state.lease is None
    assert (held.id, "parent", {"user_priority": True}) in submitted
    wakes = [run for run in threads._runs().list("parent") if run.origin == "task-completion"]
    terminal = threads._runs().get(child.thread_id, child.id)
    if outcome in {"success", "error"}:
        assert terminal.status == outcome
        assert len(wakes) == 1
        assert all(run_id == held.id for run_id, tid, _kwargs in submitted if tid == "parent")
        threads._complete_child_handoff(terminal)
        assert [run.id for run in threads._runs().list("parent")
                if run.origin == "task-completion"] == [wakes[0].id]
        threads._runs().claim("parent", held.id)
        threads._runs().transition("parent", held.id, "success", result="follow-up done")
        submitted.clear()
        threads._dispatch_pending_after("parent", held.id)
        assert submitted == [(wakes[0].id, "parent", {"user_priority": False})]
    else:
        assert terminal.status == ("cancelled" if outcome == "cancel" else "interrupted")
        assert wakes == []


@pytest.mark.parametrize("existing_wake", [False, True])
def test_handoff_selects_user_during_reset_notification_gap(monkeypatch, tmp_path, existing_wake):
    monkeypatch.setattr(threads.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(threads, "_get_status", lambda _tid: {})
    monkeypatch.setattr(browser, "configured_directory", lambda: None)
    monkeypatch.setattr(browser, "_bounded_cli", lambda *_args, **_kwargs: b"")
    monkeypatch.setattr(browser.BrowserManager, "current_session", lambda _tid: None)
    submitted = []
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit",
                        lambda run_id, tid, **kwargs: submitted.append((run_id, tid, kwargs)))
    threads.MANAGER.reserve("parent")
    parent = threads._create_run("parent", "question")
    threads._runs().claim("parent", parent.id)
    threads._runs().transition("parent", parent.id, "success", result="started task")
    metadata = {"parent_thread_id": "parent", "parent_run_id": parent.id,
                "dispatch_key": "start-task"}
    SERVICE.create_thread("sub-task", metadata)
    child = SERVICE.create_run("sub-task", "context-agent", "inspect files", metadata=metadata)
    threads._runs().claim(child.thread_id, child.id)
    child = threads._runs().transition(child.thread_id, child.id, "success", result="task result")
    held = threads._create_run("parent", "follow up", user_origin=True,
                               revocation_pending=True, browser_reset_run_id=child.id)
    if existing_wake:
        threads._create_run("parent", "task completed", origin="task-completion",
                            work_id=f"task-completion:{child.work_id}",
                            dispatch_key=f"task-completion:{child.id}")
    submitted.clear()
    promoted, release = Event(), Event()

    def pause_notification(_tid):
        assert threads._runs().get("parent", held.id).status == "pending"
        promoted.set()
        assert release.wait(5)

    monkeypatch.setattr(threads._RESUME_SCHEDULER, "promote", pause_notification)
    with ThreadPoolExecutor(max_workers=1) as pool:
        resetting = pool.submit(threads._drain_held_browser_events, "parent")
        try:
            assert promoted.wait(5)
            wake = threads._complete_child_handoff(child)
            assert wake is not None
            assert submitted == [(held.id, "parent", {"user_priority": True})]
            threads._runs().claim("parent", held.id)
            assert threads._runs().get("parent", wake.id).status == "pending"
        finally:
            release.set()
        assert resetting.result(timeout=5)
    assert submitted == [(held.id, "parent", {"user_priority": True})]
    threads._runs().transition("parent", held.id, "success", result="follow-up done")
    threads._dispatch_pending_after("parent", held.id)
    assert submitted[-1] == (wake.id, "parent", {"user_priority": False})
