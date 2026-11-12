"""Durable child lineage controls borrowing a shared execution lifetime."""
from types import SimpleNamespace
from contextlib import contextmanager, nullcontext

import pytest

from assist.browser import authority, manager as browser
from assist.run_service import RunService
from assist.thread_queue import ThreadAffinityQueue
from manage.web import threads


@pytest.mark.parametrize("kind,bound", [("task-completion", True), ("approval", True), ("approval", False)])
def test_fair_successor_reuses_execution_without_git_or_decision_replay(tmp_path, monkeypatch, kind, bound):
    """Special continuations retain execution without repeating their first-slice work."""
    from manage.web.thread_git_lifecycle import ThreadGitLifecycle
    tid = "parent"
    directory = tmp_path / tid
    directory.mkdir()
    authority.mark_new_thread(str(tmp_path), tid)
    service = RunService(str(tmp_path))
    direct = service.create(tid, "general-agent", "Do this task", user_origin=True)
    service.claim(tid, direct.id)
    if kind == "approval":
        action = {"name": "synthetic_action", "args": {}}
        record = service.publish_approval(tid, direct.id, action=action, requests=[action],
            proposal={"kind": "synthetic_action"}, interrupt_id="proposal", checkpoint_id="before")
        owner, _ = service.accept_approval(tid, record.id, {"type": "approve"}, [{"type": "approve"}])
        service.claim(tid, owner.id)
        service.reconcile_approval(tid, record.id, checkpoint_id="after", consumed=True)
        successor = service.continue_approval(tid, owner.id)
    else:
        service.transition(tid, direct.id, "success")
        owner = service.create(tid, "general-agent", "Child findings", origin=kind)
        service.claim(tid, owner.id)
        service.transition(tid, owner.id, "interrupted")
        successor = service.create(tid, "general-agent", None, origin=kind,
                                   work_id=owner.work_id, resume=True, pending_text=owner.text)
    backend = SimpleNamespace(container=SimpleNamespace(id="retained-generation"))
    request = threads._browser_user_request(owner, service.list(tid))
    session = browser.BrowserSession(tid, owner.id, str(directory), str(tmp_path), request,
        work_id=owner.work_id, sandbox_backend=backend,
        sandbox_generation=backend.container.id, run_service=service)
    session.paused = True
    with authority.fence(str(tmp_path), tid) as state:
        state.begin(owner.id, direct.admission_sequence)
        state.add_generation(owner.id, backend.container.id)
    monkeypatch.setattr(browser.BrowserManager, "_sessions", {tid: session})
    monkeypatch.setattr(browser.BrowserManager, "_reconciled_roots", {str(tmp_path)})
    monkeypatch.setattr(browser.BrowserManager, "reconcile_startup", lambda *_a: True)
    monkeypatch.setattr(browser.BrowserManager, "ready_for_browser", lambda *_a: False)
    monkeypatch.setattr(browser.SandboxManager, "_containers", {str(directory): backend.container})
    controls, resumed, git_handoffs = [], [], []
    monkeypatch.setattr(browser, "_bounded_cli", lambda argv, **_k: controls.append(argv) or b"")
    monkeypatch.setattr(browser.BrowserSession, "close", lambda self: setattr(self, "closed", True))
    monkeypatch.setattr(browser.SandboxManager, "cleanup", lambda *_a: None)
    monkeypatch.setattr(threads, "_runs", lambda: service)
    monkeypatch.setattr(threads, "THREAD_QUEUE", ThreadAffinityQueue())
    monkeypatch.setattr(threads.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(threads.MANAGER, "thread_dir", lambda _tid: str(directory))
    monkeypatch.setattr(threads.MANAGER, "thread_default_working_dir", lambda _tid: str(directory))
    monkeypatch.setattr(threads.MANAGER, "touch", lambda *_a: None)
    monkeypatch.setattr(threads, "get_cached_description", lambda *_a: "test")
    monkeypatch.setattr(threads, "_dispatch_pending_after", lambda *_a: None)
    monkeypatch.setattr(threads, "_get_sandbox_backend", lambda *_a, **_k:
                        pytest.fail("fair successor replaced retained execution"))
    lifecycle = ThreadGitLifecycle(SimpleNamespace() if bound else None, str(directory))

    @contextmanager
    def host_fence():
        git_handoffs.append(True)
        raise AssertionError("credentialed Git would stop the retained writer")
        yield

    monkeypatch.setattr(lifecycle, "_host_fence", host_fence)
    monkeypatch.setattr(lifecycle, "preserve_on_error", lambda *_a, **_k: nullcontext())
    monkeypatch.setattr(lifecycle, "cleanup_model", lambda *_a: None)
    monkeypatch.setattr(lifecycle, "finish_visible", lambda *_a: None)
    monkeypatch.setattr(threads.GitLifecycle, "acquire", lambda *_a, **_k: nullcontext(lifecycle))

    class Chat:
        agent = None
        pending_reply = lambda self: None
        get_messages = lambda self: []
        def resume(self):
            resumed.append(True)
            return "finished once"
        def resume_actions(self, _decisions):
            pytest.fail("already consumed decision replayed")

    def get(_tid, **kwargs):
        assert kwargs["sandbox_backend"] is backend
        return Chat()

    monkeypatch.setattr(threads.MANAGER, "get", get)
    monkeypatch.setattr(threads, "_approval_checkpoint", lambda *_a:
                        (SimpleNamespace(next=() if resumed else ("model",), interrupts=()), "after", True))
    threads._execute_run(successor.id, tid)
    result = service.get(tid, successor.id)
    assert result.status == "success", result.error
    assert resumed == [True] and git_handoffs == []
    assert controls == [["docker", "unpause", backend.container.id]]
    assert session.closed and browser.BrowserManager.current_session(tid) is None


@pytest.mark.parametrize("old_child", [False, True])
def test_child_after_fair_resume_borrows_only_its_own_logical_work(tmp_path, monkeypatch, old_child):
    parent_tid, child_tid = "parent", "sub-child"
    (tmp_path / parent_tid).mkdir()
    (tmp_path / child_tid).mkdir()
    (tmp_path / child_tid / ".subagent").touch()
    authority.mark_new_thread(str(tmp_path), parent_tid)
    service = RunService(str(tmp_path))
    old = service.create(parent_tid, "general-agent", "Old task", user_origin=True)
    service.claim(parent_tid, old.id)
    service.transition(parent_tid, old.id, "success")
    owner = service.create(parent_tid, "general-agent", "Current task", user_origin=True)
    service.claim(parent_tid, owner.id)
    service.transition(parent_tid, owner.id, "interrupted")
    successor = service.create(parent_tid, "general-agent", None, work_id=owner.work_id,
                               resume=True, user_event_id=owner.id)
    service.claim(parent_tid, successor.id)
    service.transition(parent_tid, successor.id, "interrupted")
    child = service.create(child_tid, "delegate-agent", "Write findings", mode="child",
                           parent_thread_id=parent_tid,
                           parent_run_id=old.id if old_child else successor.id,
                           dispatch_key="test-child")
    work_dir = str(tmp_path / parent_tid / "domain")
    backend = SimpleNamespace(container=SimpleNamespace(id="shared-generation"))
    session = browser.BrowserSession(
        parent_tid, owner.id, work_dir, str(tmp_path),
        browser.BrowserUserRequest(owner.id, owner.work_id, owner.text, owner.admission_sequence),
        work_id=owner.work_id, sandbox_backend=backend,
        sandbox_generation=backend.container.id, run_service=service)
    session.paused = True
    session.identity = browser._ContainerIdentity(str(tmp_path), "172.20.0.2", backend.container.id)
    with authority.fence(str(tmp_path), parent_tid) as state:
        state.begin(owner.id, owner.admission_sequence)
        state.add_generation(owner.id, backend.container.id)
    controls, executed = [], []
    monkeypatch.setattr(browser.BrowserManager, "_sessions", {parent_tid: session})
    monkeypatch.setattr(browser.SandboxManager, "_containers", {work_dir: backend.container})
    monkeypatch.setattr(browser, "_bounded_cli", lambda argv, **_k: controls.append(argv) or b"")
    monkeypatch.setattr(threads.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(threads.MANAGER, "thread_dir", lambda tid: str(tmp_path / tid))
    monkeypatch.setattr(threads.MANAGER, "thread_default_working_dir", lambda _tid: work_dir)
    monkeypatch.setattr(threads, "_runs", lambda: service)
    monkeypatch.setattr(threads, "THREAD_QUEUE", ThreadAffinityQueue())
    monkeypatch.setattr(threads, "EGRESS_STORE", None)
    monkeypatch.setattr(threads, "_complete_child_handoff", lambda _r: None)
    monkeypatch.setattr(threads, "_dispatch_pending_after", lambda *_a: None)
    monkeypatch.setattr(threads, "_get_sandbox_backend", lambda *_a, **_k:
                        pytest.fail("shared child must never replace the owner generation"))

    def get(_tid, **kwargs):
        assert kwargs["sandbox_backend"] is backend
        assert "browser_tools" not in kwargs
        return SimpleNamespace(message=lambda _text: executed.append(True) or "child findings")

    monkeypatch.setattr(threads.MANAGER, "get", get)
    threads._execute_child_run(child)
    result = service.get(child_tid, child.id)
    assert result.status == ("error" if old_child else "success")
    assert executed == ([] if old_child else [True])
    assert controls == ([] if old_child else [
        ["docker", "unpause", "shared-generation"], ["docker", "pause", "shared-generation"]])
    assert session.paused and not session.closed
    assert browser.BrowserManager.current_session(parent_tid) is session
    assert browser.SandboxManager.current_container(work_dir) is backend.container
