"""Durable child lineage controls borrowing a shared execution lifetime."""
from types import SimpleNamespace

import pytest

from assist.browser import authority, manager as browser
from assist.run_service import RunService
from assist.thread_queue import ThreadAffinityQueue
from manage.web import threads


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
