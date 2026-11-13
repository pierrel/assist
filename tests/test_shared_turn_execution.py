"""Durable child lineage and uncertain resume control shared execution."""
from types import SimpleNamespace
from contextlib import contextmanager, nullcontext
import os

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


@pytest.mark.parametrize("old_child,unpause_failure,real_docker", [
    (False, False, False), (True, False, False), (False, True, False),
    pytest.param(False, True, True, marks=pytest.mark.skipif(
        not os.getenv("ASSIST_BROWSER_FAIRNESS_TEST_IMAGE"), reason="requires explicit isolated Docker image"))])
def test_child_after_fair_resume_borrows_only_its_own_logical_work(tmp_path, monkeypatch, request, old_child, unpause_failure, real_docker):
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
    if real_docker:
        import docker
        container = docker.from_env().containers.run(
            os.environ["ASSIST_BROWSER_FAIRNESS_TEST_IMAGE"], detach=True,
            network_mode="none", remove=False, entrypoint=["/bin/sh", "-c"],
            command=["while :; do sleep 1; done"])
        def remove_fixture():
            try:
                container.remove(force=True)
            except docker.errors.NotFound:
                pass
        request.addfinalizer(remove_fixture)
        container.pause()
        backend.container = container
    session = browser.BrowserSession(
        parent_tid, owner.id, work_dir, str(tmp_path),
        browser.BrowserUserRequest(owner.id, owner.work_id, owner.text, owner.admission_sequence),
        work_id=owner.work_id, sandbox_backend=backend,
        sandbox_generation=backend.container.id, run_service=service)
    session.paused = True
    session.identity = browser._ContainerIdentity(str(tmp_path), "172.20.0.2", backend.container.id)
    if real_docker:
        session.identity = None  # The execution fence must also stop pre-browser work.
    with authority.fence(str(tmp_path), parent_tid) as state:
        state.begin(owner.id, owner.admission_sequence)
        state.add_generation(owner.id, backend.container.id)
    controls, executed, stopped = [], [], []
    monkeypatch.setattr(browser.BrowserSession, "_capture_storage", lambda _self: None)
    running = False
    original_cli = browser._bounded_cli
    monkeypatch.setattr(browser.BrowserManager, "_sessions", {parent_tid: session})
    monkeypatch.setattr(browser.SandboxManager, "_containers", {work_dir: backend.container})
    def control(argv, **_kwargs):
        nonlocal running
        controls.append(argv)
        if unpause_failure and argv[1] == "unpause":
            if real_docker:
                original_cli(argv, **_kwargs)
                container.reload()
                assert container.status == "running"
            running = True
            raise TimeoutError("unpause took effect but acknowledgment was lost")
        return b""

    monkeypatch.setattr(browser, "_bounded_cli", control)
    if unpause_failure:
        import assist.sandbox_manager as sandbox_manager
        confirm_stop = sandbox_manager.confirm_generation_stopped
        def stop(generation):
            nonlocal running
            assert threads.THREAD_QUEUE.peek_holder() == child_tid
            assert generation == backend.container.id
            if real_docker:
                confirm_stop(generation)
                with pytest.raises(docker.errors.NotFound):
                    container.reload()
            stopped.append(generation)
            running = False
        monkeypatch.setattr(browser, "_kill_container_confirmed", stop)
        monkeypatch.setattr(sandbox_manager, "confirm_generation_stopped", stop)
        monkeypatch.setattr(browser, "disarm_sandbox_browser", lambda *_a: None)
        monkeypatch.setattr(browser, "forget_client", lambda *_a: None)
        monkeypatch.setattr(browser.SandboxManager, "_forget_egress_client", lambda *_a: None)
        monkeypatch.setattr(browser.SandboxManager, "_generation_owners", {
            backend.container.id: (str(tmp_path), parent_tid, owner.id)})
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
    if unpause_failure:
        assert result.status == "error" and not executed
        assert stopped and not running
        assert session.closed and browser.BrowserManager.current_session(parent_tid) is None
        assert browser.SandboxManager.current_container(work_dir) is None
        with authority.fence(str(tmp_path), parent_tid) as state:
            assert state.lease is None
        return
    assert result.status == ("error" if old_child else "success")
    assert executed == ([] if old_child else [True])
    assert controls == ([] if old_child else [
        ["docker", "unpause", "shared-generation"], ["docker", "pause", "shared-generation"]])
    assert session.paused and not session.closed
    assert browser.BrowserManager.current_session(parent_tid) is session
    assert browser.SandboxManager.current_container(work_dir) is backend.container
