from types import SimpleNamespace
from contextlib import contextmanager
import pytest

from assist.browser.manager import BrowserManager, BrowserSession
from assist.thread_queue import ThreadAffinityQueue, QueueWaitTimeout
from manage.web import threads
from tests.test_fair_scheduling_integration import wired  # existing production-path fixture


@pytest.mark.parametrize("same_work", [True, False])
def test_paused_retained_session_terminal_queue_wait(wired, monkeypatch, same_work):
    tid, _calls = wired
    runs = threads._runs()
    source = runs.create(tid, "general-agent", "first", user_origin=True)
    runs.claim(tid, source.id)
    runs.transition(tid, source.id, "interrupted")
    successor = runs.create(tid, "general-agent", None, resume=True,
                            work_id=source.work_id, user_event_id=source.id)
    work_dir = threads.MANAGER.thread_default_working_dir(tid)
    container = SimpleNamespace(id="exact-frozen-generation")
    session = BrowserSession(tid, source.id, work_dir, threads.MANAGER.root_dir,
                             None, work_id=source.work_id,
                             sandbox_generation=container.id,
                             sandbox_backend=SimpleNamespace(container=container),
                             run_service=runs)
    session.paused = True
    if not same_work:
        session.work_id = "newer-logical-work"
    stopped = []
    monkeypatch.setattr(threads.SandboxManager, "cleanup", lambda wd, expected=None: stopped.append(expected))
    monkeypatch.setattr(BrowserManager, "_sessions", {tid: session})
    queue = ThreadAffinityQueue(wait_timeout_s=0.03, hold_timeout_s=1)
    monkeypatch.setattr(threads, "THREAD_QUEUE", queue)
    with queue.acquire("other-thread"):
        threads._process_message(tid, None, resume=True, _run=successor,
                                 accumulated_active_ms=0, pending_text="first")
    assert threads._get_status(tid)["stage"] == "error"
    if same_work:
        assert BrowserManager.current_session(tid) is None
        assert session.closed and stopped == [container]
        assert runs.get(tid, successor.id).status == "timeout"
    else:
        assert BrowserManager.current_session(tid) is session
        assert session.paused and not session.closed and stopped == []


def test_old_terminal_ticket_preserves_valid_same_work_successor(wired, monkeypatch):
    tid, _calls = wired
    runs = threads._runs()
    source = runs.create(tid, "general-agent", "first", user_origin=True)
    runs.claim(tid, source.id)
    runs.transition(tid, source.id, "interrupted")
    old = runs.create(tid, "general-agent", None, resume=True, work_id=source.work_id,
                      user_event_id=source.id)
    runs.claim(tid, old.id)
    runs.transition(tid, old.id, "cancelled")
    successor = runs.create(tid, "general-agent", None, resume=True, work_id=source.work_id,
                            user_event_id=source.id)
    container = SimpleNamespace(id="current-frozen-generation")
    session = BrowserSession(tid, source.id, threads.MANAGER.thread_default_working_dir(tid),
        threads.MANAGER.root_dir, None, work_id=source.work_id,
        sandbox_generation=container.id, sandbox_backend=SimpleNamespace(container=container),
        run_service=runs)
    session.paused = True
    monkeypatch.setattr(BrowserManager, "_sessions", {tid: session})
    threads._execute_run(old.id, tid)
    assert runs.get(tid, successor.id).status == "pending"
    assert BrowserManager.current_session(tid) is session
    assert session.paused and not session.closed


@pytest.mark.parametrize("advanced", ["running", "interrupted", "borrowed_child"])
def test_duplicate_waiter_cannot_stop_work_claimed_after_its_capture(wired, monkeypatch, advanced):
    from assist.browser import authority
    tid, _ = wired
    runs = threads._runs()
    owner = runs.create(tid, "general-agent", "first", user_origin=True)
    runs.claim(tid, owner.id)
    runs.transition(tid, owner.id, "interrupted")
    pending = runs.create(tid, "general-agent", None, resume=True,
                         work_id=owner.work_id, user_event_id=owner.id)
    container = SimpleNamespace(id="retained-generation")
    session = BrowserSession(tid, owner.id, threads.MANAGER.thread_default_working_dir(tid),
        threads.MANAGER.root_dir, None, work_id=owner.work_id,
        sandbox_generation=container.id, sandbox_backend=SimpleNamespace(container=container),
        run_service=runs)
    session.paused = True
    monkeypatch.setattr(BrowserManager, "_sessions", {tid: session})
    monkeypatch.setattr(threads.SandboxManager, "_containers", {session.work_dir: container})
    monkeypatch.setattr("assist.browser.manager._bounded_cli", lambda *_a, **_k: b"")
    with authority.fence(threads.MANAGER.root_dir, tid) as state:
        state.begin(owner.id, owner.admission_sequence)
        state.add_generation(owner.id, container.id)
    stopped = []
    monkeypatch.setattr(threads.SandboxManager, "cleanup", lambda *_a: stopped.append(True))
    successors = []
    @contextmanager
    def wait(*_args, **_kwargs):
        # A different dispatcher claims the same ticket while this duplicate
        # waits, optionally yielding to another same-work physical successor.
        with threads._RUN_ADMISSION_LOCK:
            if advanced != "borrowed_child":
                runs.claim(tid, pending.id)
            else:
                child = runs.create("sub-borrowed", "delegate-agent", "Synthetic child", mode="child",
                    parent_thread_id=tid, parent_run_id=owner.id, dispatch_key="child")
                runs.claim(child.thread_id, child.id)
        if advanced == "borrowed_child":
            assert BrowserManager.resume_session(tid, owner.work_id, None) is session
        with threads._RUN_ADMISSION_LOCK:
            if advanced == "interrupted":
                runs.transition(tid, pending.id, "interrupted")
                successors.append(runs.create(tid, "general-agent", None, resume=True,
                    work_id=owner.work_id, user_event_id=owner.id))
            threads._set_status(tid, "paused", pending_run_id=(
                successors[-1].id if successors else pending.id))
        raise QueueWaitTimeout("synthetic duplicate queue timeout")
        yield
    monkeypatch.setattr(threads, "THREAD_QUEUE", SimpleNamespace(acquire=wait))
    threads._process_message(tid, None, resume=True, _run=pending)
    assert stopped == []
    assert BrowserManager.current_session(tid) is session and not session.closed
    assert threads._get_status(tid)["stage"] == "paused"

    assert runs.get(tid, pending.id).status == ("pending" if advanced == "borrowed_child" else advanced)
    if successors:
        assert runs.get(tid, successors[-1].id).status == "pending"
    if advanced == "borrowed_child":
        child = runs.list("sub-borrowed")[0]
        runs.transition(child.thread_id, child.id, "success")
        session.pause()
        submitted = []
        monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit",
                            lambda rid, thread_id, **_k: submitted.append((rid, thread_id)))
        threads._dispatch_pending_after(tid)
        assert submitted == [(pending.id, tid)]


@pytest.mark.parametrize("dispatch", ["direct", "queued"])
def test_scheduled_work_defers_before_replacing_paused_generation(wired, monkeypatch, dispatch):
    tid, _ = wired
    runs = threads._runs()
    owner = runs.create(tid, "general-agent", "first", user_origin=True)
    runs.claim(tid, owner.id)
    runs.transition(tid, owner.id, "interrupted")
    successor = runs.create(tid, "general-agent", None, resume=True,
                            work_id=owner.work_id, user_event_id=owner.id)
    container = SimpleNamespace(id="filled-page-generation")
    session = BrowserSession(tid, owner.id, threads.MANAGER.thread_default_working_dir(tid),
        threads.MANAGER.root_dir, None, work_id=owner.work_id,
        sandbox_generation=container.id, sandbox_backend=SimpleNamespace(container=container),
        run_service=runs)
    session.paused = True
    monkeypatch.setattr(BrowserManager, "_sessions", {tid: session})
    monkeypatch.setattr(threads, "THREAD_QUEUE", ThreadAffinityQueue())
    monkeypatch.setattr(threads, "_last_visible_user_timezone", lambda _tid: None)
    replacements = []
    def fresh_backend(*_a, **_k):
        replacements.append(container.id)
        return None
    monkeypatch.setattr(threads, "_get_sandbox_backend", fresh_backend)
    threads._set_status(tid, "paused", pending_run_id=successor.id)
    if dispatch == "direct":
        threads._scheduled_dispatch(tid, "Synthetic scheduled work", None)
        scheduled = runs.list(tid)[-1]
    else:
        scheduled = threads._scheduled_run(tid, "Synthetic scheduled work", None)
        threads._execute_run(scheduled.id, tid)
    assert replacements == [], "fresh creation would reap the parent's filled-page generation"
    assert runs.get(tid, scheduled.id).status == "pending"
    assert runs.get(tid, successor.id).status == "pending"
    assert BrowserManager.current_session(tid) is session and not session.closed
    assert threads._get_status(tid)["stage"] == "paused"

    # The existing terminal dispatcher releases the deferred schedule after the
    # parent finishes; deferral must not strand it or require a new scheduler.
    runs.claim(tid, successor.id)
    runs.transition(tid, successor.id, "success")
    BrowserManager.cleanup(tid, session)
    threads._set_status(tid, "ready")
    submitted = []
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit",
                        lambda rid, thread_id, **_k: submitted.append((rid, thread_id)))
    threads._dispatch_pending_after(tid, successor.id)
    assert submitted == [(scheduled.id, tid)]
    monkeypatch.setattr(threads.MANAGER, "get", lambda *_a, **_k: SimpleNamespace(
        agent=None, on_queue_state=None, message=lambda _text: "Synthetic schedule result",
        pending_reply=lambda: None))
    threads._execute_run(scheduled.id, tid)
    assert replacements == [container.id]
    assert runs.get(tid, scheduled.id).status == "success"
