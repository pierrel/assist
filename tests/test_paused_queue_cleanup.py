from types import SimpleNamespace
import pytest

from assist.browser.manager import BrowserManager, BrowserSession
from assist.thread_queue import ThreadAffinityQueue
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
