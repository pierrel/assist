"""Integration coverage for the fair-scheduling run path through ``_process_message``
and ``_ResumeScheduler`` (Phase 2 of docs/2026-07-08-fair-scheduling.org).

Exercises the pieces the queue-primitive tests (test_fair_scheduling.py) can't:
the ``except ThreadPauseRequested`` branch, the pending-message carry across a pause
(so the user's bubble doesn't vanish), the resume dispatch via the dedicated scheduler
thread, and the round-robin routing of a new message that arrives while paused.

``MANAGER.get`` / sandbox / domain hooks are stubbed (same shape as
test_web_process_message_e2e.py); the pause is raised where the middleware would raise
it (out of the agent run), and the real ``THREAD_QUEUE`` is used un-mocked.
"""
import contextlib

import pytest

from manage import web
from manage.web import threads
from manage.web.state import _get_status
from assist.thread_queue import ThreadPauseRequested


class _PausingChat:
    """message() raises ThreadPauseRequested once (a quantum pause), then resume()
    completes. Records call order so a test can assert the paused turn is NOT re-run."""

    def __init__(self, tid, calls):
        self.thread_id = tid
        self.agent = None
        self.on_queue_state = None
        self._calls = calls

    def message(self, text):
        self._calls.append(("message", text))
        raise ThreadPauseRequested("quantum pause")

    def resume(self):
        self._calls.append(("resume",))
        return "final answer"

    def pending_reply(self):
        return None


@pytest.fixture
def wired(tmp_path, monkeypatch):
    tid = "t-pause"
    (tmp_path / tid).mkdir()
    calls = []
    chat = _PausingChat(tid, calls)

    monkeypatch.setattr(web.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(web.MANAGER, "thread_dir", lambda t: str(tmp_path / t))
    monkeypatch.setattr(web.MANAGER, "thread_default_working_dir",
                        lambda t: str(tmp_path / t))
    monkeypatch.setattr(web.MANAGER, "touch", lambda t: None)
    monkeypatch.setattr(
        web.MANAGER, "get",
        lambda t, sandbox_backend=None, on_queue_state=None, configurable=None,
        triage=False: chat)
    monkeypatch.setattr("manage.web.threads._get_sandbox_backend",
                        lambda t, tz=None: None)
    monkeypatch.setattr("manage.web.threads._get_domain_manager", lambda t: None)
    monkeypatch.setattr("manage.web.threads.get_cached_description", lambda t: "stub")
    monkeypatch.setattr("manage.web.threads.SandboxManager.cleanup", lambda wd: None)
    # Drain the scheduler queue so an item from another test can't leak in.
    with contextlib.suppress(Exception):
        while True:
            threads._RESUME_SCHEDULER._q.get_nowait()
    return tid, calls


def test_pause_carries_pending_and_submits_resume(wired):
    tid, calls = wired

    # 1. The initial turn pauses (message() raises ThreadPauseRequested).
    threads._process_message(tid, "hello")

    st = _get_status(tid)
    assert st["stage"] == "paused"
    # fix #1: the user's message bubble survives the pause (not dropped).
    assert st.get("pending_message") == "hello"

    # A resume was queued on the dedicated scheduler (NOT a BackgroundTask), carrying
    # the pending text so the resume keeps the bubble too.
    item = threads._RESUME_SCHEDULER._q.get_nowait()
    assert item["tid"] == tid and item["resume"] is True
    assert item["pending"] == "hello"

    # 2. Run the resume as the scheduler would.
    threads._process_message(item["tid"], None, resume=True,
                             accumulated_active_ms=item["acc"],
                             pending_text=item["pending"])
    assert _get_status(tid)["stage"] == "ready"

    # Lossless + no re-run: the paused message() ran once, then resume() ran once —
    # the turn was NOT restarted with the original message.
    assert calls == [("message", "hello"), ("resume",)]


def test_new_message_while_paused_routes_through_scheduler(wired, monkeypatch):
    tid, _ = wired
    from manage.web.state import _set_status
    _set_status(tid, "paused", pending_message="hello")

    submitted = []
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit_message",
                        lambda t, text, rider, sender: submitted.append((t, text)))
    spawned = []
    monkeypatch.setattr(threads.BackgroundTasks, "add_task",
                        lambda self, fn, *a, **k: spawned.append(fn))

    from fastapi.testclient import TestClient
    TestClient(web.app).post(f"/thread/{tid}/message", data={"text": "follow-up"},
                             follow_redirects=False)

    # The new message was routed to the serial scheduler (runs AFTER the resume),
    # NOT spawned as a competing BackgroundTask onto a mid-flight checkpoint.
    assert submitted == [(tid, "follow-up")]
    assert threads._process_message not in spawned
