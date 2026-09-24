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
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from manage import web
from manage.web import threads
from manage.web.state import _get_status
from assist.location import LocationSnapshot
from assist.browser.authority import mark_new_thread
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
    mark_new_thread(str(tmp_path), tid)
    calls = []
    chat = _PausingChat(tid, calls)

    monkeypatch.setattr(web.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(web.MANAGER, "thread_dir", lambda t: str(tmp_path / t))
    # Repoint the legacy journal used by migration-only assertions in this fixture.
    from manage.web.state import MESSAGE_BACKLOG
    monkeypatch.setattr(MESSAGE_BACKLOG, "_root", str(tmp_path))
    monkeypatch.setattr(web.MANAGER, "thread_default_working_dir",
                        lambda t: str(tmp_path / t))
    monkeypatch.setattr(web.MANAGER, "touch", lambda t: None)
    monkeypatch.setattr(
        web.MANAGER, "get",
        lambda t, sandbox_backend=None, on_queue_state=None, configurable=None,
        triage=False, continuation=False: chat)
    monkeypatch.setattr("manage.web.threads._get_sandbox_backend",
                        lambda t, tz=None: None)
    monkeypatch.setattr("manage.web.threads._get_domain_manager", lambda t: None)
    monkeypatch.setattr("manage.web.threads.get_cached_description", lambda t: "stub")
    monkeypatch.setattr("manage.web.threads.SandboxManager.cleanup", lambda wd, expected=None: None)
    # Drain the scheduler queue so an item from another test can't leak in.
    with contextlib.suppress(Exception):
        while True:
            threads._RESUME_SCHEDULER._q.get_nowait()
    with threads._RESUME_SCHEDULER._q._cond:
        threads._RESUME_SCHEDULER._q._reserved.clear()
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
    run = threads._runs().get(tid, item["run_id"])
    assert item["tid"] == tid and run.resume is True
    assert run.pending_text == "hello"

    # 2. Run the resume as the scheduler would.
    threads._execute_run(item["run_id"], item["tid"])
    assert _get_status(tid)["stage"] == "ready"

    # Lossless + no re-run: the paused message() ran once, then resume() ran once —
    # the turn was NOT restarted with the original message.
    assert calls == [("message", "hello"), ("resume",)]


def test_pause_publishes_paused_before_an_inline_successor_or_follower(wired, monkeypatch):
    """An eager scheduler cannot let the old pausing slice overwrite its child.

    The deterministic scheduler below is deliberately more adversarial than the
    production worker: it executes the same-work successor inline after commit
    and records the status it can observe.  A follower remains pending until
    that logical work finishes.
    """
    tid, calls = wired
    head = threads._create_run(tid, "hello")
    follower = threads._create_run(tid, "follow-up")
    submitted, status_at_submit, admission_released, successor_at_commit = [], [], [], []
    commit = threads._RESUME_SCHEDULER.commit

    def execute_inline(ticket):
        unlocked = threads._RUN_ADMISSION_LOCK.acquire(blocking=False)
        admission_released.append(unlocked)
        if unlocked:
            threads._RUN_ADMISSION_LOCK.release()
        status_at_submit.append(_get_status(tid)["stage"])
        successor_at_commit.append(next(
            run.id for run in threads._runs().list(tid)
            if run.work_id == head.work_id and run.id != head.id))
        commit(ticket)
        item = threads._RESUME_SCHEDULER._q.get_nowait()
        submitted.append(item["run_id"])
        run = threads._runs().get(tid, item["run_id"])
        if run.work_id == head.work_id:
            threads._execute_run(item["run_id"], tid)

    monkeypatch.setattr(threads._RESUME_SCHEDULER, "commit", execute_inline)

    threads._execute_run(head.id, tid)

    successor = next(run for run in threads._runs().list(tid)
                     if run.work_id == head.work_id and run.id != head.id)
    assert status_at_submit[0] == "paused"
    assert admission_released[0] is True
    assert successor_at_commit == [successor.id]
    assert successor.status == "success"
    assert threads._runs().get(tid, follower.id).status == "pending"
    assert submitted == [successor.id]
    assert threads._RESUME_SCHEDULER._q.get_nowait()["run_id"] == follower.id
    assert calls == [("message", "hello"), ("resume",)]


def test_pause_successor_keeps_the_original_location_snapshot(wired):
    """A later browser fix must not move an already-paused 'from here' turn."""
    tid, _ = wired
    snapshot = LocationSnapshot(37.7749, -122.4194, datetime.now(timezone.utc))
    original = threads._create_run(tid, "walk from here", location=snapshot)

    threads._execute_run(original.id, tid)

    item = threads._RESUME_SCHEDULER._q.get_nowait()
    successor = threads._runs().get(tid, item["run_id"])
    assert threads._location_from_fields(successor.location) == snapshot


def test_new_message_while_paused_routes_through_scheduler(wired, monkeypatch):
    tid, _ = wired
    from manage.web.state import _set_status
    _set_status(tid, "paused", pending_message="hello")

    submitted = []
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit",
                        lambda run_id, t: submitted.append((run_id, t)))
    spawned = []
    monkeypatch.setattr(threads.BackgroundTasks, "add_task",
                        lambda self, fn, *a, **k: spawned.append(fn))

    from fastapi.testclient import TestClient
    TestClient(web.app).post(f"/thread/{tid}/message", data={"text": "follow-up"},
                             follow_redirects=False)

    # The new message is persisted but deliberately not notified yet. The queued
    # resume releases followers only after it becomes terminal, so a required-child
    # resume created later cannot be overtaken by this message.
    (run,) = threads._runs().list(tid)
    assert run.text == "follow-up" and run.status == "pending"
    assert submitted == []
    assert threads._execute_run not in spawned


def test_resume_scheduler_user_priority_and_stable_promotion():
    q = threads._PriorityRunQueue()
    q.put({"run_id": "bg-1", "tid": "other"})
    q.put({"run_id": "user-1", "tid": "user", "user_priority": True})
    q.put({"run_id": "parent-resume", "tid": "parent"})
    q.put({"run_id": "bg-2", "tid": "other"})

    q.promote("parent")

    assert [q.get_nowait()["run_id"] for _ in range(4)] == [
        "user-1", "parent-resume", "bg-1", "bg-2"]


def test_resume_scheduler_reservation_is_invisible_until_committed_once():
    q = threads._PriorityRunQueue()
    ticket = q.reserve({"run_id": "resume", "tid": "thread"})

    assert q.empty()
    with pytest.raises(threads.queue.Empty):
        q.get_nowait()

    q.commit(ticket)
    q.commit(ticket)

    assert q.get_nowait()["run_id"] == "resume"
    with pytest.raises(threads.queue.Empty):
        q.get_nowait()


def test_resume_scheduler_dedupes_only_queued_run_ids():
    q = threads._PriorityRunQueue()
    item = {"kind": "run", "run_id": "same", "tid": "thread"}
    q.put(item)
    q.put(dict(item))
    reserved = q.reserve(dict(item))
    q.commit(reserved)
    assert q.get_nowait()["run_id"] == "same"
    with pytest.raises(threads.queue.Empty):
        q.get_nowait()
    # Once the scheduler has taken the item, a paused turn can legitimately
    # queue its next wake before this invocation has fully unwound.
    q.put(dict(item))
    assert q.get_nowait()["run_id"] == "same"
    with pytest.raises(threads.queue.Empty):
        q.get_nowait()


@pytest.mark.parametrize("via_reservation", [False, True])
def test_duplicate_run_wake_upgrades_existing_background_ticket(via_reservation):
    q = threads._PriorityRunQueue()
    q.put({"kind": "run", "run_id": "sms", "tid": "thread"})
    q.put({"kind": "run", "run_id": "unrelated", "tid": "other"})
    upgraded = {"kind": "run", "run_id": "sms", "tid": "thread",
                "user_priority": True}
    if via_reservation:
        q.commit(q.reserve(upgraded))
    else:
        q.put(upgraded)
    assert [q.get_nowait()["run_id"] for _ in range(2)] == ["sms", "unrelated"]
    with pytest.raises(threads.queue.Empty):
        q.get_nowait()


def test_resume_scheduler_promotes_reservations_on_both_sides_of_commit():
    q = threads._PriorityRunQueue()
    q.put({"run_id": "user-first", "tid": "user", "user_priority": True})
    q.put({"run_id": "background", "tid": "other"})
    before = q.reserve({"run_id": "before", "tid": "parent"})
    q.promote("parent")
    q.commit(before)
    after = q.reserve({"run_id": "after", "tid": "later"})
    q.commit(after)
    q.promote("later")

    assert [q.get_nowait()["run_id"] for _ in range(4)] == [
        "user-first", "before", "after", "background"]


def test_pause_reservation_keeps_a_concurrent_user_promotion(wired, monkeypatch):
    """A user admitted between reserve and commit still promotes the paused resume."""
    tid, calls = wired
    head = threads._create_run(tid, "hello")
    entered = threading.Event()
    release = threading.Event()
    commit = threads._RESUME_SCHEDULER.commit

    def block_commit(ticket):
        entered.set()
        assert release.wait(timeout=2)
        commit(ticket)

    monkeypatch.setattr(threads._RESUME_SCHEDULER, "commit", block_commit)
    worker = threading.Thread(target=threads._execute_run, args=(head.id, tid))
    worker.start()
    assert entered.wait(timeout=2)

    follower, busy = threads._accept_message_run(tid, "follow-up")
    assert busy is True
    release.set()
    worker.join(timeout=2)
    assert not worker.is_alive()

    successor = next(run for run in threads._runs().list(tid)
                     if run.work_id == head.work_id and run.id != head.id)
    queued = threads._RESUME_SCHEDULER._q.get_nowait()
    assert queued["run_id"] == successor.id
    assert queued["user_priority"] is True
    # The user submission remains held until exact browser reconciliation,
    # even though it has already promoted the paused predecessor's ticket.
    assert threads._runs().get(tid, follower.id).status == "revocation_pending"
    assert calls == [("message", "hello")]


def test_committed_reservation_skips_a_cancelled_run(wired):
    tid, calls = wired
    run = threads._create_run(tid, "cancel me")
    ticket = threads._RESUME_SCHEDULER.reserve(run.id, tid)
    threads._runs().transition(tid, run.id, "cancelled")

    threads._RESUME_SCHEDULER.commit(ticket)
    queued = threads._RESUME_SCHEDULER._q.get_nowait()
    threads._execute_run(queued["run_id"], queued["tid"])

    assert threads._runs().get(tid, run.id).status == "cancelled"
    assert calls == []


def test_committed_reservation_never_recreates_a_deleted_thread(wired):
    tid, calls = wired
    run = threads._create_run(tid, "delete me")
    ticket = threads._RESUME_SCHEDULER.reserve(run.id, tid)
    web.MANAGER.hard_delete(tid)

    threads._RESUME_SCHEDULER.commit(ticket)
    queued = threads._RESUME_SCHEDULER._q.get_nowait()
    threads._execute_run(queued["run_id"], queued["tid"])

    assert not (Path(web.MANAGER.root_dir) / tid).exists()
    assert calls == []


def test_busy_user_submission_promotes_both_wait_points(wired, monkeypatch):
    tid, _ = wired
    from manage.web.state import _set_status
    _set_status(tid, "paused", pending_message="hello")
    scheduler_promotions = []
    affinity_promotions = []
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "promote",
                        scheduler_promotions.append)
    monkeypatch.setattr(threads.THREAD_QUEUE, "promote", affinity_promotions.append)

    run, busy = threads._accept_message_run(tid, "change course")

    assert busy is True
    assert run.text == "change course"
    assert scheduler_promotions == [tid]
    assert affinity_promotions == [tid]


def test_pending_dispatch_selects_user_before_older_background(wired, monkeypatch):
    tid, _ = wired
    background = threads._create_run(tid, "child done", origin="task-completion")
    user = threads._create_run(tid, "new direction")
    submitted = []
    monkeypatch.setattr(
        threads._RESUME_SCHEDULER, "submit",
        lambda run_id, thread_id, **kwargs:
            submitted.append((run_id, thread_id, kwargs)))

    threads._dispatch_pending_after(tid)

    assert submitted == [(user.id, tid, {"user_priority": True})]
    assert background.status == "pending"
