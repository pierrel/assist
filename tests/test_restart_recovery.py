"""Restart recovery — legacy journal import + run/checkpoint recovery.

Pins the contracts of docs/2026-07-13-durable-message-queue.org (Step 1): a
Historical messages accepted while a thread was busy were journaled before the POST
returns; the turn claims (removes) its entry status-first when it starts; and
startup recovery delivers every journaled message exactly once and handles the
interrupted head turn — resume-vs-rest decided by GRAPH STATE, with "finalize"
requiring an EXACT match of the checkpointed message (as sent, or with the
supersede prefix). The kill-shaped test exercises a real langgraph SqliteSaver abandoned
mid-run and reopened in a fresh "process" — the crash shape, not the cooperative
pause shape.
"""
import contextlib
import json
import os
import sqlite3
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from manage import web
from manage.web import phone_api, threads
from manage.web.run_stream import RunStreamJournal
from manage.web.state import MESSAGE_BACKLOG, _get_status, _set_status
from assist.backlog import MessageBacklog, PendingMessage
from assist.thread_engine import write_new_thread_engine


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """A thread dir + repointed MANAGER/backlog, with the agent machinery stubbed
    (same shape as test_fair_scheduling_integration's fixture)."""
    tid = "t-recover"
    (tmp_path / tid).mkdir()

    monkeypatch.setattr(web.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(web.MANAGER, "thread_dir", lambda t: str(tmp_path / t))
    monkeypatch.setattr(MESSAGE_BACKLOG, "_root", str(tmp_path))
    monkeypatch.setattr(web.MANAGER, "thread_default_working_dir",
                        lambda t: str(tmp_path / t))
    monkeypatch.setattr(web.MANAGER, "touch", lambda t: None)
    monkeypatch.setattr("manage.web.threads._get_sandbox_backend",
                        lambda t, tz=None: None)
    monkeypatch.setattr("manage.web.threads._get_domain_manager", lambda t: None)
    monkeypatch.setattr("manage.web.threads.get_cached_description", lambda t: "stub")
    monkeypatch.setattr("manage.web.threads.SandboxManager.cleanup", lambda wd, expected=None: None)
    with contextlib.suppress(Exception):
        while True:
            threads._RESUME_SCHEDULER._q.get_nowait()
    with contextlib.suppress(Exception):
        while True:
            threads._INITIALIZATION_SCHEDULER._q.get_nowait()
    return tid, tmp_path


class _Chat:
    """Minimal Thread stand-in that records how it was run."""

    def __init__(self, tid, calls):
        self.thread_id = tid
        self._calls = calls

    def message(self, text):
        self._calls.append(("message", text))
        return "done"

    def resume(self):
        self._calls.append(("resume",))
        return "resumed"

    def pending_reply(self):
        return None

    def get_messages(self):
        return []


def _wire_chat(monkeypatch, tid, calls, triage_seen=None):
    def get(t, sandbox_backend=None, on_queue_state=None, configurable=None,
            triage=False, continuation=False):
        if triage_seen is not None:
            triage_seen.append((triage, (configurable or {})))
        return _Chat(t, calls)
    monkeypatch.setattr(web.MANAGER, "get", get)


# --- the journal: durable before dispatch, claimed at turn start ---------------

def test_follow_up_to_busy_thread_is_journaled_durably(wired, monkeypatch):
    tid, tmp_path = wired
    _set_status(tid, "processing", pending_message="head turn")
    spawned = []
    monkeypatch.setattr(threads.BackgroundTasks, "add_task",
                        lambda self, fn, *a, **k: spawned.append((fn, a, k)))

    from fastapi.testclient import TestClient
    TestClient(web.app).post(f"/thread/{tid}/message", data={"text": "follow-up"},
                             follow_redirects=False)

    # Durable before redirect; the background notification carries only its id.
    (run,) = threads._runs().list(tid)
    assert run.text == "follow-up" and run.status == "pending"
    assert spawned == []  # the running head's terminal handoff releases this run
    # The running head turn's status was NOT clobbered by the follow-up.
    assert _get_status(tid).get("pending_message") == "head turn"


def test_idle_thread_first_message_is_a_run_not_legacy_journal(wired, monkeypatch):
    tid, tmp_path = wired
    _set_status(tid, "ready")
    monkeypatch.setattr(threads.BackgroundTasks, "add_task",
                        lambda self, fn, *a, **k: None)
    from fastapi.testclient import TestClient
    TestClient(web.app).post(f"/thread/{tid}/message", data={"text": "first"},
                             follow_redirects=False)
    # The Run is acceptance truth; status is only the UI projection.
    assert not (tmp_path / tid / "pending_messages.json").exists()
    assert [run.text for run in threads._runs().list(tid)] == ["first"]
    assert _get_status(tid).get("pending_message") == "first"


# --- the recovery decision: graph state, not text -----------------------------

def _snap(next_=(), interrupts=(), messages=()):
    return SimpleNamespace(next=next_, interrupts=interrupts,
                           values={"messages": list(messages)})


def test_recovery_decision_matrix(wired, monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage
    tid, _ = wired
    snap_holder = {}

    class _StateChat:
        thread_id = tid
        agent = property(lambda self: SimpleNamespace(
            get_state=lambda cfg: snap_holder["snap"]))
        runconfig = {}
    monkeypatch.setattr(web.MANAGER, "get",
                        lambda t, sandbox_backend=None, **k: _StateChat())

    # Mid-flight (pending superstep) -> resume.
    snap_holder["snap"] = _snap(next_=("agent",))
    assert threads._recovery_decision(tid, "hello") == "resume"
    # Durable HITL interrupt -> resume (it re-fires).
    snap_holder["snap"] = _snap(interrupts=(object(),))
    assert threads._recovery_decision(tid, "hello") == "resume"
    # Turn completed before the kill -> finalize. EXACT match only — as sent, or
    # as the supersede fold checkpoints it (_SUPERSEDE_RIDER prefix)...
    snap_holder["snap"] = _snap(messages=[
        HumanMessage(threads._SUPERSEDE_RIDER + "hello"), AIMessage("hi")])
    assert threads._recovery_decision(tid, "hello") == "finalize"
    # ...but a pending that merely appears inside/at-the-end of the PREVIOUS
    # turn's message (crash pre-input-checkpoint) must NOT finalize — that would
    # silently drop the message.
    snap_holder["snap"] = _snap(messages=[HumanMessage("ok, book it"), AIMessage("hi")])
    assert threads._recovery_decision(tid, "ok") == "redispatch"
    snap_holder["snap"] = _snap(messages=[HumanMessage("can you book it"), AIMessage("hi")])
    assert threads._recovery_decision(tid, "book it") == "redispatch"
    # Message never reached the checkpoint at all -> redispatch.
    snap_holder["snap"] = _snap(messages=[HumanMessage("ok"), AIMessage("done"),
                                          HumanMessage("different")])
    assert threads._recovery_decision(tid, "ok") == "redispatch"
    # Unreadable state -> error (degrade loudly, don't guess).
    monkeypatch.setattr(web.MANAGER, "get",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert threads._recovery_decision(tid, "hello") == "error"


# --- _recover_thread: exactly-once end-to-end ---------------------------------

def _drain_worker_queue(calls_expected_tid):
    """Run any turn jobs the recovery submitted to the serial worker, the way
    _ResumeScheduler._loop would (synchronously, in order)."""
    while True:
        try:
            it = threads._RESUME_SCHEDULER._q.get_nowait()
        except Exception:
            return
        assert it["tid"] == calls_expected_tid
        threads._execute_run(it["run_id"], it["tid"])


def test_recover_redispatches_head_and_drains_backlog_in_order(wired, monkeypatch):
    tid, _ = wired
    calls = []
    _wire_chat(monkeypatch, tid, calls)
    monkeypatch.setattr(threads, "_recovery_decision", lambda t, p: "redispatch")
    _set_status(tid, "paused", pending_message="head")
    MESSAGE_BACKLOG.add(PendingMessage(thread_id=tid, text="f1"))
    MESSAGE_BACKLOG.add(PendingMessage(thread_id=tid, text="f2"))

    threads.queue_recovery_runs()
    # The head ran inline; the follow-ups were SUBMITTED to the serial worker
    # (behind any resume the head might have re-queued) — run them as the worker.
    _drain_worker_queue(tid)

    assert calls == [("message", "head"), ("message", "f1"), ("message", "f2")]
    assert MESSAGE_BACKLOG.for_thread(tid) == []   # each entry claimed exactly once


def test_recovery_dispatches_committed_pending_run_without_status_duplicate(
        wired, monkeypatch):
    tid, _ = wired
    run = threads._create_run(tid, "accepted")
    _set_status(tid, "processing", pending_message="accepted",
                pending_run_id=run.id)
    decisions = []
    monkeypatch.setattr(threads, "_recovery_decision",
                        lambda *args: decisions.append(args) or "redispatch")

    threads.queue_recovery_runs()

    assert decisions == []
    assert [(item.id, item.text) for item in threads._runs().list(tid)] == [
        (run.id, "accepted")]
    queued = threads._RESUME_SCHEDULER._q.get_nowait()
    assert queued["run_id"] == run.id


def test_recovery_replays_persisted_first_run_initialization(wired):
    """A crash before the first clone must not execute that Run without its worktree."""
    tid, _ = wired
    run = threads._create_run(tid, "first message")
    _set_status(tid, "initializing", pending_message="first message",
                pending_run_id=run.id, domain="repo://example")

    threads.queue_recovery_runs()

    assert threads._INITIALIZATION_SCHEDULER._q.get_nowait() == (
        run.id, tid, "repo://example", None)


def test_recovery_replays_first_run_when_status_lacks_its_id(wired):
    """The acceptance/status write boundary cannot make the first Run ordinary work."""
    tid, _ = wired
    run = threads._create_run(tid, "first message")
    _set_status(tid, "initializing", pending_message="first message", domain="repo://example")

    threads.queue_recovery_runs()

    assert threads._INITIALIZATION_SCHEDULER._q.get_nowait() == (
        run.id, tid, "repo://example", None)


def test_recovery_replays_partial_clone_before_running_first_turn(wired):
    tid, _ = wired
    run = threads._create_run(tid, "first message")
    _set_status(tid, "cloning", pending_message="first message",
                pending_run_id=run.id, domain="repo://example")

    threads.queue_recovery_runs()

    assert threads._INITIALIZATION_SCHEDULER._q.get_nowait() == (
        run.id, tid, "repo://example", None)


def test_recovery_discards_unaccepted_first_thread_reservation(wired):
    """A status-only initial thread never became accepted work, so retry may recreate it."""
    tid, tmp_path = wired
    _set_status(tid, "initializing", pending_message="first message")

    threads.queue_recovery_runs()

    assert not (tmp_path / tid).exists()


def test_initialization_failure_terminalizes_pending_followers(wired):
    tid, _ = wired
    first = threads._create_run(tid, "first", work_id="first-work")
    follower = threads._create_run(tid, "later", work_id="follower-work")
    journal = RunStreamJournal()
    for work_id in (first.work_id, follower.work_id):
        assert journal.reserve(tid, work_id)
        assert journal.activate(tid, work_id)
    # `wired` gives this test a real durable run store; the journal is the
    # phone-only process-local observer that must be retired with both runs.
    with patch.object(threads, "RUN_STREAMS", journal):
        threads._fail_initialization(tid, first.id, "first")

    assert [run.status for run in threads._runs().list(tid)] == ["error", "error"]
    assert journal.read(tid, first.work_id)["terminal"]
    assert journal.read(tid, follower.work_id)["terminal"]


def test_cancelled_initializer_finishes_its_owned_setup_before_releasing_a_follower(
        wired, monkeypatch):
    """DELETE is immediate, but a clone owner alone releases its ready projection."""
    tid, tmp_path = wired
    head = threads._create_run(tid, "first", work_id="first-work")
    follower = threads._create_run(tid, "later", work_id="later-work")
    _set_status(tid, "initializing", pending_message="first", pending_run_id=head.id,
                domain="repo://example")
    journal = RunStreamJournal()
    assert journal.reserve(tid, head.work_id) and journal.activate(tid, head.work_id)
    started, release = threading.Event(), threading.Event()
    executed, dispatched = [], []

    class BlockingDomain:
        def __init__(self, *_args, **_kwargs):
            started.set()
            assert release.wait(1)

    monkeypatch.setattr(phone_api, "RUN_STREAMS", journal)
    monkeypatch.setattr(threads, "RUN_STREAMS", journal)
    monkeypatch.setattr(threads, "DomainManager", BlockingDomain)
    monkeypatch.setattr(threads, "_execute_run", lambda *args: executed.append(args))
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit",
                        lambda run_id, thread_id, **_kwargs: dispatched.append((run_id, thread_id)))
    monkeypatch.setattr(threads._INITIALIZATION_SCHEDULER, "complete", lambda *_args: None)

    worker = threading.Thread(
        target=threads._initialize_thread, args=(tid, head.id, "repo://example"))
    worker.start()
    assert started.wait(1)

    code, value = phone_api._cancel_logical_run(tid, head.id)

    assert code == 200
    assert value["run"]["status"] == "cancelled"
    assert _get_status(tid)["stage"] == "cloning"
    assert threads._runs().get(tid, head.id).cancel_cleanup == "complete"
    assert journal.read(tid, head.work_id)["terminal"]
    assert not executed and not dispatched
    assert (tmp_path / tid).is_dir()

    release.set()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert not executed
    assert _get_status(tid)["stage"] == "ready"
    assert dispatched == [(follower.id, tid)]


def test_cancelled_no_domain_initializer_never_executes_the_head(wired, monkeypatch):
    tid, _ = wired
    head = threads._create_run(tid, "first")
    _set_status(tid, "initializing", pending_message="first", pending_run_id=head.id)
    journal = RunStreamJournal()
    assert journal.reserve(tid, head.work_id) and journal.activate(tid, head.work_id)
    executed = []

    monkeypatch.setattr(phone_api, "RUN_STREAMS", journal)
    monkeypatch.setattr(threads, "RUN_STREAMS", journal)
    monkeypatch.setattr(threads, "_execute_run", lambda *args: executed.append(args))
    monkeypatch.setattr(threads._INITIALIZATION_SCHEDULER, "complete", lambda *_args: None)

    assert phone_api._cancel_logical_run(tid, head.id)[0] == 200
    threads._initialize_thread(tid, head.id, None)

    assert not executed
    assert _get_status(tid)["stage"] == "ready"


def test_cancelled_initializer_clone_failure_does_not_execute_it(wired, monkeypatch):
    tid, _ = wired
    head = threads._create_run(tid, "first")
    _set_status(tid, "initializing", pending_message="first", pending_run_id=head.id,
                domain="repo://example")
    executed = []

    monkeypatch.setattr(threads, "DomainManager",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("clone failed")))
    monkeypatch.setattr(threads, "_execute_run", lambda *args: executed.append(args))
    monkeypatch.setattr(threads._INITIALIZATION_SCHEDULER, "complete", lambda *_args: None)

    assert phone_api._cancel_logical_run(tid, head.id)[0] == 200
    threads._initialize_thread(tid, head.id, "repo://example")

    assert not executed
    assert threads._runs().get(tid, head.id).status == "cancelled"
    assert _get_status(tid)["stage"] == "error"


def test_restart_requeues_the_exact_cancelled_initializer_without_a_new_run(wired, monkeypatch):
    tid, _ = wired
    head = threads._create_run(tid, "first")
    _set_status(tid, "initializing", pending_message="first", pending_run_id=head.id,
                domain="repo://example")
    threads._runs().cancel_logical(tid, head.id)
    threads._runs().complete_cancel_cleanup(tid, head.id)
    submitted = []

    monkeypatch.setattr(threads.MANAGER, "list", lambda: [tid])
    monkeypatch.setattr(threads._INITIALIZATION_SCHEDULER, "submit",
                        lambda *job: submitted.append(job))
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no resume")))

    threads.queue_recovery_runs()

    assert submitted == [(head.id, tid, "repo://example")]
    assert [run.id for run in threads._runs().list(tid)] == [head.id]


def test_initialization_failure_retry_repairs_only_the_status_projection(wired, monkeypatch):
    tid, _ = wired
    run = threads._create_run(tid, "first", work_id="first-work")
    _set_status(tid, "initializing", pending_message="first", pending_run_id=run.id)
    scheduler = threads._InitializationScheduler()
    retries, completions = [], []
    writes = {"status": 0}
    original_set_status = threads._set_status

    def fail_once(thread_id, stage, **kwargs):
        if stage == "error" and writes["status"] == 0:
            writes["status"] += 1
            raise OSError("status unavailable")
        original_set_status(thread_id, stage, **kwargs)

    monkeypatch.setattr(threads, "_INITIALIZATION_SCHEDULER", scheduler)
    monkeypatch.setattr(scheduler, "retry", lambda *job: retries.append(job))
    monkeypatch.setattr(scheduler, "complete", lambda *job: completions.append(job))
    monkeypatch.setattr(threads, "_set_status", fail_once)
    monkeypatch.setattr(threads, "_execute_run",
                        lambda *_args: (_ for _ in ()).throw(RuntimeError("setup failed")))

    threads._initialize_thread(tid, run.id, None)
    assert threads._runs().get(tid, run.id).status == "error"
    assert retries == [(run.id, tid, None, None)]

    threads._initialize_thread(tid, run.id, None)

    assert _get_status(tid)["stage"] == "error"
    assert completions == [(run.id, tid)]


def test_initialization_failure_requeues_the_same_run_until_one_final_dispatch(
        wired, monkeypatch):
    tid, _ = wired
    run = threads._create_run(tid, "first", work_id="first-work")
    scheduler = threads._InitializationScheduler()
    retries, completions, executions = [], [], []
    writes = 0
    service = threads._runs()
    write = service._write

    def fail_once(thread_id, runs):
        nonlocal writes
        writes += 1
        if writes == 1:
            raise OSError("durable store unavailable")
        write(thread_id, runs)

    def execute(run_id, thread_id):
        executions.append((run_id, thread_id))
        service.claim(thread_id, run_id)
        service.transition(thread_id, run_id, "success")

    monkeypatch.setattr(service, "_write", fail_once)
    monkeypatch.setattr(threads, "_INITIALIZATION_SCHEDULER", scheduler)
    monkeypatch.setattr(scheduler, "retry", lambda *job: retries.append(job))
    monkeypatch.setattr(scheduler, "complete", lambda *job: completions.append(job))
    monkeypatch.setattr(threads, "_execute_run",
                        lambda *_args: (_ for _ in ()).throw(RuntimeError("clone failed")))

    threads._initialize_thread(tid, run.id, None)

    assert retries == [(run.id, tid, None, None)]
    assert [(item.id, item.work_id, item.status) for item in service.list(tid)] == [
        (run.id, "first-work", "pending")]

    monkeypatch.setattr(threads, "_execute_run", execute)
    threads._initialize_thread(retries[0][1], retries[0][0], retries[0][2])

    assert executions == [(run.id, tid)]
    assert [(item.id, item.work_id, item.status) for item in service.list(tid)] == [
        (run.id, "first-work", "success")]
    assert completions == [(run.id, tid)]


def test_persistent_initialization_failure_stays_owned_and_requeued(wired, monkeypatch):
    tid, _ = wired
    run = threads._create_run(tid, "first", work_id="first-work")
    scheduler = threads._InitializationScheduler()
    retries = []
    service = threads._runs()

    monkeypatch.setattr(threads, "_INITIALIZATION_SCHEDULER", scheduler)
    monkeypatch.setattr(scheduler, "retry", lambda *job: retries.append(job))
    monkeypatch.setattr(service, "_write",
                        lambda *_args: (_ for _ in ()).throw(OSError("store unavailable")))
    monkeypatch.setattr(threads, "_execute_run",
                        lambda *_args: (_ for _ in ()).throw(RuntimeError("clone failed")))

    threads._initialize_thread(tid, run.id, None)
    threads._initialize_thread(tid, run.id, None)

    assert retries == [(run.id, tid, None, None), (run.id, tid, None, None)]
    assert [(item.id, item.work_id, item.status) for item in service.list(tid)] == [
        (run.id, "first-work", "pending")]


def test_initialization_retry_is_deduplicated_and_caps_its_backoff(monkeypatch):
    scheduler = threads._InitializationScheduler()
    delays = []

    class Timer:
        daemon = False

        def __init__(self, delay, _callback, args):
            delays.append((delay, args))

        def start(self):
            return None

    monkeypatch.setattr(threads.threading, "Timer", Timer)

    scheduler.retry("run-a", "thread-a", None, None)
    scheduler.retry("run-a", "thread-a", None, None)
    with scheduler._lock:
        scheduler._scheduled.clear()
        scheduler._retries[("run-a", "thread-a")] = 5
    scheduler.retry("run-a", "thread-a", None, None)

    assert [delay for delay, _ in delays] == [1, 30]


def test_execute_run_preserves_awaiting_approval_and_retires_phone_journal(wired, monkeypatch):
    """The durable Run must match the approval state a phone observer sees."""
    tid, _ = wired
    run = threads._create_run(tid, "need approval", work_id="approval-work")
    journal = RunStreamJournal()
    assert journal.reserve(tid, run.work_id)
    assert journal.activate(tid, run.work_id)

    def await_approval(_tid, _text, *, _run, **_kwargs):
        threads._runs().claim(_tid, _run.id)
        _set_status(_tid, "awaiting_approval")

    monkeypatch.setattr(threads, "RUN_STREAMS", journal)
    monkeypatch.setattr(threads, "_process_message", await_approval)
    monkeypatch.setattr(threads, "_dispatch_pending_after", lambda *_args: None)

    threads._execute_run(run.id, tid)

    assert threads._runs().get(tid, run.id).status == "awaiting_approval"
    assert journal.read(tid, run.work_id)["terminal"]


def test_thread_deletion_closes_phone_journals(wired, monkeypatch):
    tid, _ = wired
    run = threads._create_run(tid, "delete me", work_id="deleted-work")
    journal = RunStreamJournal()
    assert journal.reserve(tid, run.work_id)
    assert journal.activate(tid, run.work_id)
    monkeypatch.setattr(threads, "RUN_STREAMS", journal)
    monkeypatch.setattr(threads._PI_RUNTIME, "retire", lambda _tid: None)
    monkeypatch.setattr(threads, "_evict_caches", lambda _tid: None)
    monkeypatch.setattr(threads, "_evict_egress", lambda _tid: None)

    threads._delete_thread_and_children(tid)

    assert journal.is_gone(tid, run.work_id)


def test_invalid_engine_execution_retires_phone_journal(wired, monkeypatch):
    """A durable engine error has the same bounded observer cleanup as a turn error."""
    from assist.thread_engine import ThreadEngineError

    tid, _ = wired
    run = threads._create_run(tid, "broken engine", work_id="broken-engine-work")
    journal = RunStreamJournal()
    assert journal.reserve(tid, run.work_id)
    assert journal.activate(tid, run.work_id)
    monkeypatch.setattr(threads, "RUN_STREAMS", journal)
    monkeypatch.setattr(
        threads, "_is_pi_thread",
        lambda _tid: (_ for _ in ()).throw(ThreadEngineError("invalid marker")),
    )

    threads._execute_run(run.id, tid)

    assert threads._runs().get(tid, run.id).status == "error"
    assert journal.read(tid, run.work_id)["terminal"]


def test_recovering_pi_head_never_builds_a_deep_graph(wired, monkeypatch):
    tid, tmp_path = wired
    write_new_thread_engine(tmp_path / tid, "pi")
    head = threads._create_run(tid, "abandoned Pi turn")
    head = threads._runs().claim(tid, head.id)
    follower = threads._create_run(tid, "later manual turn")
    monkeypatch.setattr(
        web.MANAGER, "get",
        lambda *args, **kwargs: pytest.fail("Pi recovery must not construct a Deep graph"))

    threads._recover_run(head)

    recovered = threads._runs().get(tid, head.id)
    assert recovered.status == "error"
    queued = threads._RESUME_SCHEDULER._q.get_nowait()
    assert queued["run_id"] == follower.id


def test_journal_only_recovery_skips_head_and_stays_healthy(wired, monkeypatch):
    """A ready thread with surviving follow-ups (crash after the head turn
    finished): recovery must NOT touch the head — no resume, no spurious
    'could not be recovered' error — just submit the journaled follow-ups."""
    tid, _ = wired
    calls = []
    _wire_chat(monkeypatch, tid, calls)
    decisions = []
    monkeypatch.setattr(threads, "_recovery_decision",
                        lambda t, p: decisions.append(t) or "resume")
    _set_status(tid, "ready")
    MESSAGE_BACKLOG.add(PendingMessage(thread_id=tid, text="leftover"))

    threads.queue_recovery_runs()

    assert decisions == []                          # head decision never ran
    assert _get_status(tid)["stage"] == "ready"     # no error banner
    _drain_worker_queue(tid)
    assert calls == [("message", "leftover")]


def test_recover_resumes_and_preserves_triage_sender(wired, monkeypatch):
    """Legacy busy-status recovery preserves triage sender and reply target.

    New accepted work takes the same values from its durable Run; this test pins the
    pre-Run crash fallback imported during migration.
    """
    tid, _ = wired
    calls, triage_seen = [], []
    _wire_chat(monkeypatch, tid, calls, triage_seen)
    monkeypatch.setattr(threads, "_recovery_decision", lambda t, p: "resume")
    _set_status(tid, "paused", pending_message="sms text", sender="+15550001111",
                accumulated_active_ms=1234.0)

    threads.queue_recovery_runs()
    _drain_worker_queue(tid)

    assert calls == [("resume",)]
    assert triage_seen and triage_seen[0][0] is True          # triage preserved
    assert triage_seen[0][1].get(threads.SMS_SENDER_KEY) == "+15550001111"


def test_recover_dedupes_claimed_entry(wired, monkeypatch):
    """Crash between the status-claim write and the journal pop: the entry is in
    BOTH stores. Recovery removes the claimed entry (status owns that message) and
    delivers it via exactly one path — the head resume."""
    tid, _ = wired
    calls = []
    _wire_chat(monkeypatch, tid, calls)
    monkeypatch.setattr(threads, "_recovery_decision", lambda t, p: "resume")
    rec = MESSAGE_BACKLOG.add(PendingMessage(thread_id=tid, text="claimed msg"))
    _set_status(tid, "paused", pending_message="claimed msg", claimed_id=rec.id)

    threads.queue_recovery_runs()
    _drain_worker_queue(tid)

    assert calls == [("resume",)]                    # one delivery path
    assert MESSAGE_BACKLOG.for_thread(tid) == []     # deduped, not re-dispatched


def test_recover_finalizes_completed_turn(wired, monkeypatch):
    tid, _ = wired
    calls = []
    _wire_chat(monkeypatch, tid, calls)
    monkeypatch.setattr(threads, "_recovery_decision", lambda t, p: "finalize")
    _set_status(tid, "paused", pending_message="already answered")

    threads.queue_recovery_runs()

    assert calls == []                               # nothing re-run
    assert _get_status(tid)["stage"] == "ready"


def test_recover_unrecoverable_errors_with_message_surfaced(wired, monkeypatch):
    tid, _ = wired
    calls = []
    _wire_chat(monkeypatch, tid, calls)
    monkeypatch.setattr(threads, "_recovery_decision", lambda t, p: "error")
    _set_status(tid, "paused", pending_message="lost?")

    threads.queue_recovery_runs()

    st = _get_status(tid)
    assert st["stage"] == "error" and st.get("pending_message") == "lost?"
    assert calls == []


# --- lifespan wiring: rewrite-to-paused + enqueue -----------------------------

def test_lifespan_rewrites_busy_to_paused_and_enqueues_recovery(wired, monkeypatch):
    """The stage-rewrite is load-bearing: post_message routes live messages through
    the serial worker only for stage=='paused', so a recovered thread must not sit
    at 'processing' where a new message could race the recovery resume."""
    tid, _ = wired
    _set_status(tid, "processing", pending_message="mid-flight",
                sender="+15550001111")
    recovered = []
    monkeypatch.setattr(threads, "_recovery_decision", lambda t, p: "resume")
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit",
                        lambda run_id, t: recovered.append((run_id, t)))

    # The REAL scan the lifespan calls (extracted so this test can't drift from it).
    from manage.web.state import _recover_interrupted_threads
    _recover_interrupted_threads()

    st = _get_status(tid)
    assert st["stage"] == "paused"
    assert st.get("pending_message") == "mid-flight"    # carried through
    assert st.get("sender") == "+15550001111"           # triage info survives
    assert len(recovered) == 1 and recovered[0][1] == tid


def test_lifespan_preserves_initializing_for_first_run_recovery(wired, monkeypatch):
    tid, _ = wired
    run = threads._create_run(tid, "first")
    _set_status(tid, "initializing", pending_message="first",
                pending_run_id=run.id, domain="repo://example")
    recovered = []
    monkeypatch.setattr(threads._INITIALIZATION_SCHEDULER, "submit",
                        lambda run_id, thread_id, domain: recovered.append(
                            (run_id, thread_id, domain)))

    from manage.web.state import _recover_interrupted_threads
    _recover_interrupted_threads()

    assert _get_status(tid)["stage"] == "initializing"
    assert recovered == [(run.id, tid, "repo://example")]


# --- kill-shaped resume: a real graph, a real SqliteSaver, a hard abandon -----

def test_kill_shaped_resume_from_reopened_checkpointer(tmp_path):
    """The crash shape un-mocked: run a real 2-node langgraph under a SqliteSaver
    with durability='sync', ABANDON it mid-run (node B raises — but unlike a
    cooperative pause we then throw the whole graph object away), reopen the
    saver as a fresh 'process', and resume input=None. Node A's completed work
    must be preserved (not re-run) and the turn must finish."""
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.sqlite import SqliteSaver
    from typing_extensions import TypedDict

    db = str(tmp_path / "ckpt.db")
    runs = {"a": 0, "b": 0}

    class S(TypedDict):
        log: list

    def node_a(state):
        runs["a"] += 1
        return {"log": state["log"] + ["a"]}

    crash = {"on": True}

    def node_b(state):
        runs["b"] += 1
        if crash["on"]:
            raise RuntimeError("hard kill mid-superstep")
        return {"log": state["log"] + ["b"]}

    def build(conn):
        g = StateGraph(S)
        g.add_node("a", node_a)
        g.add_node("b", node_b)
        g.add_edge(START, "a")
        g.add_edge("a", "b")
        g.add_edge("b", END)
        return g.compile(checkpointer=SqliteSaver(conn))

    cfg = {"configurable": {"thread_id": "t1"}}
    conn1 = sqlite3.connect(db, check_same_thread=False)
    graph1 = build(conn1)
    with pytest.raises(RuntimeError):
        graph1.invoke({"log": []}, cfg, durability="sync")
    conn1.close()          # the "process" dies; nothing cooperative ran

    # Fresh process: reopen the saver, resume from the durable checkpoint.
    crash["on"] = False
    conn2 = sqlite3.connect(db, check_same_thread=False)
    graph2 = build(conn2)
    snap = graph2.get_state(cfg)
    assert snap.next, "mid-flight turn must show a pending superstep"
    result = graph2.invoke(None, cfg, durability="sync")
    conn2.close()

    assert result["log"] == ["a", "b"]
    assert runs["a"] == 1, "completed superstep must NOT re-run on resume"
    assert runs["b"] == 2   # the interrupted superstep re-runs (the named residual)


# --- the journal store itself -------------------------------------------------

def test_corrupt_backlog_is_loud_not_silent(tmp_path, caplog):
    store = MessageBacklog(str(tmp_path))
    (tmp_path / "t1").mkdir()
    (tmp_path / "t1" / "pending_messages.json").write_text('[{"truncated')
    import logging as _logging
    with caplog.at_level(_logging.ERROR):
        assert store.for_thread("t1") == []
    assert any("unreadable" in r.message for r in caplog.records)
    # Moved aside for inspection — a later add()'s read-modify-write would
    # otherwise replace the corrupt file and destroy the evidence.
    assert (tmp_path / "t1" / "pending_messages.json.corrupt").read_text() \
        == '[{"truncated'
    assert not (tmp_path / "t1" / "pending_messages.json").exists()
    # And a follow-up add works cleanly after the move-aside.
    store.add(PendingMessage(thread_id="t1", text="after"))
    assert [r.text for r in store.for_thread("t1")] == ["after"]


def test_sms_follow_up_to_paused_thread_routes_through_worker(wired, monkeypatch):
    """An inbound-SMS follow-up to a PAUSED thread must go to the serial worker
    (behind the queued resume), never dispatch directly — a paused thread has
    released the slot, so a direct dispatch would acquire immediately and run on
    the mid-flight checkpoint (Copilot #197 rd1)."""
    tid, _ = wired
    _set_status(tid, "paused", pending_message="mid-flight")
    sub = SimpleNamespace(thread_id=tid, render=lambda s, t: f"[{s}] {t}")
    monkeypatch.setattr(threads.SUBSCRIPTION_STORE, "route", lambda s: sub)
    ran, submitted = [], []
    monkeypatch.setattr(threads, "_execute_run",
                        lambda *a, **k: ran.append(a))
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit",
                        lambda run_id, t: submitted.append((run_id, t)))

    threads._dispatch_event("+15550001111", "hey")

    assert ran == []                                  # no direct dispatch
    (run,) = threads._runs().list(tid)
    assert run.sender == "+15550001111" and run.text == "[+15550001111] hey"
    assert submitted == [(run.id, tid)]


def test_review_to_paused_thread_waits_for_resume_handoff(wired, monkeypatch):
    """A paused thread persists reviews without outrunning its queued resume."""
    import json as _json
    from fastapi.testclient import TestClient
    tid, _ = wired
    _set_status(tid, "paused", pending_message="mid-flight")
    monkeypatch.setattr("manage.web.review._get_domain_manager", lambda t: None)
    ran, submitted = [], []
    monkeypatch.setattr(threads.BackgroundTasks, "add_task",
                        lambda self, fn, *a, **k: ran.append(fn))
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit",
                        lambda run_id, t: submitted.append((run_id, t)))

    r = TestClient(web.app).post(
        f"/thread/{tid}/review",
        data={"payload": _json.dumps({"overall": "LGTM", "lines": []})},
        follow_redirects=False)

    assert r.status_code == 303
    assert threads._execute_run not in ran        # no BackgroundTask dispatch
    (run,) = threads._runs().list(tid)
    assert submitted == []


def test_claim_is_idempotent(tmp_path):
    store = MessageBacklog(str(tmp_path))
    (tmp_path / "t1").mkdir()
    rec = store.add(PendingMessage(thread_id="t1", text="x"))
    store.claim("t1", rec.id)
    store.claim("t1", rec.id)          # double-claim: no raise
    assert store.for_thread("t1") == []


def test_event_loop_stays_live_while_store_lock_is_held(wired, monkeypatch):
    """The un-mocked contended-lock test (repo rule: exercise the risk, don't mock
    it): hold the journal store's lock in another thread while a busy-thread POST
    is in flight. The POST's append must run OFF the event loop (private
    CapacityLimiter), so the loop stays live — a concurrent GET completes promptly
    while the POST is still blocked on the lock.

    The client MUST be context-managed: only ``__enter__`` pins ONE portal (one
    event loop) shared by both requests — the no-ctx form spins a fresh loop per
    request, which would pass even if the append regressed to running inline on
    the loop (a vacuous test). The lifespan that entering runs is neutralized
    (recovery scan / scheduler / manager close are process-level, not under test)."""
    import threading as _threading
    import time as _time
    from fastapi.testclient import TestClient
    from manage.web import state as st_mod

    tid, _ = wired
    _set_status(tid, "processing", pending_message="head")
    monkeypatch.setattr(threads.BackgroundTasks, "add_task",
                        lambda self, fn, *a, **k: None)
    monkeypatch.setattr(st_mod, "_recover_interrupted_threads", lambda: None)
    monkeypatch.setattr(threads, "start_scheduler", lambda: None)
    monkeypatch.setattr(threads, "stop_scheduler", lambda: None)
    monkeypatch.setattr(web.MANAGER, "close", lambda: None)

    with TestClient(web.app) as client:   # ctx-managed: ONE portal/loop for both
        threads._runs()._lock.acquire()
        post_done = _threading.Event()
        try:
            t = _threading.Thread(
                target=lambda: (client.post(f"/thread/{tid}/message",
                                            data={"text": "follow-up"},
                                            follow_redirects=False),
                                post_done.set()),
                daemon=True)
            t.start()
            _time.sleep(0.3)
            assert not post_done.is_set(), \
                "POST should be blocked on the held store lock"
            start = _time.monotonic()
            status = client.get(f"/thread/{tid}/status")
            elapsed = _time.monotonic() - start
            assert status.status_code == 200
            assert elapsed < 2.0, \
                f"loop blocked: GET took {elapsed:.1f}s under a held store lock"
        finally:
            threads._runs()._lock.release()
        assert post_done.wait(5.0), "POST must complete once the lock is released"
