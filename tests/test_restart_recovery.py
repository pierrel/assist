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
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from manage import web
from manage.web import threads
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
