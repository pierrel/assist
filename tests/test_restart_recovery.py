"""Restart recovery — durable follow-up journal + resume-from-checkpoint.

Pins the contracts of docs/2026-07-13-durable-message-queue.org (Step 1): a
message accepted while its thread is busy is journaled durably BEFORE the POST
returns; the turn claims (removes) its entry status-first when it starts; and
startup recovery delivers every journaled message exactly once and resumes /
re-dispatches / finalizes the interrupted head turn by GRAPH STATE, not text
equality. The kill-shaped test exercises a real langgraph SqliteSaver abandoned
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
    monkeypatch.setattr("manage.web.threads.SandboxManager.cleanup", lambda wd: None)
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
            triage=False):
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

    # Durable on disk before the redirect returned, and the dispatch carries the
    # entry id so the turn claims it when it starts.
    on_disk = json.loads((tmp_path / tid / "pending_messages.json").read_text())
    assert [r["text"] for r in on_disk] == ["follow-up"]
    assert spawned and spawned[0][2].get("backlog_id") == on_disk[0]["id"]
    # The running head turn's status was NOT clobbered by the follow-up.
    assert _get_status(tid).get("pending_message") == "head turn"


def test_idle_thread_first_message_is_not_journaled(wired, monkeypatch):
    tid, tmp_path = wired
    _set_status(tid, "ready")
    monkeypatch.setattr(threads.BackgroundTasks, "add_task",
                        lambda self, fn, *a, **k: None)
    from fastapi.testclient import TestClient
    TestClient(web.app).post(f"/thread/{tid}/message", data={"text": "first"},
                             follow_redirects=False)
    # The first message is durable via status.pending_message — the journal is
    # follow-ups only.
    assert not (tmp_path / tid / "pending_messages.json").exists()
    assert _get_status(tid).get("pending_message") == "first"


def test_turn_claims_its_entry_status_first(wired, monkeypatch):
    """When a journaled follow-up's turn starts: status takes ownership (text +
    claimed_id) and THEN the journal entry is popped — the message is durable in
    at least one store at every instant."""
    tid, _ = wired
    calls = []
    _wire_chat(monkeypatch, tid, calls)
    rec = MESSAGE_BACKLOG.add(PendingMessage(thread_id=tid, text="follow-up"))

    statuses = []
    real_set = threads._set_status

    def spy_set(t, stage, **kw):
        statuses.append((stage, kw.get("claimed_id"),
                         bool(MESSAGE_BACKLOG.for_thread(tid))))
        real_set(t, stage, **kw)
    monkeypatch.setattr(threads, "_set_status", spy_set)

    threads._process_message(tid, "follow-up", backlog_id=rec.id)

    # The starting_sandbox write carried the claim id while the entry was STILL
    # journaled; the pop happened after.
    claim_writes = [s for s in statuses if s[0] == "starting_sandbox"]
    assert claim_writes == [("starting_sandbox", rec.id, True)]
    assert MESSAGE_BACKLOG.for_thread(tid) == []       # claimed after
    assert calls == [("message", "follow-up")]


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
    # Turn completed before the kill -> finalize (containment: supersede prefix).
    snap_holder["snap"] = _snap(messages=[HumanMessage("PREFIX hello"), AIMessage("hi")])
    assert threads._recovery_decision(tid, "hello") == "finalize"
    # Message never reached the checkpoint -> redispatch. The duplicate-text trap:
    # an OLDER "ok" in history must NOT read as this turn's — but state decides
    # first: with no pending superstep and no match on the LATEST human, redispatch.
    snap_holder["snap"] = _snap(messages=[HumanMessage("ok"), AIMessage("done"),
                                          HumanMessage("different")])
    assert threads._recovery_decision(tid, "ok") == "redispatch"
    # Unreadable state -> error (degrade loudly, don't guess).
    monkeypatch.setattr(web.MANAGER, "get",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert threads._recovery_decision(tid, "hello") == "error"


# --- _recover_thread: exactly-once end-to-end ---------------------------------

def test_recover_redispatches_head_and_drains_backlog_in_order(wired, monkeypatch):
    tid, _ = wired
    calls = []
    _wire_chat(monkeypatch, tid, calls)
    monkeypatch.setattr(threads, "_recovery_decision", lambda t, p: "redispatch")
    _set_status(tid, "paused", pending_message="head")
    MESSAGE_BACKLOG.add(PendingMessage(thread_id=tid, text="f1"))
    MESSAGE_BACKLOG.add(PendingMessage(thread_id=tid, text="f2"))

    threads._recover_thread(tid)

    assert calls == [("message", "head"), ("message", "f1"), ("message", "f2")]
    assert MESSAGE_BACKLOG.for_thread(tid) == []   # each entry claimed exactly once


def test_recover_resumes_and_preserves_triage_sender(wired, monkeypatch):
    """A triage (inbound-SMS) turn killed mid-processing resumes AS a triage turn:
    the sender persisted in the busy status write drives triage=True + the reply
    target — without it the recovery would rebuild a full-privilege agent."""
    tid, _ = wired
    calls, triage_seen = [], []
    _wire_chat(monkeypatch, tid, calls, triage_seen)
    monkeypatch.setattr(threads, "_recovery_decision", lambda t, p: "resume")
    _set_status(tid, "paused", pending_message="sms text", sender="+15550001111",
                accumulated_active_ms=1234.0)

    threads._recover_thread(tid)

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

    threads._recover_thread(tid)

    assert calls == [("resume",)]                    # one delivery path
    assert MESSAGE_BACKLOG.for_thread(tid) == []     # deduped, not re-dispatched


def test_recover_finalizes_completed_turn(wired, monkeypatch):
    tid, _ = wired
    calls = []
    _wire_chat(monkeypatch, tid, calls)
    monkeypatch.setattr(threads, "_recovery_decision", lambda t, p: "finalize")
    _set_status(tid, "paused", pending_message="already answered")

    threads._recover_thread(tid)

    assert calls == []                               # nothing re-run
    assert _get_status(tid)["stage"] == "ready"


def test_recover_unrecoverable_errors_with_message_surfaced(wired, monkeypatch):
    tid, _ = wired
    calls = []
    _wire_chat(monkeypatch, tid, calls)
    monkeypatch.setattr(threads, "_recovery_decision", lambda t, p: "error")
    _set_status(tid, "paused", pending_message="lost?")

    threads._recover_thread(tid)

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
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit_recover",
                        lambda t: recovered.append(t))

    # The lifespan's recovery block, run directly (the full lifespan needs the app).
    from manage.web import state as st_mod
    for t in web.MANAGER.list():
        status = _get_status(t)
        if status.get("stage") in st_mod.BUSY_STAGES:
            _set_status(t, "paused",
                        **{k: v for k, v in status.items() if k != "stage"})
            threads.submit_recovery(t)

    st = _get_status(tid)
    assert st["stage"] == "paused"
    assert st.get("pending_message") == "mid-flight"    # carried through
    assert st.get("sender") == "+15550001111"           # triage info survives
    assert recovered == [tid]


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
    # left for inspection, not overwritten
    assert (tmp_path / "t1" / "pending_messages.json").read_text() == '[{"truncated'


def test_claim_is_idempotent(tmp_path):
    store = MessageBacklog(str(tmp_path))
    (tmp_path / "t1").mkdir()
    rec = store.add(PendingMessage(thread_id="t1", text="x"))
    store.claim("t1", rec.id)
    store.claim("t1", rec.id)          # double-claim: no raise
    assert store.for_thread("t1") == []
