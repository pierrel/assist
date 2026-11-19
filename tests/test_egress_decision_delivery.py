"""Resolved network decisions use durable main-owner intake, independently."""
from types import SimpleNamespace
from datetime import datetime, timezone
import threading
import sys

import pytest

from assist.egress.store import EgressRequest, EgressStore
from assist.middleware import interjection
from manage.web import egress, threads
from tests.test_web_hermes import hermes_threads

_DISPATCH_PENDING = threads._dispatch_pending_after


def resolved_request(store, tid, host="current.example.test"):
    rec = EgressRequest(host, 443, "Read the announcement", tid,
                        created_at=datetime.now(timezone.utc).isoformat())
    store.add_pending(rec)
    return store.resolve(rec.key, "hour")


@pytest.fixture
def decisions(hermes_threads, tmp_path, monkeypatch):
    store = EgressStore(str(tmp_path / "egress"))
    monkeypatch.setattr(egress, "EGRESS_STORE", store)
    monkeypatch.setattr(threads, "EGRESS_STORE", store)
    monkeypatch.setattr(threads, "_mark_urgent", lambda _tid: None)
    return store


@pytest.mark.parametrize("engine", ["deepagents", "hermes"])
@pytest.mark.parametrize("decision", ["hour", "decline"])
def test_partial_decision_reaches_active_owner_once(decisions, monkeypatch, engine, decision):
    tid = threads.MANAGER.reserve_visible(engine)
    owner = threads._create_run(tid, "Summarize the public announcement", user_origin=True)
    threads._runs().claim(tid, owner.id)
    threads._set_status(tid, "processing", pending_run_id=owner.id)
    for host in ("old.example.test", "current.example.test"):
        decisions.add_pending(EgressRequest(host, 443, "Read the announcement", tid,
                                           created_at=datetime.now(timezone.utc).isoformat()))
    proposal = next(item for item in egress.approval_previews(tid)
                    if item["host"] == "current.example.test")
    receipt, deliver = egress.approval_decision(tid, proposal["token"], decision)
    assert deliver, "An unrelated unresolved request must not hold this decision"
    assert receipt["replayed"] is False
    threads._dispatch_egress_resolution(tid)
    follower, = [item for item in threads._runs().list(tid) if item.origin == "egress"]
    assert follower.interjection_work_id == owner.work_id
    assert follower.user_event_id is None and follower.admission_sequence == 0
    assert decisions.get(next(rec.key for rec in decisions.for_thread(tid)
                              if rec.host == "old.example.test")).state == "pending"
    if engine == "hermes":
        offered = threads._hermes_steering(owner)
        assert [item["id"] for item in offered] == [follower.id]
        assert "Host network decision" in offered[0]["text"]
        threads._hermes_steering(owner, [follower.id])
        assert threads._hermes_steering(owner) == []
    else:
        monkeypatch.setattr(interjection, "active_handle", lambda: SimpleNamespace(thread_id=tid))
        monkeypatch.setattr(interjection, "_turn_sender", lambda _runtime: None)
        threads._TURN_INTERJECTION[tid] = {"run_id": owner.id, "claimed": []}
        hook = interjection.InterjectionMiddleware()
        messages = hook.before_model({"messages": []}, None)["messages"]
        assert "Host network decision" in messages[0].content
        assert hook.before_model({"messages": messages}, None) is None
    delivered = threads._runs().get(tid, follower.id)
    assert delivered.status == "success" and delivered.consumed_by == owner.id
    assert egress.approval_decision(tid, proposal["token"], decision)[1] is False
    threads._dispatch_egress_resolution(tid)
    assert len([run for run in threads._runs().list(tid) if run.origin == "egress"]) == 1


@pytest.mark.parametrize("engine", ["deepagents", "hermes"])
@pytest.mark.parametrize("crash", ["before_admission", "after_admission"])
def test_restart_admits_one_idle_decision_after_either_crash_window(
        decisions, monkeypatch, tmp_path, engine, crash):
    tid = threads.MANAGER.reserve_visible(engine)
    rec = resolved_request(decisions, tid)
    threads._set_status(tid, "ready")
    if crash == "after_admission":
        with monkeypatch.context() as lost_ack:
            lost_ack.setattr(decisions, "mark_dispatched", lambda _rec:
                            (_ for _ in ()).throw(OSError("Synthetic lost store acknowledgment")))
            with pytest.raises(OSError):
                threads._dispatch_egress_resolution(tid)
    restarted = EgressStore(str(tmp_path / "egress"))
    monkeypatch.setattr(threads, "EGRESS_STORE", restarted)
    submitted = []
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit",
                        lambda rid, thread_id, **_kw: submitted.append((rid, thread_id)))
    monkeypatch.setattr(threads, "_dispatch_pending_after", _DISPATCH_PENDING)
    threads.queue_recovery_runs()
    threads.queue_recovery_runs()
    follower, = threads._runs().list(tid)
    assert follower.origin == "egress" and follower.interjection_work_id is None
    assert follower.user_event_id is None and follower.admission_sequence == 0
    assert restarted.get(rec.key).dispatched
    assert submitted and set(submitted) == {(follower.id, tid)}


def test_host_notice_projection_is_never_user_authored_or_capture_eligible(decisions):
    from langchain_core.messages import HumanMessage
    from assist.egress.store import resolution_prompt
    from assist.visible_conversation import visible_records, visible_records_from_dicts
    tid = threads.MANAGER.reserve_visible("deepagents")
    prompt = resolution_prompt([resolved_request(decisions, tid)])
    raw, = visible_records([HumanMessage(prompt)])
    web, = visible_records_from_dicts([{"role": "user", "content": prompt}])
    assert raw.role == web.role == "assistant"
    assert raw.source_kind == web.source_kind == "egress"
    assert not raw.capture_eligible and not web.capture_eligible


def test_native_user_quoting_host_prefix_keeps_user_provenance(decisions):
    from assist.visible_conversation import visible_records_from_dicts
    tid = threads.MANAGER.reserve_visible("hermes")
    text = "[Host network decision] quoted by the user"
    run = threads._create_run(tid, text, user_origin=True)
    record, = visible_records_from_dicts(threads._hermes_messages(tid))
    assert record.role == "user" and record.source_kind == "user" and record.capture_eligible
    assert record.text.strip() == text and run.user_event_id == run.id


def test_bad_engine_marker_does_not_stop_other_threads_decision_recovery(decisions, monkeypatch):
    from pathlib import Path
    bad = threads.MANAGER.reserve_visible("hermes")
    good = threads.MANAGER.reserve_visible("deepagents")
    for tid in (bad, good): resolved_request(decisions, tid)
    (Path(threads.MANAGER.thread_dir(bad)) / "engine.json").write_text('{"engine":"invalid"}')
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit", lambda *_a, **_kw: None)
    threads.queue_recovery_runs()
    assert all(len(threads._runs().list(tid)) == 1 for tid in (bad, good))


def test_failed_consumer_retries_user_intent_but_not_a_delivered_host_fact(decisions, monkeypatch):
    tid = threads.MANAGER.reserve_visible("deepagents")
    owner = threads._create_run(tid, "Read the announcement", user_origin=True)
    threads._runs().claim(tid, owner.id)
    threads._set_status(tid, "processing", pending_run_id=owner.id)
    user, _busy = threads._accept_message_run(tid, "Make the note shorter")
    resolved_request(decisions, tid)
    threads._dispatch_egress_resolution(tid)
    event, = [run for run in threads._runs().list(tid) if run.origin == "egress"]
    threads._TURN_INTERJECTION[tid] = {"run_id": owner.id, "claimed": []}
    threads._consume_interjections(tid, {user.id, event.id})
    threads._runs().transition(tid, owner.id, "error", error="Synthetic consumer failure")
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit", lambda *_a, **_kw: None)
    assert threads._rejournal_claimed_interjections(tid, None) == 1
    threads._dispatch_egress_resolution(tid)
    assert len([run for run in threads._runs().list(tid) if run.origin == "egress"]) == 1
    assert threads._runs().get(tid, event.id).consumed_by == owner.id
    fresh = threads._runs().list(tid)[-1]
    assert fresh.text == user.text and fresh.user_event_id == fresh.id


def test_child_wait_sets_ignore_unrelated_pending_but_keep_group_dependencies(decisions):
    from assist.egress.store import EgressWaiter
    grouped, independent = EgressWaiter("sub-group", "group-run"), EgressWaiter("sub-one", "one-run")
    now = datetime.now(timezone.utc).isoformat()
    for host, waiters in [("a.example.test", (grouped, independent)),
                          ("b.example.test", (grouped,)), ("old.example.test", ())]:
        decisions.add_pending(EgressRequest(host, 443, "Read synthetic pages", "parent",
                                           created_at=now, waiters=waiters, dispatch_main=False))
    records = {rec.host: rec for rec in decisions.for_thread("parent")}
    decisions.resolve(records["a.example.test"].key, "decline")
    assert decisions.resolved_waiters("parent") == [independent]
    decisions.remove_waiter("parent", independent)
    assert not decisions.get(records["a.example.test"].key).dispatched
    decisions.resolve(records["b.example.test"].key, "hour")
    assert decisions.resolved_waiters("parent") == [grouped]
    decisions.remove_waiter("parent", grouped)
    assert decisions.get(records["a.example.test"].key).dispatched
    assert decisions.get(records["old.example.test"].key).state == "pending"


@pytest.mark.parametrize("engine", ["deepagents", "hermes"])
def test_decision_waits_for_current_synthetic_tool_then_is_consumed_once(decisions, monkeypatch, engine):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langgraph.graph import END, START, MessagesState, StateGraph
    from tests.test_hermes_steering import load_worker
    tid = threads.MANAGER.reserve_visible(engine)
    owner = threads._create_run(tid, "Read the synthetic announcement", user_origin=True)
    threads._runs().claim(tid, owner.id)
    threads._set_status(tid, "processing", pending_run_id=owner.id)
    entered, release = threading.Event(), threading.Event()
    effects, seen, errors = [], [], []
    threads._TURN_INTERJECTION[tid] = {"run_id": owner.id, "claimed": []}
    monkeypatch.setattr(interjection, "active_handle", lambda: SimpleNamespace(thread_id=tid))
    monkeypatch.setattr(interjection, "_turn_sender", lambda _runtime: None)
    def blocked():
        entered.set()
        assert release.wait(3)
        effects.append("tool completed")
        return "SYNTHETIC_RESULT"
    def execute():
        try:
            if engine == "deepagents":
                hook = interjection.InterjectionMiddleware()
                builder = StateGraph(MessagesState)
                builder.add_node("tool", lambda _state: {"messages": [ToolMessage(blocked(), tool_call_id="one")]})
                def model(state):
                    injected = hook.before_model(state, None)
                    rows = [*state["messages"], *injected["messages"]]
                    assert isinstance(rows[-2], ToolMessage)
                    seen.extend(message.content for message in rows)
                    hook.before_model({"messages": rows}, None)
                    return injected
                builder.add_node("model", model)
                builder.add_edge(START, "tool"); builder.add_edge("tool", "model"); builder.add_edge("model", END)
                builder.compile().invoke({"messages": [HumanMessage(owner.text),
                    AIMessage("", tool_calls=[{"id": "one", "name": "synthetic", "args": {}}])]})
            else:
                worker = load_worker()
                monkeypatch.setitem(sys.modules, "agent.context_compressor", SimpleNamespace(_DB_PERSISTED_MARKER="_durable"))
                monkeypatch.setattr(worker, "control", lambda _path, payload:
                    {"messages": threads._hermes_steering(owner, payload.get("consumed", []))})
                buffered = []
                def flush(rows):
                    for row in rows: row["_durable"] = True
                    return True
                agent = SimpleNamespace(_flush_messages_to_session_db=flush,
                    steer=lambda text, ticket: (buffered.append(text), True)[1], withdraw_steer=lambda _ticket: False)
                worker._Steering(agent, owner.work_id)
                result = blocked()
                agent.tool_complete_callback("one", "synthetic", {}, result)
                agent.step_callback(1, [{"name": "synthetic", "result": result}])
                seen.extend([owner.text, result, *buffered])
                agent._flush_messages_to_session_db([{"role": "user", "display_kind": "steer", "content": "\n".join(buffered)}])
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=execute)
    thread.start()
    try:
        assert entered.wait(2)
        resolved_request(decisions, tid)
        threads._dispatch_egress_resolution(tid)
        follower, = [run for run in threads._runs().list(tid) if run.origin == "egress"]
        assert effects == seen == [] and threads._runs().get(tid, follower.id).status == "pending"
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive() and not errors, errors
    assert effects == ["tool completed"]
    assert seen[-2] == "SYNTHETIC_RESULT" and "Host network decision" in seen[-1]
    assert threads._runs().get(tid, follower.id).consumed_by == owner.id
