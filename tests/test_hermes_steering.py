"""Native follow-up identities, durable receipts and queued fallback."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import threading
from types import SimpleNamespace

import pytest

from assist.hermes_provider_relay import HermesProviderRelay, HermesRelayError
from assist.hermes_steering import committed_receipts
from assist.hermes_runtime import HermesRuntimeManager
from assist.thread_queue import ThreadAffinityQueue
from manage.web import threads
from tests.test_web_hermes import hermes_threads

_DISPATCH_PENDING = threads._dispatch_pending_after


def owner_and_followers(hermes_threads, count=2):
    tid = threads.MANAGER.reserve_visible("hermes")
    threads._create_empty_pi_workspace(tid)
    owner = threads._create_run(tid, "Create a blue note", user_origin=True)
    threads._runs().claim(tid, owner.id)
    threads._set_status(tid, "processing", pending_run_id=owner.id)
    followers = [threads._accept_message_run(tid, "Make the note green")[0]
                 for _ in range(count)]
    return tid, owner, followers


def test_same_owner_followups_keep_identity_until_persisted(hermes_threads):
    tid, owner, followers = owner_and_followers(hermes_threads)
    assert all(run.interjection_work_id == owner.work_id for run in followers)
    offered = threads._hermes_steering(owner)
    assert [item["id"] for item in offered] == [run.id for run in followers]
    assert offered[0]["text"] != offered[1]["text"]
    assert all(threads._runs().get(tid, run.id).status == "pending" for run in followers)
    assert [item["id"] for item in threads._hermes_steering(owner, [followers[0].id])] == [followers[1].id]
    assert threads._runs().get(tid, followers[0].id).consumed_by == owner.id
    threads._hermes_steering(owner, [followers[0].id])
    assert threads._runs().get(tid, followers[1].id).status == "pending"


def test_native_history_keeps_each_owners_consumed_followups_before_its_reply(hermes_threads):
    tid, owner, followers = owner_and_followers(hermes_threads, count=2)
    threads._hermes_steering(owner, [run.id for run in followers])
    threads._runs().transition(tid, owner.id, "success", result="First reply")
    later = threads._create_run(tid, "Next ordinary turn", user_origin=True)
    threads._runs().claim(tid, later.id)
    threads._runs().transition(tid, later.id, "success", result="Second reply")
    messages = threads._hermes_messages(tid)
    assert [(row["role"], row["content"]) for row in messages] == [
        ("user", "Create a blue note"), ("user", "Make the note green"),
        ("user", "Make the note green"), ("assistant", "First reply"),
        ("user", "Next ordinary turn"), ("assistant", "Second reply")]
    assert len({row["message_id"] for row in messages}) == len(messages)


@pytest.mark.parametrize("boundary", ["cancel", "other_work", "sender", "scheduled"])
def test_ineligible_messages_cannot_enter_native_owner(hermes_threads, boundary):
    tid, owner, followers = owner_and_followers(hermes_threads, count=1)
    follower = followers[0]
    if boundary == "cancel":
        threads._runs().transition(tid, owner.id, "cancelled")
        with pytest.raises(RuntimeError, match="terminal or cancelled"):
            threads._hermes_steering(owner)
        return
    if boundary == "other_work":
        threads._runs().transition(tid, owner.id, "success")
        other = threads._create_run(tid, "Independent work", origin="system")
        threads._runs().claim(tid, other.id)
        assert threads._hermes_steering(other) == []
    else:
        extra = threads._create_run(tid, "Unrelated message",
            sender="+15550001234" if boundary == "sender" else None,
            origin="system" if boundary == "scheduled" else None)
        assert extra.id not in {item["id"] for item in threads._hermes_steering(owner)}
    assert threads._runs().get(tid, follower.id).status == "pending"


def receipt_database(tid, owner, ids, *, active=1):
    directory = Path(threads.MANAGER.thread_agent_dir(tid)) / "hermes"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "session.db"
    with sqlite3.connect(path) as database:
        database.execute("CREATE TABLE messages (role TEXT, display_kind TEXT, display_metadata TEXT, active INTEGER)")
        database.execute("INSERT INTO messages VALUES (?, ?, ?, ?)", ("user", "steer", json.dumps({
            "assist_steering": {"work_id": owner.work_id, "run_ids": ids}}), active))
    return path


@pytest.mark.parametrize("active", [0, 1])
def test_commit_before_host_ack_is_reconciled_without_replay(hermes_threads, monkeypatch, active):
    tid, owner, followers = owner_and_followers(hermes_threads)
    receipt_database(tid, owner, [followers[0].id], active=active)
    threads._runs().transition(tid, owner.id, "error", error="Synthetic process loss")
    monkeypatch.setattr(threads._HERMES_RUNTIME, "run_slice", lambda *_a, **_k:
                        pytest.fail("Committed native follow-up must not replay"))
    threads._execute_hermes_run(followers[0], user_priority=False)
    delivered = threads._runs().get(tid, followers[0].id)
    assert delivered.status == "success" and delivered.consumed_by == owner.id
    assert threads._runs().get(tid, followers[1].id).status == "pending"
    threads._execute_hermes_run(delivered, user_priority=False)


def test_missing_native_state_keeps_queued_fallback(hermes_threads):
    tid, _owner, followers = owner_and_followers(hermes_threads, count=1)
    threads._reconcile_hermes_steering(tid)
    assert threads._runs().get(tid, followers[0].id).status == "pending"


def test_unconsumed_followup_executes_once_through_real_run_dispatch(hermes_threads, monkeypatch):
    hermes_threads.set_enabled(True)
    tid, owner, followers = owner_and_followers(hermes_threads, count=1)
    threads._runs().transition(tid, owner.id, "success", result="Original work finished")
    calls = []
    runtime = HermesRuntimeManager()
    def native(run, **_kwargs):
        calls.append(run.id)
        return "Queued follow-up completed"
    monkeypatch.setattr(runtime, "run_slice", native)
    monkeypatch.setattr(threads, "_HERMES_RUNTIME", runtime)
    monkeypatch.setattr(threads, "THREAD_QUEUE", ThreadAffinityQueue())
    threads._execute_run(followers[0].id, tid)
    threads._execute_run(followers[0].id, tid)
    result = threads._runs().get(tid, followers[0].id)
    assert calls == [followers[0].id]
    assert result.status == "success" and result.consumed_by is None
    assert result.result == "Queued follow-up completed"


def test_no_active_owner_creates_an_ordinary_queued_turn(hermes_threads):
    tid = threads.MANAGER.reserve_visible("hermes")
    threads._create_empty_pi_workspace(tid)
    run, _busy = threads._accept_message_run(tid, "Create a note")
    assert run.interjection_work_id is None and run.status == "pending"


def test_unreadable_native_receipt_is_visible_without_replay(hermes_threads, monkeypatch):
    tid, owner, followers = owner_and_followers(hermes_threads, count=1)
    path = receipt_database(tid, owner, [followers[0].id])
    path.write_bytes(b"Synthetic corrupted native state")
    threads._runs().transition(tid, owner.id, "error", error="Synthetic process loss")
    monkeypatch.setattr(threads._HERMES_RUNTIME, "run_slice", lambda *_a, **_k:
                        pytest.fail("Ambiguous delivery must not replay"))
    threads._execute_hermes_run(followers[0], user_priority=False)
    failed = threads._runs().get(tid, followers[0].id)
    assert failed.status == "error" and "could not be verified" in failed.error
    assert threads._get_status(tid)["stage"] == "error"


def test_unreadable_receipts_dispatch_each_remaining_follower_without_replay(hermes_threads, monkeypatch):
    tid, owner, followers = owner_and_followers(hermes_threads, count=2)
    path = receipt_database(tid, owner, [followers[0].id])
    path.write_bytes(b"Synthetic corrupted native state")
    threads._runs().transition(tid, owner.id, "error", error="Synthetic process loss")
    submitted = []
    monkeypatch.setattr(threads, "_dispatch_pending_after", _DISPATCH_PENDING)
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit",
        lambda rid, thread_id, **_kwargs: submitted.append((rid, thread_id)))
    monkeypatch.setattr(threads._HERMES_RUNTIME, "run_slice", lambda *_a, **_k:
                        pytest.fail("Ambiguous delivery must not replay"))
    threads._execute_hermes_run(followers[0], user_priority=False)
    assert submitted == [(followers[1].id, tid)]
    threads._execute_hermes_run(followers[1], user_priority=False)
    assert all(threads._runs().get(tid, run.id).status == "error" for run in followers)
    assert submitted == [(followers[1].id, tid)]


@pytest.mark.parametrize("ending", ["unadmitted", "result", "stopped"])
def test_late_or_parked_intake_cannot_claim_journal(tmp_path, ending):
    relay = HermesProviderRelay(tmp_path, "http://127.0.0.1:9/v1", "dummy", "dummy", 8192,
        lambda: ending != "unadmitted", steering=lambda _ids: pytest.fail("Closed intake"))
    if ending == "result":
        relay.result = {"reply": "Finished", "failed": False}
    elif ending == "stopped":
        relay.stop_admission()
    try:
        with pytest.raises(HermesRelayError):
            relay.steer({"consumed": ["admission-a"]})
    finally:
        relay.close()


def load_worker():
    path = Path(__file__).parents[1] / "assist/hermes_runtime/worker.py"
    spec = importlib.util.spec_from_file_location("steering_worker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_steer_row(monkeypatch, agent, tickets):
    """Model the pinned native drain; its real formatter is covered in-image."""
    parts = [(item["id"], item["text"].strip()) for item in tickets]
    text = "\n".join(value for _ticket, value in parts)
    agent._steer_last_drain = (text, parts)
    row = lambda value: {"role": "user", "display_kind": "steer",
                         "content": "Native steer:\n" + value}
    monkeypatch.setitem(sys.modules, "agent.prompt_builder", SimpleNamespace(steer_user_row=row))
    return row(text)


@pytest.mark.parametrize("embedded", ["quoted", "framed"])
def test_receipt_uses_only_exact_native_drain_ticket(monkeypatch, embedded):
    worker = load_worker()
    monkeypatch.setitem(sys.modules, "agent.context_compressor",
                        SimpleNamespace(_DB_PERSISTED_MARKER="_durable"))
    other = "[Assist follow-up id=other]\nSame text"
    body = ('Quote: "' + other + '"') if embedded == "quoted" else other
    tickets = [{"id": "actual", "text": "[Assist follow-up id=actual]\n" + body},
               {"id": "other", "text": other}]
    receipts = []
    def control(_path, payload):
        receipts.extend(payload.get("consumed", []))
        return {"messages": tickets}
    monkeypatch.setattr(worker, "control", control)
    def flush(rows):
        for row in rows:
            row["_durable"] = True
        return True
    agent = SimpleNamespace(_flush_messages_to_session_db=flush,
                            steer=lambda *_a, **_kw: True, withdraw_steer=lambda _ticket: False)
    delivery = worker._Steering(agent, "logical-work")
    delivery.tool_complete()
    delivery.step()
    row = native_steer_row(monkeypatch, agent, tickets[:1])
    agent._flush_messages_to_session_db([row])
    assert row["display_metadata"]["assist_steering"]["run_ids"] == ["actual"]
    assert receipts == ["actual"]
    assert set(delivery.offered) == {"other"}


def test_finish_restores_original_hooks_even_when_withdraw_fails():
    worker = load_worker()
    def flush(_rows):
        return True
    original_step, original_tool = object(), object()
    def withdraw(_ticket):
        raise RuntimeError("Synthetic withdrawal failure")
    agent = SimpleNamespace(_flush_messages_to_session_db=flush,
                            step_callback=original_step, tool_complete_callback=original_tool,
                            withdraw_steer=withdraw)
    delivery = worker._Steering(agent, "logical-work")
    delivery.offered["pending"] = "pending text"
    with pytest.raises(RuntimeError, match="Synthetic withdrawal failure"):
        delivery.finish()
    assert agent._flush_messages_to_session_db is flush
    assert agent.step_callback is original_step
    assert agent.tool_complete_callback is original_tool


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_failed_conversation_restores_hooks_before_close(monkeypatch, tmp_path, failure):
    from contextlib import nullcontext
    import yaml
    worker = load_worker()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    events = []
    def flush(_rows):
        return True
    def conversation(*_args, **_kwargs):
        raise failure("Synthetic conversation failure")
    agent = SimpleNamespace(_flush_messages_to_session_db=flush, step_callback=None,
                            tool_complete_callback=None, run_conversation=conversation)
    def close():
        assert agent._flush_messages_to_session_db is flush
        assert agent.step_callback is None and agent.tool_complete_callback is None
        events.append("agent closed with original hooks")
    agent.close = close
    database = SimpleNamespace(get_compression_tip=lambda value: value,
        get_messages_as_conversation=lambda *_a, **_kw: [],
        close=lambda: events.append("database closed"))
    bridge = SimpleNamespace(server_port=9, serve_forever=lambda: None,
        shutdown=lambda: events.append("bridge stopped"), server_close=lambda: None)
    for name, value in {
        "hermes_yaml": yaml,
        "hermes_cli.config_defaults": SimpleNamespace(DEFAULT_CONFIG={}),
        "tools.skills_sync": SimpleNamespace(sync_skills=lambda **_kw: None),
        "run_agent": SimpleNamespace(AIAgent=lambda **_kw: agent),
        "hermes_state": SimpleNamespace(SessionDB=lambda _path: database),
        "tools.delegate_tool_config": SimpleNamespace(gateway_delegation_inline_scope=lambda _: nullcontext()),
        "agent": SimpleNamespace(title_generator=SimpleNamespace(_UPGRADE_THREADS=[])),
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(worker.http.server, "ThreadingHTTPServer", lambda *_a: bridge)
    monkeypatch.setattr(worker, "_register_boundaries", lambda *_a: None)
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"session_id": "synthetic", "model": "synthetic",
                                   "toolsets": [], "prompt": "Synthetic turn"}))
    with pytest.raises(failure, match="Synthetic conversation failure"):
        worker.main(str(request))
    assert events == ["agent closed with original hooks", "database closed", "bridge stopped"]


def test_startup_and_prior_tool_history_do_not_enable_steering(monkeypatch):
    worker = load_worker()
    requests, buffered = [], []
    def control(_path, payload):
        requests.append(payload)
        return {"messages": [{"id": "new-owner-message", "text": "Use green"}]}
    monkeypatch.setattr(worker, "control", control)
    agent = SimpleNamespace(_flush_messages_to_session_db=lambda _rows: True,
        steer=lambda text, ticket: (buffered.append((ticket, text)), True)[1],
        withdraw_steer=lambda _ticket: False)
    worker._Steering(agent, "current-work")
    prior_round = [{"name": "terminal", "result": "PRIOR_RESULT", "arguments": "{}"}]
    agent.step_callback(0, prior_round)
    agent.step_callback(1, prior_round)
    assert requests == [] and buffered == []
    agent.tool_complete_callback("current-tool-id", "terminal", {}, "CURRENT_RESULT")
    agent.step_callback(2, [{"name": "terminal", "result": "CURRENT_RESULT", "arguments": "{}"}])
    assert buffered == [("new-owner-message", "Use green")]


def test_each_followup_batch_requires_fresh_tool_completion(monkeypatch):
    worker = load_worker()
    phase = ["green"]
    requests, buffered = [], []
    def control(_path, _payload):
        requests.append(phase[0])
        return {"messages": [{"id": phase[0], "text": "Use " + phase[0]}]}
    monkeypatch.setattr(worker, "control", control)
    agent = SimpleNamespace(_flush_messages_to_session_db=lambda _rows: True,
        steer=lambda text, ticket: (buffered.append((ticket, text)), True)[1],
        withdraw_steer=lambda _ticket: False)
    worker._Steering(agent, "current-work")
    agent.tool_complete_callback("tool-1", "terminal", {}, "FIRST_RESULT")
    agent.step_callback(1, [{"name": "terminal", "result": "FIRST_RESULT"}])
    phase[0] = "red"
    agent.step_callback(2, [{"name": "terminal", "result": "FIRST_RESULT"}])
    assert requests == ["green"] and buffered == [("green", "Use green")]
    agent.tool_complete_callback("tool-2", "terminal", {}, "SECOND_RESULT")
    agent.step_callback(3, [{"name": "terminal", "result": "SECOND_RESULT"}])
    assert requests == ["green", "red"]
    assert buffered == [("green", "Use green"), ("red", "Use red")]


@pytest.mark.parametrize("edge", ["trailing_newline", "ack_failure", "large_batch"])
def test_native_receipts_survive_transport_and_batch_edges(monkeypatch, edge):
    worker = load_worker()
    monkeypatch.setitem(sys.modules, "agent.context_compressor",
                        SimpleNamespace(_DB_PERSISTED_MARKER="_durable"))
    count = 257 if edge == "large_batch" else 1
    tickets = [{"id": f"admission-{n}",
                "text": f"[Assist follow-up id=admission-{n}]\nMake it green" +
                        ("\n" if edge == "trailing_newline" else "")}
               for n in range(count)]
    buffered, receipts = [], []
    def control(_path, payload):
        if payload.get("consumed"):
            if edge == "ack_failure":
                raise OSError("Synthetic lost acknowledgment")
            assert len(payload["consumed"]) <= 256
            receipts.extend(payload["consumed"])
        return {"messages": tickets}
    monkeypatch.setattr(worker, "control", control)
    def flush(rows):
        for row in rows:
            row["_durable"] = True
        return True
    agent = SimpleNamespace(_flush_messages_to_session_db=flush,
        steer=lambda text, ticket: (buffered.append(text.strip()), True)[1],
        withdraw_steer=lambda _ticket: False)
    delivery = worker._Steering(agent, "logical-work")
    agent.tool_complete_callback("current-tool", "terminal", {}, "CURRENT_RESULT")
    agent.step_callback(0, [])
    rows = [native_steer_row(monkeypatch, agent, tickets)]
    assert agent._flush_messages_to_session_db(rows) is True
    assert rows[0]["display_metadata"]["assist_steering"]["run_ids"] == [item["id"] for item in tickets]
    if edge == "ack_failure":
        assert len(delivery.offered) == count
    else:
        assert receipts == sorted(item["id"] for item in tickets)
        assert not delivery.offered


@pytest.mark.parametrize("persisted", [True, False, None])
def test_worker_only_receipts_successful_native_flush(monkeypatch, persisted):
    worker = load_worker()
    monkeypatch.setitem(sys.modules, "agent.context_compressor",
                        SimpleNamespace(_DB_PERSISTED_MARKER="_durable"))
    requests, buffered = [], []
    tickets = [{"id": "admission-a", "text": "[Assist follow-up id=admission-a]\nSame text"},
               {"id": "admission-b", "text": "[Assist follow-up id=admission-b]\nSame text"}]
    def control(path, payload):
        requests.append((path, payload))
        return {"messages": tickets}
    monkeypatch.setattr(worker, "control", control)
    def flush(rows):
        if persisted is True:
            for row in rows:
                row["_durable"] = True
        return persisted
    agent = SimpleNamespace(_flush_messages_to_session_db=flush,
        steer=lambda text, ticket: (buffered.append((ticket, text)), True)[1],
        withdraw_steer=lambda ticket: True)
    delivery = worker._Steering(agent, "logical-work")
    agent.tool_complete_callback("current-tool", "terminal", {}, "CURRENT_RESULT")
    agent.step_callback(0, [])
    agent.step_callback(1, [])
    assert len(buffered) == 2
    row = native_steer_row(monkeypatch, agent, tickets)
    agent._flush_messages_to_session_db([row])
    assert row["display_metadata"]["assist_steering"] == {
        "work_id": "logical-work", "run_ids": ["admission-a", "admission-b"]}
    receipts = [payload for _path, payload in requests if payload.get("consumed")]
    assert bool(receipts) is (persisted is True)
    delivery.finish()
    assert agent.step_callback is None


def test_steering_does_not_hold_parking_lock_while_journal_is_busy(tmp_path):
    entered, release = threading.Event(), threading.Event()
    def callback(_ids):
        entered.set()
        assert release.wait(2)
        return []
    relay = HermesProviderRelay(tmp_path, "http://127.0.0.1:9/v1", "dummy", "dummy", 8192,
                                lambda: True, steering=callback)
    thread = threading.Thread(target=lambda: relay.steer({}))
    try:
        thread.start()
        assert entered.wait(1)
        assert relay.request_pause()
        relay.confirm_parked()
        assert relay.parked
    finally:
        release.set()
        thread.join(2)
        relay.close()
    assert not thread.is_alive()


@pytest.mark.parametrize("value", [{"consumed": "not-list"}, {"consumed": [None]}])
def test_relay_rejects_malformed_receipt_before_journal_access(tmp_path, value):
    relay = HermesProviderRelay(tmp_path, "http://127.0.0.1:9/v1", "dummy", "dummy", 8192,
                                lambda: True, steering=lambda _ids: pytest.fail("Invalid receipt"))
    try:
        with pytest.raises(HermesRelayError, match="Invalid native steering receipt"):
            relay.steer(value)
    finally:
        relay.close()
