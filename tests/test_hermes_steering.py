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
from manage.web import threads
from tests.test_web_hermes import hermes_threads


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
    agent.step_callback(0, [])
    rows = [{"role": "user", "display_kind": "steer", "content": "\n".join(buffered)}]
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
    agent.step_callback(0, [])
    agent.step_callback(1, [])
    assert len(buffered) == 2
    row = {"role": "user", "display_kind": "steer",
           "content": "\n".join(text for _ticket, text in buffered)}
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
