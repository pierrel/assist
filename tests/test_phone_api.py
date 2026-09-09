"""Phone API contract: auth, visible snapshots, and safe worktree data."""
from __future__ import annotations

import asyncio
import io
import os
import sqlite3
import tarfile
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import MessagesState, START, StateGraph

from assist.run_service import RunService
from manage.web import state
from manage.web.app import app
from manage.web import phone_api
from manage.web import run_stream
from manage.web.run_stream import RunStreamJournal, encode_sse


def _client(monkeypatch) -> TestClient:
    monkeypatch.setenv(phone_api.PHONE_API_TOKEN_ENV, "phone-test-token")
    return TestClient(app)


def _auth() -> dict[str, str]:
    return {"Authorization": "Bearer phone-test-token"}


def _thread_environment(tmp_path, monkeypatch, messages):
    thread_dir = tmp_path / "thread-a"
    workspace = thread_dir / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "notes.md").write_text("hello")
    monkeypatch.setattr(state.MANAGER, "thread_dir", lambda tid: str(tmp_path / tid))
    monkeypatch.setattr(state.MANAGER, "thread_default_working_dir",
                        lambda tid: str(tmp_path / tid / "workspace"))
    monkeypatch.setattr(phone_api, "_thread_messages", lambda tid: messages)
    monkeypatch.setattr(phone_api.threads, "_is_pi_thread", lambda tid: True)
    def fixture_history(tid, before):
        if before is None:
            return messages, False, False
        end = next((ordinal - 1 for ordinal, raw in enumerate(messages, start=1)
                    if phone_api._message_id(tid, raw, ordinal) == before), -1)
        if end < 0:
            raise phone_api.HTTPException(status_code=409, detail="History cursor is no longer available")
        return messages[:end], False, False
    monkeypatch.setattr(phone_api, "_thread_history", fixture_history)
    monkeypatch.setattr(phone_api, "read_thread_engine",
                        lambda directory: SimpleNamespace(name="deepagents"))
    monkeypatch.setattr(phone_api, "_thread_workspace", lambda tid: {
        "repo_key": "repo-a", "repo_label": "Repo A", "branch": "assist/demo",
        "revision": "abc", "dirty": True,
    })
    monkeypatch.setattr(state, "_thread_title", lambda tid: "🧪 release check")
    monkeypatch.setattr(state, "_get_status", lambda tid: {"stage": "ready"})
    return thread_dir


def test_phone_api_requires_a_configured_bearer_token(monkeypatch):
    monkeypatch.delenv(phone_api.PHONE_API_TOKEN_ENV, raising=False)
    client = TestClient(app)

    assert client.get("/api/v1/phone/threads").status_code == 503

    monkeypatch.setenv(phone_api.PHONE_API_TOKEN_ENV, "phone-test-token")
    assert client.get("/api/v1/phone/threads").status_code == 401


def test_phone_api_rejects_hidden_thread_directories(tmp_path, monkeypatch):
    thread_dir = _thread_environment(tmp_path, monkeypatch, [])
    (thread_dir / ".deleted").write_text("gone")

    response = _client(monkeypatch).get("/api/v1/phone/threads/thread-a", headers=_auth())

    assert response.status_code == 404


def test_phone_api_rejects_symlinked_thread_directories(tmp_path, monkeypatch):
    thread_dir = _thread_environment(tmp_path, monkeypatch, [])
    replacement = tmp_path / "replacement"
    thread_dir.rename(replacement)
    thread_dir.symlink_to(replacement, target_is_directory=True)

    response = _client(monkeypatch).get("/api/v1/phone/threads/thread-a", headers=_auth())

    assert response.status_code == 404


def test_phone_api_has_no_server_pin_endpoints(tmp_path, monkeypatch):
    _thread_environment(tmp_path, monkeypatch, [])
    client = _client(monkeypatch)

    assert client.get("/api/v1/phone/threads/thread-a/pins", headers=_auth()).status_code == 404
    assert client.post("/api/v1/phone/threads/thread-a/pins", headers=_auth(),
                       json={"response_id": "m-anything"}).status_code == 404
    assert client.delete("/api/v1/phone/threads/thread-a/pins/m-anything",
                         headers=_auth()).status_code == 404


def test_thread_list_uses_stored_titles_and_supplies_chooser_metadata(tmp_path, monkeypatch):
    _thread_environment(tmp_path, monkeypatch, [])
    monkeypatch.setattr(state.MANAGER, "list", lambda: ["thread-a"])
    monkeypatch.setattr(state.MANAGER, "get",
                        lambda tid: (_ for _ in ()).throw(AssertionError("no model title")))
    monkeypatch.setattr(state, "_get_status", lambda tid: {
        "stage": "ready", "domain": "https://example.com/repo.git",
    })
    monkeypatch.setattr(phone_api, "_thread_workspace",
                        lambda tid: (_ for _ in ()).throw(AssertionError("no Git worktree scan")))

    response = _client(monkeypatch).get("/api/v1/phone/threads", headers=_auth())

    assert response.status_code == 200
    thread = response.json()["threads"][0]
    assert thread["description"] == "thread-a"
    assert thread["search_description"] == "thread-a"
    assert thread["repo_key"] == phone_api._repo_key("https://example.com/repo.git")
    assert thread["repo_label"] == "repo"
    assert isinstance(thread["activity_at"], float)
    assert len(thread["revision"]) == 24


def test_thread_list_normalizes_only_leading_pictographs(tmp_path, monkeypatch):
    thread_dir = _thread_environment(tmp_path, monkeypatch, [])
    (thread_dir / "description.txt").write_text(" 🧪 #release check")
    monkeypatch.setattr(state.MANAGER, "list", lambda: ["thread-a"])

    response = _client(monkeypatch).get("/api/v1/phone/threads", headers=_auth())

    assert response.json()["threads"][0]["search_description"] == "#release check"


def test_create_queues_initialization_on_the_dedicated_scheduler(monkeypatch):
    scheduled = []
    monkeypatch.setattr(phone_api, "_create_and_submit",
                        lambda body, key, **kwargs: ("thread-a", SimpleNamespace(id="run-a", work_id=kwargs["work_id"]), "repo-a", False))
    monkeypatch.setattr(phone_api.threads._INITIALIZATION_SCHEDULER, "submit",
                        lambda run_id, tid, domain: scheduled.append((run_id, tid, domain)))

    response = _client(monkeypatch).post(
        "/api/v1/phone/threads", headers={**_auth(), "Idempotency-Key": "a" * 16},
        json={"message": "start"})

    assert response.status_code == 200
    assert scheduled == [("run-a", "thread-a", "repo-a")]


def test_message_queues_on_the_dedicated_scheduler(monkeypatch):
    scheduled = []
    monkeypatch.setattr(phone_api, "_submit_existing",
                        lambda tid, text, key, **kwargs: (SimpleNamespace(id="run-a", status="pending", work_id=kwargs["work_id"]), False, False))
    monkeypatch.setattr(phone_api.threads._RESUME_SCHEDULER, "submit",
                        lambda run_id, tid, **kwargs: scheduled.append((run_id, tid, kwargs)))

    response = _client(monkeypatch).post(
        "/api/v1/phone/threads/thread-a/messages",
        headers={**_auth(), "Idempotency-Key": "a" * 16}, json={"message": "continue"})

    assert response.status_code == 200
    assert scheduled == [("run-a", "thread-a", {"user_priority": True})]


def test_phone_thread_creation_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(state.MANAGER, "thread_dir", lambda tid: str(tmp_path / tid))
    monkeypatch.setattr(state.MANAGER, "list",
                        lambda: [f"phone-{index}" for index in range(phone_api.MAX_PHONE_THREADS)])

    with pytest.raises(phone_api.HTTPException) as error:
        phone_api._create_and_submit(phone_api._CreateThread(message="start"), "a" * 16)

    assert error.value.status_code == 429


def test_phone_thread_creation_allows_only_one_waiting_initialization(tmp_path, monkeypatch):
    monkeypatch.setattr(state.MANAGER, "thread_dir", lambda tid: str(tmp_path / tid))
    monkeypatch.setattr(state.MANAGER, "list", lambda: ["phone-existing"])
    monkeypatch.setattr(state, "_get_status", lambda tid: {"stage": "initializing"})

    with pytest.raises(phone_api.HTTPException) as error:
        phone_api._create_and_submit(phone_api._CreateThread(message="start"), "a" * 16)

    assert error.value.status_code == 429
    assert error.value.detail == "Phone initialization is busy"


def test_all_first_thread_routes_are_bounded_before_reserving_a_directory(monkeypatch):
    monkeypatch.setattr(phone_api.threads.MANAGER, "list",
                        lambda: ["thread-a", "thread-b"])
    monkeypatch.setattr(phone_api.threads, "_get_status",
                        lambda tid: {"stage": "initializing"})

    with pytest.raises(phone_api.HTTPException) as error:
        phone_api.threads.create_thread_with_message_core("start", domain=None)

    assert error.value.status_code == 429
    assert error.value.detail == "Thread setup is busy"


def _real_run_admission(tmp_path, monkeypatch):
    service = RunService(str(tmp_path))
    scheduled = []
    _thread_environment(tmp_path, monkeypatch, [])
    monkeypatch.setattr(phone_api.threads, "_runs", lambda: service)
    monkeypatch.setattr(phone_api.threads, "_is_pi_thread", lambda tid: False)
    monkeypatch.setattr(phone_api.threads, "_pi_message_admits", lambda tid: True)
    monkeypatch.setattr(phone_api.threads, "_get_status", lambda tid: {"stage": "ready"})
    monkeypatch.setattr(phone_api.threads, "_set_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(phone_api.threads.THREAD_QUEUE, "peek_holder", lambda: None)
    monkeypatch.setattr(phone_api.threads._RESUME_SCHEDULER, "submit",
                        lambda run_id, tid, **kwargs: scheduled.append((run_id, tid, kwargs)))
    return service, scheduled


def test_mature_thread_accepts_and_replays_one_phone_message(tmp_path, monkeypatch):
    service, scheduled = _real_run_admission(tmp_path, monkeypatch)
    for index in range(201):
        run = service.create("thread-a", "general-agent", f"historic {index}")
        service.transition("thread-a", run.id, "success")
    client = _client(monkeypatch)
    headers = {**_auth(), "Idempotency-Key": "mature-thread-key"}

    accepted = client.post(
        "/api/v1/phone/threads/thread-a/messages",
        headers=headers, json={"message": "continue"})
    replayed = client.post(
        "/api/v1/phone/threads/thread-a/messages",
        headers=headers, json={"message": "continue"})

    assert accepted.status_code == 200
    assert replayed.status_code == 200
    assert accepted.json()["replayed"] is False
    assert replayed.json() == {**accepted.json(), "replayed": True}
    assert len(service.list("thread-a")) == 202
    assert service.list("thread-a")[-1].status == "pending"
    assert scheduled == [(accepted.json()["run_id"], "thread-a", {"user_priority": True})]


def test_phone_pending_limit_returns_a_clear_backpressure_response(tmp_path, monkeypatch):
    service, scheduled = _real_run_admission(tmp_path, monkeypatch)
    for index in range(phone_api.MAX_PHONE_PENDING_RUNS):
        service.create("thread-a", "general-agent", f"pending {index}")

    response = _client(monkeypatch).post(
        "/api/v1/phone/threads/thread-a/messages",
        headers={**_auth(), "Idempotency-Key": "pending-limit-key"},
        json={"message": "continue"})

    assert response.status_code == 429
    assert response.json()["detail"] == "pending run limit reached"
    assert len(service.list("thread-a")) == phone_api.MAX_PHONE_PENDING_RUNS
    assert scheduled == []


def test_phone_message_rejects_a_pending_email_approval(tmp_path, monkeypatch):
    _thread_environment(tmp_path, monkeypatch, [])
    monkeypatch.setattr(phone_api.threads, "_pi_message_admits", lambda tid: True)
    monkeypatch.setattr(phone_api.threads, "_get_status",
                        lambda tid: {"stage": "paused", "pending_email_token": "token"})

    response = _client(monkeypatch).post(
        "/api/v1/phone/threads/thread-a/messages",
        headers={**_auth(), "Idempotency-Key": "pending-email-key"},
        json={"message": "continue"})

    assert response.status_code == 409
    assert response.json()["detail"] == "Resolve the pending approval first"


def test_phone_cannot_cancel_its_first_run_while_setup_is_pending(tmp_path, monkeypatch):
    _thread_environment(tmp_path, monkeypatch, [])
    monkeypatch.setattr(state, "_get_status", lambda tid: {
        "stage": "cloning", "pending_run_id": "run-a"})

    response = _client(monkeypatch).delete(
        "/api/v1/phone/threads/thread-a/runs/run-a", headers=_auth())

    assert response.status_code == 409
    assert response.json()["detail"] == "Thread setup is already in progress"


def test_phone_journal_hides_reservations_and_resets_the_retained_attempt():
    journal = RunStreamJournal()

    assert journal.reserve("thread-a", "work-a")
    assert journal.read("thread-a", "work-a") is None
    assert journal.activate("thread-a", "work-a")
    assert journal.publish_delta("thread-a", "work-a", "first") == {
        "attempt": 1, "index": 1, "text": "first"}
    assert journal.reset_attempt("thread-a", "work-a") == 2
    assert journal.publish_delta("thread-a", "work-a", "second") == {
        "attempt": 2, "index": 1, "text": "second"}
    assert journal.read("thread-a", "work-a") == {
        "attempt": 2, "deltas": [{"attempt": 2, "index": 1, "text": "second"}],
        "truncated": False, "terminal": False}


def test_phone_journal_saturates_status_only_then_evicts_only_terminal(monkeypatch):
    monkeypatch.setattr(run_stream, "MAX_WORKS", 2)
    journal = RunStreamJournal()
    assert journal.reserve("t", "active")
    assert journal.reserve("t", "terminal")
    assert journal.activate("t", "active")
    assert journal.activate("t", "terminal")
    assert not journal.reserve("t", "status-only")
    journal.finish("t", "terminal")
    assert journal.reserve("t", "replacement")
    assert journal.read("t", "active") is not None
    assert journal.read("t", "terminal") is None


def test_phone_journal_truncates_once_and_stops_further_publication(monkeypatch):
    monkeypatch.setattr(run_stream, "MAX_TEXT_BYTES", 1)
    journal = RunStreamJournal()
    assert journal.reserve("t", "w") and journal.activate("t", "w")
    assert journal.publish_delta("t", "w", "x") is not None
    assert journal.publish_delta("t", "w", "y") is None
    assert journal.read("t", "w")["truncated"]
    assert journal.publish_delta("t", "w", "z") is None


def test_phone_sse_encoder_uses_real_record_newlines_and_json_escapes_control_text():
    record = encode_sse("assistant-delta", {"text": "x\nevent: forged\rdata: y"})

    assert record.splitlines() == [
        "event: assistant-delta", 'data: {"text":"x\\nevent: forged\\rdata: y"}', ""]
    assert record.endswith("\n\n")
    assert "\\n" in record and "\\r" in record


def test_phone_sse_rejects_an_encoded_record_over_its_transport_bound():
    with pytest.raises(ValueError, match="record exceeds bound"):
        phone_api._sse("assistant-delta", {"text": "\\" * phone_api.MAX_PHONE_SSE_RECORD_BYTES})


def test_phone_run_events_replay_reset_delta_truncation_status_and_terminal(monkeypatch):
    """The real route emits one ordered, replayable observation before terminal truth."""
    journal = RunStreamJournal()
    assert journal.reserve("thread-a", "work-a") and journal.activate("thread-a", "work-a")
    monkeypatch.setattr(run_stream, "MAX_TEXT_BYTES", 1)
    assert journal.publish_delta("thread-a", "work-a", "x") is not None
    assert journal.publish_delta("thread-a", "work-a", "y") is None
    monkeypatch.setattr(phone_api, "RUN_STREAMS", journal)
    monkeypatch.setattr(phone_api, "_logical_status", lambda tid, run_id: {
        "id": run_id, "thread_id": tid, "work_id": "work-a", "physical_run_id": "slice-a",
        "status": "success", "error": None, "updated_at": 1, "thread_status": "ready"})

    response = _client(monkeypatch).get(
        "/api/v1/phone/threads/thread-a/runs/run-a/events", headers=_auth())

    assert response.status_code == 200
    assert [line.removeprefix("event: ") for line in response.text.splitlines()
            if line.startswith("event: ")] == [
                "assistant-reset", "assistant-delta", "assistant-truncated", "status", "terminal"]
    assert all(line.startswith("data: {") for line in response.text.splitlines()
               if line.startswith("data: "))


def test_phone_run_events_reports_the_bounded_route_timeout(monkeypatch):
    """The route has an explicit error outcome instead of silently ending a live Run."""
    monkeypatch.setattr(phone_api, "range", lambda *_args: range(1), raising=False)
    monkeypatch.setattr(phone_api, "_logical_status", lambda tid, run_id: {
        "id": run_id, "thread_id": tid, "work_id": "work-a", "physical_run_id": "slice-a",
        "status": "pending", "error": None, "updated_at": 1, "thread_status": "ready"})

    response = _client(monkeypatch).get(
        "/api/v1/phone/threads/thread-a/runs/run-a/events", headers=_auth())

    assert response.status_code == 200
    assert [line.removeprefix("event: ") for line in response.text.splitlines()
            if line.startswith("event: ")] == ["status", "error"]


def test_phone_reservation_discards_on_replay_and_submission_failure(monkeypatch):
    journal = RunStreamJournal()
    monkeypatch.setattr(phone_api, "RUN_STREAMS", journal)
    monkeypatch.setattr(phone_api.threads, "_is_pi_thread", lambda _: False)
    existing = SimpleNamespace(id="replayed", work_id="prior-work")
    assert journal.reserve("thread-a", "prior-work") and journal.activate("thread-a", "prior-work")
    monkeypatch.setattr(phone_api, "_submit_existing",
                        lambda *_args, **_kwargs: (existing, False, True))

    run, busy, replay, live_text = phone_api._reserve_existing("thread-a", "hello", "key")

    assert run is existing and not busy and replay and live_text
    assert list(journal._entries) == [("thread-a", "prior-work")]

    monkeypatch.setattr(phone_api, "_submit_existing",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("no durable run")))
    with pytest.raises(RuntimeError, match="no durable run"):
        phone_api._reserve_existing("thread-a", "hello", "key")
    assert list(journal._entries) == [("thread-a", "prior-work")]


def test_phone_create_reservation_is_active_but_initialization_stays_status_only(monkeypatch):
    journal = RunStreamJournal()
    monkeypatch.setattr(phone_api, "RUN_STREAMS", journal)
    monkeypatch.setattr(phone_api, "_create_and_submit",
                        lambda body, key, **kwargs: (
                            phone_api._phone_thread_id(key),
                            SimpleNamespace(id=kwargs["run_id"], work_id=kwargs["work_id"]), None, False))

    tid, run, _domain, replay, live_text = phone_api._reserve_create(
        phone_api._CreateThread(message="start"), "a" * 16)
    assert tid == phone_api._phone_thread_id("a" * 16) and not replay and live_text
    assert journal.read(tid, run.work_id) is not None

    _tid, pi_run, _domain, replay, live_text = phone_api._reserve_create(
        phone_api._CreateThread(message="start", harness="pi"), "b" * 16)
    assert not replay and not live_text
    assert journal.read(_tid, pi_run.work_id) is None


def test_logical_projection_uses_one_run_snapshot_for_successor_and_crash_states(monkeypatch):
    def run(identifier, work_id, status):
        return SimpleNamespace(id=identifier, work_id=work_id, status=status,
                               updated_at=1, error=None)

    cases = [
        ([run("run-a", "work-a", "interrupted"), run("run-b", "work-a", "pending")],
         "paused", "pending", "run-b"),
        ([run("run-a", "work-a", "interrupted")], "paused", "transitioning", "run-a"),
        ([run("run-a", "work-a", "interrupted")], "ready", "interrupted", "run-a"),
        ([run("run-a", "work-a", "interrupted"), run("run-z", "other", "running")],
         "paused", "interrupted", "run-a"),
    ]
    monkeypatch.setattr(phone_api, "_thread_dir", lambda _tid: None)
    for runs, stage, expected, physical in cases:
        monkeypatch.setattr(phone_api.threads, "_runs", lambda: SimpleNamespace(list=lambda _tid: runs))
        monkeypatch.setattr(state, "_get_status", lambda _tid, stage=stage: {"stage": stage})
        projection = phone_api._logical_status("thread-a", "run-a")
        assert (projection["status"], projection["physical_run_id"]) == (expected, physical)


def test_logical_cancel_closes_only_its_chain_and_dispatches_one_follower(monkeypatch):
    def run(identifier, work_id, status):
        return SimpleNamespace(id=identifier, work_id=work_id, status=status,
                               updated_at=1, error=None)

    predecessor = run("run-a", "work-a", "interrupted")
    successor = run("run-b", "work-a", "pending")
    follower = run("run-c", "work-b", "pending")
    stale = run("run-d", "old-work", "interrupted")
    records = [predecessor, successor, follower, stale]
    dispatched, statuses = [], ["paused"]

    class Runs:
        def list(self, _tid):
            return records

        def cancel_pending(self, _tid, run_id):
            selected = next(item for item in records if item.id == run_id)
            assert selected.status == "pending"
            selected.status = "cancelled"
            return selected

        def cancel(self, _tid, run_id):
            selected = next(item for item in records if item.id == run_id)
            selected.status = "cancelled"
            return selected

    journal = RunStreamJournal()
    assert journal.reserve("thread-a", "work-a") and journal.activate("thread-a", "work-a")
    monkeypatch.setattr(phone_api, "RUN_STREAMS", journal)
    monkeypatch.setattr(phone_api, "_thread_dir", lambda _tid: None)
    monkeypatch.setattr(phone_api.threads, "_runs", Runs)
    monkeypatch.setattr(state, "_get_status", lambda _tid: {"stage": statuses[-1]})
    monkeypatch.setattr(phone_api.threads, "_set_status",
                        lambda _tid, stage, **_kw: statuses.append(stage))
    monkeypatch.setattr(phone_api.threads, "_dispatch_pending_after",
                        lambda tid, run_id: dispatched.append((tid, run_id)))

    code, value = phone_api._cancel_logical_run("thread-a", "run-a")
    again, repeated = phone_api._cancel_logical_run("thread-a", "run-a")

    assert code == 200 and value["outcome"] == "cancelled"
    assert [item.status for item in records] == ["cancelled", "cancelled", "pending", "interrupted"]
    assert statuses == ["paused", "ready"]
    assert dispatched == [("thread-a", "run-b")]
    assert journal.read("thread-a", "work-a")["terminal"]
    assert again == 409 and repeated["outcome"] == "cancelled"


def test_sse_outer_lifetime_closes_the_iterator_after_multiple_sublimit_sends(monkeypatch):
    """One absolute deadline covers accumulated send time, then releases the body."""
    monkeypatch.setattr(phone_api, "PHONE_SSE_LIFETIME_SECONDS", 0.015)
    monkeypatch.setattr(phone_api, "PHONE_SSE_SEND_TIMEOUT_SECONDS", 0.1)
    closed, messages = [], []

    async def body():
        try:
            while True:
                yield b"x"
        finally:
            closed.append(True)

    async def send(message):
        messages.append(message)
        await asyncio.sleep(0.008)

    async def exercise():
        response = phone_api._BoundedSSEStreamingResponse(body())
        await response.stream_response(send)

    asyncio.run(exercise())

    assert len(messages) >= 2
    assert closed == [True]


def test_sse_blocked_send_hits_outer_deadline_and_releases_its_stream_slot(monkeypatch):
    """A stalled ASGI consumer cannot retain the slot after the observation deadline."""
    monkeypatch.setattr(phone_api, "PHONE_SSE_LIFETIME_SECONDS", 0.01)
    monkeypatch.setattr(phone_api, "PHONE_SSE_SEND_TIMEOUT_SECONDS", 0.1)
    slot = threading.BoundedSemaphore(1)
    assert slot.acquire(blocking=False)
    closed = []

    async def body():
        try:
            yield b"x"
        finally:
            closed.append(True)

    async def blocked_send(_message):
        await asyncio.sleep(1)

    async def exercise():
        response = phone_api._BoundedSSEStreamingResponse(body(), on_close=slot.release)
        await response.stream_response(blocked_send)

    asyncio.run(exercise())

    assert closed == []  # Header backpressure can prevent the generator from starting.
    assert slot.acquire(blocking=False)


def test_busy_phone_acceptance_advertises_its_active_future_worker_journal(monkeypatch):
    journal = RunStreamJournal()
    monkeypatch.setattr(phone_api, "RUN_STREAMS", journal)
    monkeypatch.setattr(phone_api.threads, "_is_pi_thread", lambda _: False)
    monkeypatch.setattr(phone_api, "_submit_existing",
                        lambda *_args, **kwargs: (SimpleNamespace(
                            id=kwargs["run_id"], work_id=kwargs["work_id"]), True, False))

    run, busy, replay, live_text = phone_api._reserve_existing("thread-a", "hello", "key")

    assert busy and not replay and live_text
    assert journal.read("thread-a", run.work_id) is not None


def test_phone_reservation_activates_before_immediate_durable_worker_publish(monkeypatch):
    """A worker interleaved inside admission always finds its active journal."""
    journal = RunStreamJournal()
    monkeypatch.setattr(phone_api, "RUN_STREAMS", journal)
    monkeypatch.setattr(phone_api.threads, "_is_pi_thread", lambda _: False)
    published = []

    def submit(tid, _text, _key, **kwargs):
        # This is the first point at which a durable Run would be visible to a
        # busy holder.  Publish immediately, before the route receives a reply.
        value = journal.publish_delta(tid, kwargs["work_id"], "first")
        published.append(value)
        return SimpleNamespace(id=kwargs["run_id"], work_id=kwargs["work_id"]), False, False

    monkeypatch.setattr(phone_api, "_submit_existing", submit)
    run, busy, replay, live_text = phone_api._reserve_existing("thread-a", "hello", "key")

    assert not busy and not replay and live_text
    assert published == [{"attempt": 1, "index": 1, "text": "first"}]
    assert journal.read("thread-a", run.work_id)["deltas"] == published


def test_phone_reservation_discards_when_activation_itself_fails(monkeypatch):
    """The pre-admission staging slot cannot leak through an activation exception."""
    journal = RunStreamJournal()
    monkeypatch.setattr(phone_api, "RUN_STREAMS", journal)
    monkeypatch.setattr(phone_api.threads, "_is_pi_thread", lambda _: False)
    monkeypatch.setattr(journal, "activate",
                        lambda *_args: (_ for _ in ()).throw(RuntimeError("activate failed")))

    with pytest.raises(RuntimeError, match="activate failed"):
        phone_api._reserve_existing("thread-a", "hello", "key")

    assert not journal._entries


def test_logical_resolver_never_exposes_an_active_candidate_without_a_run(monkeypatch):
    """Journal activation is not external visibility before durable acceptance."""
    journal = RunStreamJournal()
    assert journal.reserve("thread-a", "candidate")
    assert journal.activate("thread-a", "candidate")
    monkeypatch.setattr(phone_api, "RUN_STREAMS", journal)
    monkeypatch.setattr(phone_api, "_thread_dir", lambda _tid: None)
    monkeypatch.setattr(phone_api.threads, "_runs", lambda: SimpleNamespace(list=lambda _tid: []))

    with pytest.raises(phone_api.HTTPException) as error:
        phone_api._logical_status("thread-a", "not-a-run")

    assert error.value.status_code == 404
    assert journal.read("thread-a", "candidate") is not None


def test_pi_phone_acceptance_never_advertises_unserved_live_text(monkeypatch):
    journal = RunStreamJournal()
    monkeypatch.setattr(phone_api, "RUN_STREAMS", journal)
    monkeypatch.setattr(phone_api.threads, "_is_pi_thread", lambda _: True)
    monkeypatch.setattr(phone_api, "_submit_existing",
                        lambda *_args, **kwargs: (SimpleNamespace(
                            id=kwargs["run_id"], work_id=kwargs["work_id"]), False, False))

    run, busy, replay, live_text = phone_api._reserve_existing("thread-a", "hello", "key")

    assert not busy and not replay and not live_text
    assert journal.read("thread-a", run.work_id) is None


def test_archive_rejects_a_concurrent_download(monkeypatch):
    monkeypatch.setattr(phone_api, "_ARCHIVE_SLOTS", phone_api.threading.BoundedSemaphore(0))

    response = _client(monkeypatch).get(
        "/api/v1/phone/threads/thread-a/workspace/archive", headers=_auth())

    assert response.status_code == 429


def test_initialization_scheduler_never_uses_a_request_worker(monkeypatch):
    scheduler = phone_api.threads._InitializationScheduler()
    completed = threading.Event()
    seen = []
    monkeypatch.setattr(phone_api.threads, "_initialize_thread",
                        lambda tid, run_id, domain, rider=None: (
                            seen.append((tid, run_id, domain, rider)), completed.set()))

    scheduler.start()
    scheduler.submit("run-a", "thread-a", "repo-a")

    assert completed.wait(1)
    assert seen == [("thread-a", "run-a", "repo-a", None)]


def test_blocked_initialization_cannot_stall_the_run_scheduler(monkeypatch):
    initializer = phone_api.threads._InitializationScheduler()
    scheduler = phone_api.threads._ResumeScheduler()
    cloning, release, ran = threading.Event(), threading.Event(), threading.Event()
    monkeypatch.setattr(phone_api.threads, "_initialize_thread",
                        lambda *args: (cloning.set(), release.wait(1)))
    monkeypatch.setattr(phone_api.threads, "_execute_run",
                        lambda *args, **kwargs: ran.set())

    initializer.start()
    initializer.submit("clone-run", "clone-thread", "repo-a")
    assert cloning.wait(1)
    scheduler.start()
    scheduler.submit("turn-run", "turn-thread")

    assert ran.wait(1)
    release.set()


def test_snapshot_exposes_visible_messages_and_safe_file_references(tmp_path, monkeypatch):
    _thread_environment(tmp_path, monkeypatch, [
        {"role": "user", "content": "what changed", "message_id": "u1"},
        {"role": "tools", "content": "private tool output", "message_id": "tool1"},
        {"role": "assistant", "content": "Read notes.md and /etc/passwd", "message_id": "a1"},
    ])
    client = _client(monkeypatch)

    response = client.get("/api/v1/phone/threads/thread-a", headers=_auth())

    assert response.status_code == 200
    payload = response.json()
    assert [message["role"] for message in payload["messages"]] == ["user", "assistant"]
    answer = payload["messages"][-1]
    assert answer["id"].startswith("m-")
    assert answer["file_refs"] == [{"path": "notes.md", "label": "notes.md"}]
    assert "private tool output" not in response.text
    assert response.headers["cache-control"] == "no-store"


def test_snapshot_never_exposes_a_backend_error(tmp_path, monkeypatch):
    _thread_environment(tmp_path, monkeypatch, [])
    monkeypatch.setattr(state, "_get_status", lambda tid: {
        "stage": "error", "error": "clone https://user:secret@example.invalid/repo.git failed",
    })

    response = _client(monkeypatch).get("/api/v1/phone/threads/thread-a", headers=_auth())

    assert response.json()["thread"]["error"] == "Thread failed; inspect Assist Web for details."
    assert "secret" not in response.text


def test_logical_run_status_never_exposes_a_backend_error(monkeypatch):
    monkeypatch.setattr(phone_api, "_thread_dir", lambda tid: "/thread-a")
    monkeypatch.setattr(phone_api.threads, "_runs", lambda: SimpleNamespace(
        list=lambda tid: [SimpleNamespace(
            id="run-a", work_id="work-a", status="error",
            error="clone https://user:secret@example.invalid/repo.git failed",
            updated_at="2026-09-04T05:00:00+00:00")]))
    monkeypatch.setattr(state, "_get_status", lambda tid: {"stage": "error"})

    status = phone_api._logical_status("thread-a", "run-a")

    assert status["error"] == "Run failed; inspect Assist Web for details."


def test_snapshot_marks_the_live_last_user_message_incomplete(tmp_path, monkeypatch):
    _thread_environment(tmp_path, monkeypatch, [
        {"role": "user", "content": "still working", "message_id": "u1",
         "timestamp": "2026-09-04T05:00:00+00:00"},
    ])
    monkeypatch.setattr(state, "_get_status", lambda tid: {"stage": "processing"})

    message = _client(monkeypatch).get(
        "/api/v1/phone/threads/thread-a", headers=_auth()).json()["messages"][0]

    assert message["timestamp"] == "2026-09-04T05:00:00+00:00"
    assert message["state"] == "incomplete"


def test_snapshot_revision_changes_when_completion_state_changes(tmp_path, monkeypatch):
    _thread_environment(tmp_path, monkeypatch, [
        {"role": "user", "content": "still working", "message_id": "u1"},
    ])
    status = {"stage": "processing"}
    monkeypatch.setattr(state, "_get_status", lambda tid: status)
    client = _client(monkeypatch)

    processing = client.get("/api/v1/phone/threads/thread-a", headers=_auth()).json()
    status["stage"] = "ready"
    ready = client.get("/api/v1/phone/threads/thread-a", headers=_auth()).json()

    assert processing["thread"]["revision"] != ready["thread"]["revision"]


def test_snapshot_history_pages_are_bounded_and_nonoverlapping(tmp_path, monkeypatch):
    messages = [{"role": "user", "content": f"message {index}",
                 "message_id": f"u{index}"} for index in range(161)]
    _thread_environment(tmp_path, monkeypatch, messages)
    client = _client(monkeypatch)

    newest = client.get("/api/v1/phone/threads/thread-a", headers=_auth()).json()
    older = client.get(
        f"/api/v1/phone/threads/thread-a/history?before={newest['next_before']}",
        headers=_auth()).json()
    oldest = client.get(
        f"/api/v1/phone/threads/thread-a/history?before={older['next_before']}",
        headers=_auth()).json()

    assert len(newest["messages"]) == 80
    assert len(older["messages"]) == 80
    assert len(oldest["messages"]) == 1
    assert newest["messages"][0]["text"] == "message 81"
    assert older["messages"][0]["text"] == "message 1"
    assert oldest["messages"][0]["text"] == "message 0"
    assert newest["next_before"].startswith("m-")
    assert older["next_before"].startswith("m-")
    assert oldest["next_before"] is None


def test_history_cursor_is_stable_when_a_new_message_arrives(tmp_path, monkeypatch):
    messages = [{"role": "user", "content": f"message {index}",
                 "message_id": f"u{index}"} for index in range(161)]
    _thread_environment(tmp_path, monkeypatch, messages)
    client = _client(monkeypatch)

    newest = client.get("/api/v1/phone/threads/thread-a", headers=_auth()).json()
    messages.append({"role": "assistant", "content": "new reply", "message_id": "a-new"})
    older = client.get(
        f"/api/v1/phone/threads/thread-a/history?before={newest['next_before']}",
        headers=_auth()).json()

    assert [message["text"] for message in older["messages"]] == [
        f"message {index}" for index in range(1, 81)]


def test_history_cursor_survives_a_page_of_hidden_records(tmp_path, monkeypatch):
    messages = [{"role": "user", "content": "visible", "message_id": "u1"}] + [
        {"role": "tools", "content": "hidden", "message_id": f"t{index}"}
        for index in range(80)]
    _thread_environment(tmp_path, monkeypatch, messages)
    client = _client(monkeypatch)

    hidden = client.get("/api/v1/phone/threads/thread-a", headers=_auth()).json()
    older = client.get(
        f"/api/v1/phone/threads/thread-a/history?before={hidden['next_before']}",
        headers=_auth()).json()

    assert hidden["messages"] == [] and hidden["next_before"].startswith("m-")
    assert [message["text"] for message in older["messages"]] == ["visible"]


def test_checkpoint_history_pages_message_writes_without_hydrating_graph_state(tmp_path, monkeypatch):
    """The phone reader pages LangGraph's append writes, never get_state()."""
    thread_dir = tmp_path / "thread-a"
    (thread_dir / "workspace").mkdir(parents=True)
    (thread_dir / "workspace" / "notes.md").write_text("hello")
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    saver = SqliteSaver(connection)
    graph = StateGraph(MessagesState)
    graph.add_node("reply", lambda _state: {"messages": [{"role": "assistant", "content": "ok"}]})
    graph.add_edge(START, "reply")
    compiled = graph.compile(checkpointer=saver)
    for index in range(3):
        compiled.invoke({"messages": [{"role": "user", "content": f"message {index}"}]},
                        {"configurable": {"thread_id": "thread-a"}})
    monkeypatch.setattr(state.MANAGER, "checkpointer", saver)
    monkeypatch.setattr(state.MANAGER, "thread_dir", lambda tid: str(tmp_path / tid))
    monkeypatch.setattr(state.MANAGER, "thread_default_working_dir",
                        lambda tid: str(tmp_path / tid / "workspace"))
    monkeypatch.setattr(phone_api, "read_thread_engine", lambda directory: SimpleNamespace(name="deepagents"))
    monkeypatch.setattr(phone_api, "_thread_workspace", lambda tid: {
        "repo_key": None, "repo_label": "No repository", "branch": None,
        "revision": None, "dirty": False,
    })
    monkeypatch.setattr(state, "_get_status", lambda tid: {"stage": "ready"})
    monkeypatch.setattr(phone_api, "MAX_HISTORY_MESSAGES", 2)
    client = _client(monkeypatch)

    newest = client.get("/api/v1/phone/threads/thread-a", headers=_auth()).json()
    compiled.invoke({"messages": [{"role": "user", "content": "message 3"}]},
                    {"configurable": {"thread_id": "thread-a"}})
    older = client.get(
        f"/api/v1/phone/threads/thread-a/history?before={newest['next_before']}",
        headers=_auth()).json()

    assert [message["text"] for message in newest["messages"]] == ["message 2", "ok"]
    assert [message["text"] for message in older["messages"]] == ["message 1", "ok"]
    assert newest["next_before"].startswith("c-")

    monkeypatch.setattr(phone_api, "_thread_messages",
                        lambda tid: (_ for _ in ()).throw(AssertionError("no full state read")))


def test_pi_message_ids_remain_stable_across_reordering(tmp_path, monkeypatch):
    _thread_environment(tmp_path, monkeypatch, [])
    message = {"role": "assistant", "content": "Pi reply", "message_id": "pi:run-1"}

    assert phone_api._message_id("thread-a", message, 1) == phone_api._message_id(
        "thread-a", message, 99)
    assert phone_api._message_id("thread-a", message, 1) != phone_api._message_id(
        "thread-a", {**message, "role": "user"}, 1)


def test_workspace_archive_omits_git_and_symlinked_host_files(tmp_path, monkeypatch):
    thread_dir = _thread_environment(tmp_path, monkeypatch, [])
    workspace = thread_dir / "workspace"
    (workspace / ".git").mkdir()
    (workspace / ".git" / "config").write_text("private")
    (workspace / "host-link").symlink_to("/etc/passwd")
    client = _client(monkeypatch)

    response = client.get("/api/v1/phone/threads/thread-a/workspace", headers=_auth())
    archive = client.get("/api/v1/phone/threads/thread-a/workspace/archive", headers=_auth())

    assert response.status_code == 200
    assert response.json()["files"] == [{"path": "notes.md", "type": "file", "size": 5}]
    assert archive.status_code == 200
    assert b"private" not in archive.content


def test_workspace_listing_closes_fwalk_after_a_truncated_walk(tmp_path, monkeypatch):
    thread_dir = _thread_environment(tmp_path, monkeypatch, [])
    workspace = thread_dir / "workspace"
    closed = False

    class Walker:
        def __init__(self):
            self.fd = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY)

        def __iter__(self):
            yield ".", [], ["notes.md"], self.fd

        def close(self):
            nonlocal closed
            closed = True
            os.close(self.fd)

    monkeypatch.setattr(phone_api.os, "fwalk", lambda *args, **kwargs: Walker())
    monkeypatch.setattr(phone_api, "MAX_WORKSPACE_NODES", 0)

    assert phone_api._workspace_entries("thread-a") == []
    assert closed


def test_workspace_archive_rechecks_a_file_that_turns_into_a_symlink(tmp_path, monkeypatch):
    thread_dir = _thread_environment(tmp_path, monkeypatch, [])
    workspace = thread_dir / "workspace"
    listed = [{"path": "notes.md", "type": "file", "size": 5}]
    (workspace / "notes.md").unlink()
    (workspace / "notes.md").symlink_to("/etc/hostname")
    monkeypatch.setattr(phone_api, "_workspace_entries", lambda *args, **kwargs: listed)

    archive = _client(monkeypatch).get(
        "/api/v1/phone/threads/thread-a/workspace/archive", headers=_auth())

    with tarfile.open(fileobj=io.BytesIO(archive.content), mode="r:gz") as result:
        assert result.getnames() == []


def test_workspace_archive_rejects_an_intermediate_symlink(tmp_path, monkeypatch):
    thread_dir = _thread_environment(tmp_path, monkeypatch, [])
    workspace = thread_dir / "workspace"
    nested = workspace / "nested"
    nested.mkdir()
    (nested / "hostname").write_text("not-a-host-file")
    listed = [{"path": "nested/hostname", "type": "file", "size": 15}]
    (nested / "hostname").unlink()
    nested.rmdir()
    nested.symlink_to("/etc")
    monkeypatch.setattr(phone_api, "_workspace_entries", lambda *args, **kwargs: listed)

    archive = _client(monkeypatch).get(
        "/api/v1/phone/threads/thread-a/workspace/archive", headers=_auth())

    with tarfile.open(fileobj=io.BytesIO(archive.content), mode="r:gz") as result:
        assert result.getnames() == []


def test_workspace_root_symlink_is_rejected(tmp_path, monkeypatch):
    thread_dir = _thread_environment(tmp_path, monkeypatch, [])
    workspace = thread_dir / "workspace"
    (workspace / "notes.md").unlink()
    workspace.rmdir()
    workspace.symlink_to("/etc")
    client = _client(monkeypatch)

    manifest = client.get("/api/v1/phone/threads/thread-a/workspace", headers=_auth())
    archive = client.get("/api/v1/phone/threads/thread-a/workspace/archive", headers=_auth())

    assert manifest.json()["files"] == []
    assert archive.status_code == 409


def test_workspace_manifest_bounds_empty_directory_traversal(tmp_path, monkeypatch):
    thread_dir = _thread_environment(tmp_path, monkeypatch, [])
    workspace = thread_dir / "workspace"
    for index in range(5):
        (workspace / f"empty-{index}").mkdir()
    monkeypatch.setattr(phone_api, "MAX_WORKSPACE_NODES", 3)

    response = _client(monkeypatch).get(
        "/api/v1/phone/threads/thread-a/workspace", headers=_auth())

    assert response.json()["truncated"] is True
