"""Exact phone approval admission, with no model or mail delivery."""
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from assist.run_service import RunService
from assist.thread import Thread
from manage.web import phone_api, state, threads
from manage.web.app import app


@pytest.fixture
def approval(tmp_path, monkeypatch):
    (tmp_path / "thread-a").mkdir()
    monkeypatch.setattr(state.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(state.MANAGER, "thread_dir", lambda tid: str(tmp_path / tid))
    monkeypatch.setattr(threads, "_require_deep_thread", lambda tid: None)
    runs = RunService(str(tmp_path))
    monkeypatch.setattr(threads, "_runs", lambda: runs)
    monkeypatch.setenv(phone_api.PHONE_API_TOKEN_ENV, "phone-test-token")
    monkeypatch.setenv("EMAIL_FROM_ADDRESS", "assistant@example.test")
    monkeypatch.setenv("EMAIL_FROM_NAME", "Assistant")
    monkeypatch.setenv("EMAIL_ALWAYS_CC", "oversight@example.test")
    scheduled = []
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit", lambda *a, **kw: scheduled.append(a))
    state._set_status("thread-a", "awaiting_approval", pending_email_token="original-token",
                      pending_email_to="recipient@example.test", pending_email_subject="Subject",
                      pending_email_body="Complete\nbody", pending_run_id="old-run")
    return TestClient(app), runs, scheduled


HEADERS = {"Authorization": "Bearer phone-test-token"}
URL = "/api/v1/phone/threads/thread-a/approval"


def _proposal(client):
    response = client.get(URL, headers=HEADERS)
    assert response.status_code == 200
    return response.json()["proposal"]


def test_preview_is_authenticated_complete_and_not_cacheable(approval):
    client, _, _ = approval
    assert client.get(URL).status_code == 401
    response = client.get(URL, headers=HEADERS)
    assert response.headers["cache-control"] == "no-store"
    proposal = response.json()["proposal"]
    assert proposal["from"] == "Assistant <assistant@example.test>"
    assert proposal["cc"] == "oversight@example.test"
    assert proposal["action"]["args"] == {
        "to": "recipient@example.test", "subject": "Subject", "body": "Complete\nbody"}


@pytest.mark.parametrize("decision", ["approve", "reject", "edit"])
def test_decisions_are_bound_to_body_and_fixed_cc(approval, monkeypatch, decision):
    client, runs, scheduled = approval
    proposal = _proposal(client)
    monkeypatch.setenv("EMAIL_ALWAYS_CC", "different@example.test")
    response = client.post(URL, headers=HEADERS, json={
        "kind": "send_email", "token": proposal["token"], "decision": decision})
    assert response.status_code == 409
    assert runs.list("thread-a") == [] and scheduled == []


def test_edited_email_uses_only_the_user_editable_fields(approval):
    client, runs, scheduled = approval
    proposal = _proposal(client)
    response = client.post(URL, headers=HEADERS, json={
        "kind": "send_email", "token": proposal["token"], "decision": "edit",
        "to": "edited@example.test", "subject": "Edited", "body": "New\nbody"})
    assert response.status_code == 200 and len(scheduled) == 1
    assert runs.get("thread-a", response.json()["run_id"]).resume_decision == {
        "type": "edit", "edited_action": {"name": "send_email", "args": {
            "to": "edited@example.test", "subject": "Edited", "body": "New\nbody"}}}
    assert _proposal(client) is None


def test_concurrent_decisions_create_only_one_resume(approval):
    client, runs, _ = approval
    token = _proposal(client)["token"]
    barrier = threading.Barrier(2)

    def decide():
        barrier.wait()
        try:
            return threads.email_decision_core("thread-a", "approve", token, preview_token=True)
        except HTTPException as error:
            return error.status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: decide(), range(2)))
    assert sum(isinstance(result, tuple) for result in results) == 1
    assert 409 in results
    assert len(runs.list("thread-a")) == 1


def test_duplicate_old_token_cannot_decide_a_new_proposal(approval):
    client, runs, scheduled = approval
    old = _proposal(client)
    payload = {"kind": "send_email", "token": old["token"], "decision": "approve"}
    assert client.post(URL, headers=HEADERS, json=payload).status_code == 200
    state._set_status("thread-a", "awaiting_approval", pending_email_token="new-token",
                      pending_email_to="recipient@example.test", pending_email_subject="Subject",
                      pending_email_body="Another body")
    assert client.post(URL, headers=HEADERS, json=payload).status_code == 409
    assert len(runs.list("thread-a")) == len(scheduled) == 1


def test_full_valid_email_handles_json_escaping_without_truncation(approval):
    client, runs, _ = approval
    body = "\t" * (64 * 1024)
    state._set_status("thread-a", "awaiting_approval", pending_email_token="original-token",
                      pending_email_to="recipient@example.test", pending_email_subject="Subject",
                      pending_email_body=body)
    assert _proposal(client)["action"]["args"]["body"] == body
    response = client.post(URL, headers=HEADERS, json={
        "kind": "send_email", "token": _proposal(client)["token"], "decision": "edit",
        "to": "edited@example.test", "subject": "Edited", "body": body})
    assert response.status_code == 200
    assert runs.get("thread-a", response.json()["run_id"]).resume_decision["edited_action"]["args"]["body"] == body


def test_gmail_preview_and_reject_only_error_share_owner_contract(approval, monkeypatch):
    client, _, scheduled = approval
    state._set_status("thread-a", "awaiting_approval", pending_gmail_token="gmail-token-12345",
                      pending_gmail_action={"name": "gmail_delete", "args": {"message_ids": ["abc"]}},
                      pending_gmail_messages=[], pending_gmail_error="Complete body unavailable")
    proposal = _proposal(client)
    assert proposal["error"] == "Complete body unavailable"
    assert proposal["kind"] == "gmail_delete"
    assert proposal["action"]["args"]["message_ids"] == ["abc"]
    calls = []

    def core(tid, decision, token):
        calls.append((tid, decision, token))
        return type("Run", (), {"id": "resume-run", "status": "pending"})(), False

    monkeypatch.setattr(threads, "gmail_decision_core", core, raising=False)
    assert client.post(URL, headers=HEADERS, json={
        "kind": "gmail_delete", "token": proposal["token"], "decision": "reject"}).status_code == 200
    assert calls == [("thread-a", "reject", "gmail-token-12345")]
    assert scheduled == [("resume-run", "thread-a")]


def test_approval_input_is_strict_and_bounded(approval):
    client, runs, _ = approval
    payload = {"kind": "send_email", "token": _proposal(client)["token"], "decision": "approve"}
    assert client.post(URL, headers=HEADERS, json={**payload, "body": "Changed"}).status_code == 422
    assert client.post(URL, headers=HEADERS, json={**payload, "from": "attacker@example.test"}).status_code == 422
    assert client.post(URL, headers=HEADERS, json={**payload, "token": "é"}).status_code == 422
    assert client.post(URL, headers=HEADERS, content=b" " * (phone_api.MAX_APPROVAL_BYTES + 1)).status_code == 413
    assert runs.list("thread-a") == []


def test_retry_dispatches_a_pending_resume_after_status_write_failure(approval, monkeypatch):
    client, runs, scheduled = approval
    payload = {"kind": "send_email", "token": _proposal(client)["token"], "decision": "approve"}
    original = threads._set_status
    with monkeypatch.context() as patch:
        patch.setattr(threads, "_set_status", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
        assert client.post(URL, headers=HEADERS, json=payload).status_code == 500
    assert len(runs.list("thread-a")) == 1 and scheduled == []
    response = client.post(URL, headers=HEADERS, json=payload)
    assert response.status_code == 200
    assert response.json()["replayed"] is True
    assert len(runs.list("thread-a")) == 1 and len(scheduled) == 1
    assert threads._set_status is original


@pytest.mark.parametrize("decision", [
    {"type": "approve"},
    {"type": "reject", "message": "Declined"},
    {"type": "edit", "edited_action": {"name": "send_email", "args": {
        "to": "edited@example.test", "subject": "Edited", "body": "New body"}}},
])
def test_real_hitl_resumes_only_the_displayed_email(decision):
    sent = []

    def send_email(to: str, subject: str, body: str) -> str:
        """Synthetic email sink for an isolated real-framework HITL test."""
        sent.append({"to": to, "subject": subject, "body": body})
        return "Synthetic delivery recorded"

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    first = {"to": "first@example.test", "subject": "First", "body": "First body"}
    second = {"to": "unseen@example.test", "subject": "Second", "body": "Unseen body"}
    model = Model(responses=[AIMessage(content="", tool_calls=[
        {"name": "send_email", "args": first, "id": "email-1"},
        {"name": "send_email", "args": second, "id": "email-2"}]), AIMessage(content="done")])
    agent = create_agent(model, tools=[send_email], checkpointer=InMemorySaver(),
                         middleware=[HumanInTheLoopMiddleware(interrupt_on={
                             "send_email": {"allowed_decisions": ["approve", "reject", "edit"]}})])
    config = {"configurable": {"thread_id": "isolated-email"}}
    agent.invoke({"messages": [{"role": "user", "content": "Email both people"}]}, config)
    chat = Thread.__new__(Thread)
    chat.agent, chat.runconfig, chat.thread_id = agent, config, "isolated-email"
    chat._run = lambda command: agent.invoke(command, config)
    assert chat.pending_email() == first
    threads._resume_email(chat, decision)
    expected = ([] if decision["type"] == "reject" else
                [decision["edited_action"]["args"] if decision["type"] == "edit" else first])
    assert sent == expected
    assert chat.pending_actions() == []
