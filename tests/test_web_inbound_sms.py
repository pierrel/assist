"""Route tests for inbound-SMS + reply-approval: auth, dedup, dispatch queued, gating.

The route + dispatch are exercised for real (auth, the durable claim, BackgroundTask
scheduling); the triage turn itself (_dispatch_event → _process_message) is stubbed to a
spy so the LLM/sandbox isn't needed.
"""
import pytest
from fastapi.testclient import TestClient

from manage import web
from manage.web import threads
from manage.web.state import _set_status


@pytest.fixture
def client(tmp_path, monkeypatch):
    tdir = tmp_path / "t-sub"
    tdir.mkdir()
    monkeypatch.setattr(web.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(web.MANAGER, "thread_dir", lambda tid: str(tmp_path / tid))
    # Repoint the durable inbound log at the tmp root so dedup is isolated per test.
    from assist.events.inbound import InboundLog
    monkeypatch.setattr(threads, "INBOUND_LOG", InboundLog(str(tmp_path)))
    return TestClient(web.app)


def _body(mid="abc123", sender="+15551234567", text="hi"):
    return {"message_id": mid, "sender": sender, "text": text}


def test_inbound_503_when_secret_unset(client, monkeypatch):
    monkeypatch.delenv("ASSIST_SMS_SECRET", raising=False)
    assert client.post("/inbound/sms", json=_body()).status_code == 503


def test_inbound_401_bad_secret(client, monkeypatch):
    monkeypatch.setenv("ASSIST_SMS_SECRET", "s3cret")
    r = client.post("/inbound/sms", json=_body(), headers={"X-Assist-SMS-Secret": "wrong"})
    assert r.status_code == 401
    r2 = client.post("/inbound/sms", json=_body())  # missing header
    assert r2.status_code == 401


def test_inbound_accepts_and_dispatches(client, monkeypatch):
    monkeypatch.setenv("ASSIST_SMS_SECRET", "s3cret")
    calls = []
    monkeypatch.setattr(threads, "_dispatch_event", lambda sender, text: calls.append((sender, text)))
    r = client.post("/inbound/sms", json=_body(), headers={"X-Assist-SMS-Secret": "s3cret"})
    assert r.status_code == 200 and r.json()["status"] == "accepted"
    assert calls == [("+15551234567", "hi")]   # dispatched once, off the response path


def test_inbound_dedup_same_message_id(client, monkeypatch):
    monkeypatch.setenv("ASSIST_SMS_SECRET", "s3cret")
    calls = []
    monkeypatch.setattr(threads, "_dispatch_event", lambda sender, text: calls.append(1))
    h = {"X-Assist-SMS-Secret": "s3cret"}
    first = client.post("/inbound/sms", json=_body(mid="dup1"), headers=h)
    second = client.post("/inbound/sms", json=_body(mid="dup1"), headers=h)
    assert first.json()["status"] == "accepted"
    assert second.json()["status"] == "duplicate"
    assert len(calls) == 1                       # the duplicate did NOT re-dispatch


def test_inbound_400_bad_message_id(client, monkeypatch):
    monkeypatch.setenv("ASSIST_SMS_SECRET", "s3cret")
    r = client.post("/inbound/sms", json=_body(mid="../etc/passwd"),
                    headers={"X-Assist-SMS-Secret": "s3cret"})
    assert r.status_code == 400


def test_dispatch_no_matching_subscription_is_noop(client, monkeypatch):
    seen = []
    monkeypatch.setattr(threads.SUBSCRIPTION_STORE, "route", lambda sender: None)
    monkeypatch.setattr(threads, "_process_message", lambda *a, **k: seen.append(1))
    threads._dispatch_event("+1999", "hi")
    assert seen == []                            # no subscription → no turn


def test_reply_decision_gated_on_awaiting_approval(client, monkeypatch):
    monkeypatch.setattr(web.MANAGER, "get", lambda tid, **k: object())
    queued = []
    monkeypatch.setattr(threads, "_execute_run", lambda *a: queued.append(a))
    # not awaiting → 409
    _set_status("t-sub", "ready")
    assert client.post("/thread/t-sub/reply/approve").status_code == 409
    # awaiting → 303 + resume queued with an approve decision + the stored sender
    _set_status("t-sub", "awaiting_approval", pending_reply="draft", pending_sender="+1555")
    r = client.post("/thread/t-sub/reply/approve", follow_redirects=False)
    assert r.status_code == 303
    assert len(queued) == 1
    run = threads._runs().get("t-sub", queued[0][0])
    assert queued[0][1] == "t-sub"
    assert run.sender == "+1555" and run.resume_decision == {"type": "approve"}


def test_reply_decision_bad_verb(client, monkeypatch):
    monkeypatch.setattr(web.MANAGER, "get", lambda tid, **k: object())
    _set_status("t-sub", "awaiting_approval", pending_reply="d", pending_sender="+1")
    assert client.post("/thread/t-sub/reply/nonsense").status_code == 400


def test_dispatch_calls_process_message_with_rendered_template(client, monkeypatch):
    from assist.events.model import Subscription
    sub = Subscription(id="s", thread_id="t-sub", sender_regexp=".*", template="from {sender}: {text}")
    monkeypatch.setattr(threads.SUBSCRIPTION_STORE, "route", lambda sender: sub)
    calls = []
    monkeypatch.setattr(threads, "_execute_run", lambda *a: calls.append(a))
    threads._dispatch_event("+1555", "hello")
    assert len(calls) == 1                                # supersede now lives in _process_message
    run = threads._runs().get("t-sub", calls[0][0])
    assert calls[0][1] == "t-sub" and "hello" in run.text
    assert run.sender == "+1555"


def test_triage_tools_exclude_host_effect_tools():
    # The untrusted-SMS triage turn must NOT get the host-effect config tools (schedule/
    # subscription) — only the HITL-gated reply. Normal turns keep the config tools.
    from assist import thread_manager as tm
    triage = {getattr(f, "__name__", "") for f in tm._web_triage_tools}
    normal = {getattr(f, "__name__", "") for f in tm._web_tools}
    assert "send_reply" in triage
    assert "send_email" not in triage
    assert "create_subscription" not in triage and "create_schedule" not in triage
    assert "delete_subscription" not in triage
    assert "create_subscription" in normal and "send_reply" not in normal
    assert "send_email" in normal
    assert "get_location" in normal and "get_location" not in triage
    assert "send_email" in tm._web_interrupt_on
    assert "send_reply" in tm._web_triage_interrupt_on


def test_reply_approve_refuses_superseded_draft(client, monkeypatch):
    from manage.web.state import _set_status
    monkeypatch.setattr(threads, "_existing_thread_dir", lambda tid: str(tid))
    queued = []
    monkeypatch.setattr(threads, "_process_message", lambda *a, **k: queued.append(1))
    _set_status("t-sub", "awaiting_approval", pending_reply="NEW draft", pending_sender="+1")
    # user approves the OLD draft they saw → mismatch → 409, nothing queued
    r = client.post("/thread/t-sub/reply/approve", data={"seen": "OLD draft"})
    assert r.status_code == 409 and queued == []
    # approving the current draft goes through
    r2 = client.post("/thread/t-sub/reply/approve", data={"seen": "NEW draft"},
                     follow_redirects=False)
    assert r2.status_code == 303 and len(queued) == 1


def test_email_approval_requires_its_token_and_exact_review(client, monkeypatch):
    monkeypatch.setattr(web.MANAGER, "get", lambda tid, **k: object())
    queued = []
    monkeypatch.setattr(threads, "_execute_run", lambda *a: queued.append(a))
    _set_status("t-sub", "awaiting_approval", pending_email_to="a@example.test",
                pending_email_subject="Subject", pending_email_body="Body",
                pending_email_token="approval-token")

    missing = client.post("/thread/t-sub/email/approve")
    assert missing.status_code == 409
    stale = client.post("/thread/t-sub/email/approve", data={
        "token": "approval-token", "seen_to": "other@example.test",
        "seen_subject": "Subject", "seen_body": "Body"})
    assert stale.status_code == 409
    changed = client.post("/thread/t-sub/email/approve", data={
        "token": "approval-token", "to": "other@example.test", "subject": "Subject",
        "body": "Body", "seen_to": "a@example.test", "seen_subject": "Subject",
        "seen_body": "Body"})
    assert changed.status_code == 409
    approved = client.post("/thread/t-sub/email/approve", data={
        "token": "approval-token", "to": "a@example.test", "subject": "Subject",
        "body": "Body", "seen_to": "a@example.test",
        "seen_subject": "Subject", "seen_body": "Body"}, follow_redirects=False)
    assert approved.status_code == 303 and len(queued) == 1
    run = threads._runs().get("t-sub", queued[0][0])
    assert run.resume_decision == {"type": "approve"}


def test_email_edit_rewrites_only_user_editable_fields(client, monkeypatch):
    monkeypatch.setattr(web.MANAGER, "get", lambda tid, **k: object())
    queued = []
    monkeypatch.setattr(threads, "_execute_run", lambda *a: queued.append(a))
    _set_status("t-sub", "awaiting_approval", pending_email_to="a@example.test",
                pending_email_subject="Subject", pending_email_body="Body",
                pending_email_token="approval-token")

    edited = client.post("/thread/t-sub/email/edit", data={
        "token": "approval-token", "to": "b@example.test", "subject": "Edited",
        "body": "Edited body"}, follow_redirects=False)

    assert edited.status_code == 303 and len(queued) == 1
    run = threads._runs().get("t-sub", queued[0][0])
    assert run.resume_decision == {
        "type": "edit", "edited_action": {"name": "send_email", "args": {
            "to": "b@example.test", "subject": "Edited", "body": "Edited body"}}}


def test_message_post_refuses_while_email_is_awaiting_approval(client):
    _set_status("t-sub", "awaiting_approval", pending_email_token="approval-token")

    response = client.post("/thread/t-sub/message", data={"text": "new message"})

    assert response.status_code == 409


def test_email_approval_card_renders_full_message(client, monkeypatch):
    monkeypatch.setattr(
        web.MANAGER, "get", lambda tid, sandbox_backend=None, **k:
        type("C", (), {"get_messages": lambda self: [],
                        "pending_reply": lambda self: None})())
    monkeypatch.setattr("manage.web.threads.get_cached_description", lambda tid: "Thread")
    monkeypatch.setenv("EMAIL_FROM_ADDRESS", "assistant@example.test")
    monkeypatch.setenv("EMAIL_FROM_NAME", "Assistant")
    monkeypatch.setenv("EMAIL_ALWAYS_CC", "oversight@example.test")
    _set_status("t-sub", "awaiting_approval", pending_email_to="to@example.test",
                pending_email_subject="Subject", pending_email_body="A full\nmessage",
                pending_email_token="approval-token")

    page = client.get("/thread/t-sub").text

    assert "Email awaiting your approval" in page
    assert "Assistant &lt;assistant@example.test&gt;" in page
    assert "oversight@example.test" in page
    assert "A full\nmessage" in page
    assert 'name="token" value="approval-token"' in page
