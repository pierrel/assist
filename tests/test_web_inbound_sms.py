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
    monkeypatch.setattr(threads, "_process_message", lambda *a, **k: queued.append((a, k)))
    # not awaiting → 409
    _set_status("t-sub", "ready")
    assert client.post("/thread/t-sub/reply/approve").status_code == 409
    # awaiting → 303 + resume queued with an approve decision + the stored sender
    _set_status("t-sub", "awaiting_approval", pending_reply="draft", pending_sender="+1555")
    r = client.post("/thread/t-sub/reply/approve", follow_redirects=False)
    assert r.status_code == 303
    assert len(queued) == 1
    args = queued[0][0]
    assert args[0] == "t-sub" and args[3] == "+1555" and args[4] == {"type": "approve"}


def test_reply_decision_bad_verb(client, monkeypatch):
    monkeypatch.setattr(web.MANAGER, "get", lambda tid, **k: object())
    _set_status("t-sub", "awaiting_approval", pending_reply="d", pending_sender="+1")
    assert client.post("/thread/t-sub/reply/nonsense").status_code == 400
