import hashlib
import json
from types import SimpleNamespace

import pytest
import requests

from assist.events import email


class _Response:
    def __init__(self, status_code=200, body=b'{"id":"message-123"}'):
        self.status_code = status_code
        self.body = body
        self.closed = False

    def iter_content(self, size):
        yield self.body

    def close(self):
        self.closed = True


@pytest.fixture
def configured_email(monkeypatch, tmp_path):
    key = tmp_path / "resend-key"
    key.write_text("re_abcdefghijklmnop")
    key.chmod(0o600)
    monkeypatch.setenv("EMAIL_RESEND_API_KEY_FILE", str(key))
    monkeypatch.setenv("EMAIL_FROM_ADDRESS", "assistant@example.test")
    monkeypatch.setenv("EMAIL_FROM_NAME", "Assistant")
    monkeypatch.setenv("EMAIL_ALWAYS_CC", "oversight@example.test")


def test_send_email_fixes_identity_and_uses_tool_call_id(configured_email, monkeypatch):
    response = _Response()
    sent = {}
    monkeypatch.setattr(email.requests, "post", lambda url, **kwargs: sent.update(
        url=url, **kwargs) or response)

    result = email.send_email(
        "recipient@example.test", "A subject", "Plain message",
        SimpleNamespace(tool_call_id="call-1"))

    assert result == "Email sent to recipient@example.test."
    assert response.closed
    assert sent["url"] == "https://api.resend.com/emails"
    assert sent["json"] == {
        "from": "Assistant <assistant@example.test>",
        "to": ["recipient@example.test"],
        "cc": ["oversight@example.test"],
        "subject": "A subject",
        "text": "Plain message",
    }
    assert sent["headers"]["Idempotency-Key"] == "assist-email/" + hashlib.sha256(
        b"call-1").hexdigest()
    assert sent["headers"]["Authorization"] == "Bearer re_abcdefghijklmnop"
    assert sent["stream"] is True and sent["timeout"] == 10


def test_send_email_fails_closed_without_complete_configuration(monkeypatch):
    monkeypatch.delenv("EMAIL_RESEND_API_KEY_FILE", raising=False)
    called = []
    monkeypatch.setattr(email.requests, "post", lambda *args, **kwargs: called.append(1))

    result = email.send_email(
        "recipient@example.test", "Subject", "Body", SimpleNamespace(tool_call_id="call-1"))

    assert result.startswith("Email not sent:")
    assert called == []


@pytest.mark.parametrize("to,subject", [
    ("recipient@example.test\nBcc: hidden@example.test", "Subject"),
    ("one@example.test, two@example.test", "Subject"),
    ("recipient@example.test", "Subject\nBcc: hidden@example.test"),
])
def test_send_email_rejects_header_injection(configured_email, monkeypatch, to, subject):
    called = []
    monkeypatch.setattr(email.requests, "post", lambda *args, **kwargs: called.append(1))

    result = email.send_email(to, subject, "Body", SimpleNamespace(tool_call_id="call-1"))

    assert result.startswith("Email not sent:")
    assert called == []


def test_send_email_does_not_retry_an_uncertain_delivery(configured_email, monkeypatch):
    calls = []

    def timeout(*args, **kwargs):
        calls.append(1)
        raise requests.Timeout("timed out")

    monkeypatch.setattr(email.requests, "post", timeout)
    result = email.send_email(
        "recipient@example.test", "Subject", "Body", SimpleNamespace(tool_call_id="call-1"))

    assert result == "Email delivery status is unknown: the provider connection failed."
    assert calls == [1]


def test_send_email_rejects_invalid_provider_response(configured_email, monkeypatch):
    response = _Response(body=json.dumps({"wrong": "shape"}).encode())
    monkeypatch.setattr(email.requests, "post", lambda *args, **kwargs: response)

    result = email.send_email(
        "recipient@example.test", "Subject", "Body", SimpleNamespace(tool_call_id="call-1"))

    assert result == "Email not sent: the delivery provider returned an invalid response."
    assert response.closed


@pytest.mark.parametrize("subject,body", [
    ("x" * 999, "Body"),
    ("Subject", "x" * (64 * 1024 + 1)),
])
def test_send_email_bounds_user_content(configured_email, monkeypatch, subject, body):
    called = []
    monkeypatch.setattr(email.requests, "post", lambda *args, **kwargs: called.append(1))

    result = email.send_email(
        "recipient@example.test", subject, body, SimpleNamespace(tool_call_id="call-1"))

    assert result.startswith("Email not sent:")
    assert called == []
