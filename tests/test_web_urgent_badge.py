"""Tests for the notify tool + the in-app "urgent" thread-list pill.

No model/GPU: the notify tool is invoked directly (thread_id stubbed), urgent state
is driven via state helpers, and render_index/get_thread are called directly.
See docs/2026-07-04-notify-tool.org.
"""
import asyncio
import os

import pytest
import requests

from manage import web
from manage.web import state
from manage.web.threads import render_index, get_thread
from assist.events import notify as notify_mod


@pytest.fixture
def threads_root(tmp_path, monkeypatch):
    monkeypatch.setattr(web.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr("manage.web.threads._has_unmerged_changes", lambda tid: False)
    state.DESCRIPTION_CACHE.clear()
    state._UNSEEN.clear()
    state._URGENT.clear()
    yield tmp_path
    state.DESCRIPTION_CACHE.clear()
    state._UNSEEN.clear()
    state._URGENT.clear()


def _make_thread(root, tid, title="A thread"):
    os.makedirs(root / tid, exist_ok=True)
    state.DESCRIPTION_CACHE[tid] = title


def _notify_tool():
    return notify_mod.notify_tools(state._mark_urgent)[0]


class TestNotifyTool:
    def test_marks_urgent_and_sends_message_with_thread_link(self, threads_root, monkeypatch):
        _make_thread(threads_root, "t1")
        monkeypatch.setattr(notify_mod, "_thread_id", lambda: "t1")
        monkeypatch.setenv("URGENT_SMS_RECIPIENT", "+15555550100")
        monkeypatch.setenv("URGENT_SMS_THREAD_URL_BASE", "https://web.example.test:5050/")
        monkeypatch.setenv("ASSIST_SMS_OUTBOUND_URL", "http://phone.example.test/outbound/sms")
        monkeypatch.setenv("ASSIST_SMS_SECRET", "test-secret")
        sent = {}
        monkeypatch.setattr(
            "assist.events.reply.requests.post",
            lambda url, **kwargs: sent.update(url=url, **kwargs) or type("R", (), {"status_code": 200})())

        out = _notify_tool()("Reply to the landlord by 5pm")

        assert out.endswith("SMS sent.")
        assert state._has_urgent("t1")
        assert os.path.isfile(state._urgent_path("t1"))
        assert sent["json"] == {
            "to": "+15555550100",
            "text": "Reply to the landlord by 5pm\nhttps://web.example.test:5050/thread/t1",
        }
        assert sent["allow_redirects"] is False

    def test_marks_urgent_without_sending_when_recipient_is_unset(
            self, threads_root, monkeypatch, caplog):
        _make_thread(threads_root, "t1")
        monkeypatch.setattr(notify_mod, "_thread_id", lambda: "t1")
        monkeypatch.delenv("URGENT_SMS_RECIPIENT", raising=False)
        called = []
        monkeypatch.setattr("assist.events.reply.requests.post", lambda *args, **kwargs: called.append(1))

        out = _notify_tool()("Reply to the landlord by 5pm")

        assert "recipient isn't configured" in out
        assert state._has_urgent("t1")
        assert called == []
        assert "URGENT_SMS_RECIPIENT is not configured" in caplog.text

    def test_empty_message_marks_urgent_without_sending(self, threads_root, monkeypatch):
        _make_thread(threads_root, "t1")
        monkeypatch.setattr(notify_mod, "_thread_id", lambda: "t1")
        monkeypatch.setenv("URGENT_SMS_RECIPIENT", "+15555550100")
        called = []
        monkeypatch.setattr("assist.events.reply.requests.post", lambda *args, **kwargs: called.append(1))

        out = _notify_tool()("  ")

        assert "message is empty" in out
        assert state._has_urgent("t1")
        assert called == []

    def test_marks_urgent_when_sms_delivery_fails(self, threads_root, monkeypatch):
        _make_thread(threads_root, "t1")
        monkeypatch.setattr(notify_mod, "_thread_id", lambda: "t1")
        monkeypatch.setenv("URGENT_SMS_RECIPIENT", "+15555550100")
        monkeypatch.setenv("URGENT_SMS_THREAD_URL_BASE", "https://web.example.test:5050")
        monkeypatch.setenv("ASSIST_SMS_OUTBOUND_URL", "http://phone.example.test/outbound/sms")
        monkeypatch.setenv("ASSIST_SMS_SECRET", "test-secret")
        monkeypatch.setattr(
            "assist.events.reply.requests.post",
            lambda *args, **kwargs: type("R", (), {"status_code": 503})())

        out = _notify_tool()("Reply to the landlord by 5pm")

        assert "SMS not sent" in out
        assert state._has_urgent("t1")

    def test_sms_exception_does_not_expose_endpoint(self, threads_root, monkeypatch):
        _make_thread(threads_root, "t1")
        monkeypatch.setattr(notify_mod, "_thread_id", lambda: "t1")
        monkeypatch.setenv("URGENT_SMS_RECIPIENT", "+15555550100")
        monkeypatch.setenv("URGENT_SMS_THREAD_URL_BASE", "https://web.example.test:5050")
        monkeypatch.setenv("ASSIST_SMS_OUTBOUND_URL", "http://internal.phone.example.test/outbound/sms")
        monkeypatch.setenv("ASSIST_SMS_SECRET", "test-secret")

        def fail(*args, **kwargs):
            raise requests.ConnectionError("internal.phone.example.test failed")

        monkeypatch.setattr("assist.events.reply.requests.post", fail)

        out = _notify_tool()("Reply to the landlord by 5pm")

        assert "reach the phone" in out
        assert "internal.phone.example.test" not in out
        assert state._has_urgent("t1")

    def test_no_active_thread_returns_corrective_not_raises(self, monkeypatch):
        monkeypatch.setattr(notify_mod, "_thread_id", lambda: None)
        out = _notify_tool()("x")
        assert "no active thread" in out.lower()

    def test_notify_in_normal_tools_not_triage(self):
        # containment by construction: an untrusted SMS-triage turn never gets notify.
        from assist.thread_manager import _web_interrupt_on, _web_tools, _web_triage_tools
        names = lambda ts: [getattr(t, "__name__", getattr(t, "name", "")) for t in ts]
        assert "notify" in names(_web_tools)
        assert "notify" not in names(_web_triage_tools)
        assert "notify" not in (_web_interrupt_on or {})


class TestUrgentBadge:
    def test_pill_appears(self, threads_root):
        _make_thread(threads_root, "t1", "Coffee")
        state._mark_urgent("t1")
        assert ">urgent<" in render_index()

    def test_pill_clears_on_clear(self, threads_root):
        _make_thread(threads_root, "t1", "Coffee")
        state._mark_urgent("t1")
        assert ">urgent<" in render_index()
        state._clear_urgent("t1")
        assert ">urgent<" not in render_index()

    def test_survives_restart_via_load_cache(self, threads_root):
        _make_thread(threads_root, "t1", "Coffee")
        state._mark_urgent("t1")
        state._URGENT.clear()                 # simulate restart
        assert ">urgent<" not in render_index()
        state.load_urgent_cache()
        assert ">urgent<" in render_index()

    def test_route_clears_on_open(self, threads_root):
        _make_thread(threads_root, "t1", "Coffee")
        state._set_status("t1", "initializing")   # INIT stage -> get_thread skips the model
        state._mark_urgent("t1")
        assert state._has_urgent("t1")
        html = asyncio.run(get_thread("t1"))
        assert isinstance(html, str)
        assert not state._has_urgent("t1")

    def test_evict_drops_urgent(self, threads_root):
        _make_thread(threads_root, "t1")
        state._mark_urgent("t1")
        state._evict_caches("t1")
        assert not state._has_urgent("t1")


class TestPrecedence:
    def test_urgent_beats_new_and_unmerged(self, threads_root, monkeypatch):
        monkeypatch.setattr("manage.web.threads._has_unmerged_changes", lambda tid: True)
        _make_thread(threads_root, "t1", "Coffee")
        state._mark_unseen_response("t1")   # also "new"
        state._mark_urgent("t1")
        html = render_index()
        assert ">urgent<" in html
        assert ">new<" not in html
        assert ">unmerged<" not in html

    def test_error_beats_urgent(self, threads_root):
        _make_thread(threads_root, "t1", "Coffee")
        state._set_status("t1", "error", error="boom")
        state._mark_urgent("t1")
        html = render_index()
        assert ">error<" in html
        assert ">urgent<" not in html
