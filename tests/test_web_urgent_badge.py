"""Tests for the notify tool + "urgent" badge + iOS PWA badge seam.

No model/GPU: the notify tool is invoked directly (thread_id stubbed), urgent state
is driven via state helpers, and render_index/get_thread are called directly.
See docs/2026-07-04-notify-tool.org.
"""
import asyncio
import json
import os

import pytest

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
    def test_marks_urgent(self, threads_root, monkeypatch):
        _make_thread(threads_root, "t1")
        monkeypatch.setattr(notify_mod, "_thread_id", lambda: "t1")
        out = _notify_tool()("landlord needs a reply by 5pm")
        assert "urgent" in out.lower()
        assert state._has_urgent("t1")
        assert state._any_urgent()
        assert os.path.isfile(state._urgent_path("t1"))

    def test_no_active_thread_returns_corrective_not_raises(self, monkeypatch):
        monkeypatch.setattr(notify_mod, "_thread_id", lambda: None)
        out = _notify_tool()("x")
        assert "no active thread" in out.lower()

    def test_notify_in_normal_tools_not_triage(self):
        # containment by construction: an untrusted SMS-triage turn never gets notify.
        from assist.thread_manager import _web_tools, _web_triage_tools
        names = lambda ts: [getattr(t, "__name__", getattr(t, "name", "")) for t in ts]
        assert "notify" in names(_web_tools)
        assert "notify" not in names(_web_triage_tools)


class TestUrgentBadge:
    def test_pill_appears(self, threads_root):
        _make_thread(threads_root, "t1", "Coffee")
        state._mark_urgent("t1")
        assert ">urgent<" in render_index("")

    def test_pill_clears_on_clear(self, threads_root):
        _make_thread(threads_root, "t1", "Coffee")
        state._mark_urgent("t1")
        assert ">urgent<" in render_index("")
        state._clear_urgent("t1")
        assert ">urgent<" not in render_index("")

    def test_urgent_meta_reflects_any_urgent(self, threads_root):
        _make_thread(threads_root, "t1", "Coffee")
        assert 'name="assist-urgent" content="0"' in render_index("")
        state._mark_urgent("t1")
        assert 'name="assist-urgent" content="1"' in render_index("")

    def test_survives_restart_via_load_cache(self, threads_root):
        _make_thread(threads_root, "t1", "Coffee")
        state._mark_urgent("t1")
        state._URGENT.clear()                 # simulate restart
        assert ">urgent<" not in render_index("")
        state.load_urgent_cache()
        assert ">urgent<" in render_index("")

    def test_route_clears_on_open(self, threads_root):
        _make_thread(threads_root, "t1", "Coffee")
        state._set_status("t1", "initializing")   # INIT stage -> get_thread skips the model
        state._mark_urgent("t1")
        assert state._has_urgent("t1")
        html = asyncio.run(get_thread("t1"))
        assert isinstance(html, str)
        assert not state._has_urgent("t1")
        assert not state._any_urgent()

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
        html = render_index("")
        assert ">urgent<" in html
        assert ">new<" not in html
        assert ">unmerged<" not in html

    def test_error_beats_urgent(self, threads_root):
        _make_thread(threads_root, "t1", "Coffee")
        state._set_status("t1", "error", error="boom")
        state._mark_urgent("t1")
        html = render_index("")
        assert ">error<" in html
        assert ">urgent<" not in html


class TestPWAAssets:
    def test_head_has_manifest_and_apple_metas(self, threads_root):
        html = render_index("")
        assert 'rel="manifest"' in html
        assert 'apple-touch-icon' in html
        assert 'apple-mobile-web-app-capable' in html
        assert "setAppBadge" in html   # the badge script is present

    def test_thread_page_also_syncs_badge(self, threads_root):
        # opening an urgent thread clears it server-side; the thread page must carry
        # the meta + badge script so foregrounding there re-syncs the icon dot (not
        # only on a return to /).
        _make_thread(threads_root, "t1", "Coffee")
        state._set_status("t1", "initializing")
        html = asyncio.run(get_thread("t1"))
        assert 'name="assist-urgent"' in html
        assert "setAppBadge" in html

    def test_manifest_file_is_valid_json(self):
        path = os.path.join(os.path.dirname(state.__file__), "static", "manifest.webmanifest")
        with open(path) as f:
            m = json.load(f)
        assert m["display"] == "standalone"
        assert any(i["sizes"] == "512x512" for i in m["icons"])
