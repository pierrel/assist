"""Tests for the "new" (unread AI response) badge on the web thread list.

No model/GPU: titles seeded into DESCRIPTION_CACHE; the unseen state is driven via
the state helpers / a status write; the route-clears test opens a thread parked in
an INIT stage so get_thread skips constructing a Thread.
See docs/2026-07-03-unread-badge.org.
"""
import asyncio
import os

import pytest

from manage import web
from manage.web import state
from manage.web.threads import render_index, get_thread


@pytest.fixture
def threads_root(tmp_path, monkeypatch):
    monkeypatch.setattr(web.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr("manage.web.threads._has_unmerged_changes", lambda tid: False)
    state.DESCRIPTION_CACHE.clear()
    state._UNSEEN.clear()
    yield tmp_path
    state.DESCRIPTION_CACHE.clear()
    state._UNSEEN.clear()


def _make_thread(root, tid, title="A thread"):
    os.makedirs(root / tid, exist_ok=True)
    state.DESCRIPTION_CACHE[tid] = title


class TestBadgeAppearsAndClears:
    def test_badge_appears_when_unseen(self, threads_root):
        _make_thread(threads_root, "t1", "Coffee thread")
        state._mark_unseen_response("t1")
        assert ">new<" in render_index("")

    def test_no_badge_when_seen(self, threads_root):
        _make_thread(threads_root, "t1", "Coffee thread")
        # never marked unseen -> no "new"
        assert ">new<" not in render_index("")

    def test_badge_clears_after_clear(self, threads_root):
        _make_thread(threads_root, "t1", "Coffee thread")
        state._mark_unseen_response("t1")
        assert ">new<" in render_index("")
        state._clear_unseen_response("t1")
        assert ">new<" not in render_index("")

    def test_marker_survives_restart_via_load_cache(self, threads_root):
        _make_thread(threads_root, "t1", "Coffee thread")
        state._mark_unseen_response("t1")     # writes the marker FILE
        state._UNSEEN.clear()                 # simulate a restart (cache lost)
        assert ">new<" not in render_index("")
        state.load_unseen_cache()             # rebuild from disk
        assert ">new<" in render_index("")

    def test_clear_removes_file_so_reload_doesnt_resurrect(self, threads_root):
        _make_thread(threads_root, "t1", "Coffee thread")
        state._mark_unseen_response("t1")
        state._clear_unseen_response("t1")     # removes set entry AND file
        state._UNSEEN.clear()
        state.load_unseen_cache()
        assert ">new<" not in render_index("")

    def test_load_cache_is_true_rebuild_drops_stale(self, threads_root):
        # a stale in-memory entry with no marker file must be dropped on reload.
        _make_thread(threads_root, "t1")
        state._UNSEEN.add("ghost")   # no marker file backs this
        state.load_unseen_cache()
        assert "ghost" not in state._UNSEEN

    def test_evict_caches_drops_unseen(self, threads_root):
        # hard-deleting a never-opened unseen thread must not leak its tid in _UNSEEN.
        _make_thread(threads_root, "t1")
        state._mark_unseen_response("t1")
        assert state._has_unseen_response("t1")
        state._evict_caches("t1")              # the hard_delete on_delete callback
        assert not state._has_unseen_response("t1")


class TestChokePoint:
    def test_set_status_ready_marks_unseen(self, threads_root):
        _make_thread(threads_root, "t1")
        state._set_status("t1", "ready")
        assert state._has_unseen_response("t1")

    def test_set_status_awaiting_approval_marks_unseen(self, threads_root):
        _make_thread(threads_root, "t1")
        state._set_status("t1", "awaiting_approval")
        assert state._has_unseen_response("t1")

    def test_set_status_error_does_not_mark(self, threads_root):
        _make_thread(threads_root, "t1")
        state._set_status("t1", "error", error="boom")
        assert not state._has_unseen_response("t1")


class TestPrecedence:
    def test_new_beats_unmerged(self, threads_root, monkeypatch):
        monkeypatch.setattr("manage.web.threads._has_unmerged_changes", lambda tid: True)
        _make_thread(threads_root, "t1", "Coffee thread")
        state._mark_unseen_response("t1")
        html = render_index("")
        assert ">new<" in html
        assert ">unmerged<" not in html

    def test_error_beats_new(self, threads_root):
        _make_thread(threads_root, "t1", "Coffee thread")
        state._set_status("t1", "error", error="boom")   # does not mark
        state._mark_unseen_response("t1")                # mark separately
        html = render_index("")
        assert ">error<" in html
        assert ">new<" not in html

    def test_busy_beats_new(self, threads_root):
        _make_thread(threads_root, "t1", "Coffee thread")
        state._set_status("t1", "processing")
        state._mark_unseen_response("t1")
        html = render_index("")
        assert ">new<" not in html   # a busy/process-state badge wins


class TestRouteClears:
    def test_opening_thread_clears_marker(self, threads_root):
        # Park the thread in an INIT stage so get_thread skips constructing a Thread
        # (no model); the _clear runs before the stage check regardless. Call the
        # handler directly (not TestClient) to avoid the app lifespan/scheduler.
        _make_thread(threads_root, "t1", "Coffee thread")
        state._set_status("t1", "initializing")
        state._mark_unseen_response("t1")
        assert state._has_unseen_response("t1")
        html = asyncio.run(get_thread("t1"))
        assert isinstance(html, str)
        assert not state._has_unseen_response("t1")   # route cleared it
