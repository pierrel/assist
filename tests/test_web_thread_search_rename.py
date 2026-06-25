"""Tests for thread search (index filter) and thread rename (description edit).

No model/GPU: titles are seeded into DESCRIPTION_CACHE (or description.txt is
pre-written), so neither MANAGER.get nor chat.description() is exercised.
"""
import os

import pytest
from fastapi.testclient import TestClient

from manage import web
from manage.web import state
from manage.web.threads import render_index


@pytest.fixture
def threads_root(tmp_path, monkeypatch):
    monkeypatch.setattr(web.MANAGER, "root_dir", str(tmp_path))
    # Use the REAL thread_dir (root_dir-based) so its tid validation is exercised.
    # render_index reaches _has_unmerged_changes (git I/O) for idle threads.
    monkeypatch.setattr("manage.web.threads._has_unmerged_changes", lambda tid: False)
    state.DESCRIPTION_CACHE.clear()
    yield tmp_path
    state.DESCRIPTION_CACHE.clear()


def _make_thread(root, tid, title):
    os.makedirs(root / tid, exist_ok=True)
    state.DESCRIPTION_CACHE[tid] = title  # title == cached description; no model


class TestThreadSearch:
    def test_filters_by_title_substring_case_insensitive(self, threads_root):
        _make_thread(threads_root, "t1", "Apple pie recipe")
        _make_thread(threads_root, "t2", "Banana bread")
        _make_thread(threads_root, "t3", "apple cider notes")
        html = render_index("apple")
        assert "Apple pie recipe" in html
        assert "apple cider notes" in html
        assert "Banana bread" not in html

    def test_empty_query_shows_all(self, threads_root):
        _make_thread(threads_root, "t1", "Apple pie")
        _make_thread(threads_root, "t2", "Banana bread")
        html = render_index("")
        assert "Apple pie" in html
        assert "Banana bread" in html

    def test_no_match_shows_message_and_no_rows(self, threads_root):
        _make_thread(threads_root, "t1", "Apple pie")
        html = render_index("zzzznope")
        assert "No threads match" in html
        assert "Apple pie" not in html

    def test_query_is_escaped_in_search_box(self, threads_root):
        _make_thread(threads_root, "t1", "Apple")
        html = render_index('foo<bar"')
        # The raw query must not appear unescaped (XSS via the value attribute).
        assert 'foo<bar"' not in html
        assert "foo&lt;bar" in html


class TestSetDescription:
    def test_writes_file_and_updates_cache(self, threads_root):
        os.makedirs(threads_root / "t1", exist_ok=True)
        state.set_description("t1", "My renamed thread")
        assert (threads_root / "t1" / "description.txt").read_text() == "My renamed thread"
        assert state.DESCRIPTION_CACHE["t1"] == "My renamed thread"

    def test_rename_sticks_without_regeneration(self, threads_root, monkeypatch):
        # The load-bearing contract: once description.txt exists, the title is
        # READ from disk, never regenerated via the model.
        os.makedirs(threads_root / "t1", exist_ok=True)
        state.set_description("t1", "Sticky name")
        state.DESCRIPTION_CACHE.clear()  # force the FS path on next read

        class _Chat:
            def description(self):
                raise AssertionError("must not regenerate when description.txt exists")

        monkeypatch.setattr(web.MANAGER, "get", lambda tid: _Chat())
        assert state.get_cached_description("t1") == "Sticky name"


class TestRenameRoute:
    def test_writes_and_redirects(self, threads_root):
        os.makedirs(threads_root / "t1", exist_ok=True)
        client = TestClient(web.app)
        r = client.post("/thread/t1/rename", data={"description": "Renamed via route"},
                        follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/thread/t1"
        assert (threads_root / "t1" / "description.txt").read_text() == "Renamed via route"

    def test_empty_rename_is_noop(self, threads_root):
        os.makedirs(threads_root / "t1", exist_ok=True)
        client = TestClient(web.app)
        r = client.post("/thread/t1/rename", data={"description": "   "},
                        follow_redirects=False)
        assert r.status_code == 303
        assert not (threads_root / "t1" / "description.txt").exists()

    def test_truncates_overlong_name(self, threads_root):
        os.makedirs(threads_root / "t1", exist_ok=True)
        client = TestClient(web.app)
        client.post("/thread/t1/rename", data={"description": "x" * 200},
                    follow_redirects=False)
        assert len((threads_root / "t1" / "description.txt").read_text()) == 120

    def test_missing_thread_is_404(self, threads_root):
        client = TestClient(web.app)
        r = client.post("/thread/nope/rename", data={"description": "x"},
                        follow_redirects=False)
        assert r.status_code == 404


class TestTidValidation:
    """tid traversal is rejected BY CONSTRUCTION at the source (ThreadManager),
    so every tid-based route is covered, not just rename/delete."""

    def test_thread_dir_accepts_valid(self, threads_root):
        assert web.MANAGER.thread_dir("good") == os.path.join(str(threads_root), "good")

    @pytest.mark.parametrize("bad", ["..", ".", "", "../etc", "a/b", "a\\b", "x\x00y"])
    def test_thread_dir_rejects_traversal(self, threads_root, bad):
        from assist.thread_manager import InvalidThreadId
        with pytest.raises(InvalidThreadId):
            web.MANAGER.thread_dir(bad)

    def test_existing_thread_dir_404_for_missing(self, threads_root):
        from fastapi import HTTPException
        from manage.web.threads import _existing_thread_dir
        with pytest.raises(HTTPException) as ei:
            _existing_thread_dir("does-not-exist")
        assert ei.value.status_code == 404

    @pytest.mark.parametrize("method,path,data", [
        ("post", "/thread/%2e%2e/rename", {"description": "pwn"}),
        ("post", "/thread/%2e%2e/message", {"text": "pwn"}),
        ("post", "/thread/%2e%2e/delete", None),
        ("get", "/thread/%2e%2e/status", None),
        ("get", "/thread/%2e%2e", None),
    ])
    def test_traversal_tid_404s_on_every_route(self, threads_root, method, path, data):
        # The InvalidThreadId handler maps a crafted tid to a clean 404 (never a
        # 500) on EVERY tid route — not just rename/delete (Copilot #143).
        client = TestClient(web.app, raise_server_exceptions=False)
        kwargs = {"follow_redirects": False}
        if data is not None:
            kwargs["data"] = data
        r = getattr(client, method)(path, **kwargs)
        assert r.status_code == 404, f"{method} {path} -> {r.status_code}"
        assert not (threads_root.parent / "description.txt").exists()


class TestRenameVisibility:
    def test_rename_control_shown_when_idle(self, threads_root):
        from manage.web.threads import render_thread
        os.makedirs(threads_root / "t1", exist_ok=True)
        state.DESCRIPTION_CACHE["t1"] = "Idle title"
        html = render_thread("t1", None)
        assert 'id="titleEdit"' in html
        assert 'onclick="showRename()"' in html

    def test_rename_control_hidden_while_busy(self, threads_root, monkeypatch):
        from manage.web.state import BUSY_STAGES
        from manage.web.threads import render_thread
        os.makedirs(threads_root / "t1", exist_ok=True)
        state.DESCRIPTION_CACHE["t1"] = "Busy title"
        busy_stage = sorted(BUSY_STAGES)[0]
        monkeypatch.setattr("manage.web.threads._get_status",
                            lambda tid: {"stage": busy_stage, "pending_message": "hi"})
        html = render_thread("t1", None)
        assert 'id="titleEdit"' not in html
        assert 'onclick="showRename()"' not in html
