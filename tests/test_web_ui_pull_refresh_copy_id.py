"""Tests for the two UI touches: pull-to-refresh (every page) + a copy-thread-id
button on the thread page.  Render the pages and assert the elements are present
(symptom-level — assert against the HTML, not a proxy).  No model/GPU.
"""
import asyncio
import os
from pathlib import Path

import pytest

from manage import web
from manage.web import state
from manage.web.threads import render_index, get_thread


@pytest.fixture
def threads_root(tmp_path, monkeypatch):
    monkeypatch.setattr(web.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr("manage.web.threads._has_unmerged_changes", lambda tid: False)
    state.DESCRIPTION_CACHE.clear()
    yield tmp_path
    state.DESCRIPTION_CACHE.clear()


def _make_thread(root, tid, title="A thread"):
    os.makedirs(root / tid, exist_ok=True)
    state.DESCRIPTION_CACHE[tid] = title


class TestPullToRefresh:
    def test_index_has_pull_to_refresh(self, threads_root):
        html = render_index()
        assert "pull to refresh" in html and "location.reload()" in html

    def test_thread_page_has_pull_to_refresh(self, threads_root):
        _make_thread(threads_root, "t1")
        state._set_status("t1", "initializing")   # INIT → get_thread skips the model
        html = asyncio.run(get_thread("t1"))
        assert "pull to refresh" in html and "location.reload()" in html


class TestPullToRefreshEveryPage:
    """Pierre asked for it on every page — cover the other navigable full pages."""

    def test_schedules_page_has_it(self):
        from manage.web.schedules import _render
        assert "pull to refresh" in _render([])

    def test_evals_index_has_it(self):
        from manage.web.evals import render_evals
        assert "pull to refresh" in render_evals()   # no-results page

    def test_evals_and_review_modules_inject_it(self):
        # structural guard: the main-table (evals) + review pages inject the shared
        # script (they need data/a tid to render, so assert the source wiring).  Count
        # the braced INJECTION forms, not the bare identifier — so the import line and
        # any comment mention don't inflate the count.
        import manage.web.evals as e, manage.web.review as r
        assert Path(e.__file__).read_text(encoding="utf-8").count("{_PULL_TO_REFRESH_SCRIPT}") == 3   # 3 templates
        assert Path(r.__file__).read_text(encoding="utf-8").count("{_threads._PULL_TO_REFRESH_SCRIPT}") == 2   # 2 pages


class TestCopyThreadId:
    def test_thread_page_has_copy_id_button(self, threads_root):
        _make_thread(threads_root, "20260707000000-abcd1234")
        state._set_status("20260707000000-abcd1234", "initializing")
        html = asyncio.run(get_thread("20260707000000-abcd1234"))
        # the button + its handler, and the actual tid copied
        assert "Copy ID" in html and "copyThreadId" in html
        assert "20260707000000-abcd1234" in html   # the id the button copies
        assert "navigator.clipboard" in html   # the copy path

    def test_index_has_no_copy_id(self, threads_root):
        # copy-id is thread-scoped — not on the list page
        assert "copyThreadId" not in render_index()
