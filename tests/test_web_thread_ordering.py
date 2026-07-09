"""Thread-list ordering: STATUS band first (urgent > processing > queued > new > else),
then last-message age (newest first) within a band. Merge status has no bearing.
See manage.web.threads._thread_status_rank. Assertions key on the /thread/<tid> href,
which is always present regardless of the busy-stage title placeholder."""
import os

import pytest

from manage import web
from manage.web import state
from manage.web.threads import render_index


@pytest.fixture
def threads_root(tmp_path, monkeypatch):
    monkeypatch.setattr(web.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr("manage.web.threads._has_unmerged_changes", lambda tid: False)
    state.DESCRIPTION_CACHE.clear(); state._UNSEEN.clear(); state._URGENT.clear()
    yield tmp_path
    state.DESCRIPTION_CACHE.clear(); state._UNSEEN.clear(); state._URGENT.clear()


def _mk(root, tid, title="A thread"):
    os.makedirs(root / tid, exist_ok=True)
    state.DESCRIPTION_CACHE[tid] = title


def _positions(html, tids):
    return [html.index(f"/thread/{tid}") for tid in tids]


class TestThreadListOrdering:
    def test_status_band_order(self, threads_root):
        # Created in a deliberately NON-status order so mtime alone wouldn't sort them right.
        _mk(threads_root, "t_else")
        _mk(threads_root, "t_urgent"); state._URGENT.add("t_urgent")
        _mk(threads_root, "t_new"); state._UNSEEN.add("t_new")
        _mk(threads_root, "t_proc"); state._set_status("t_proc", "processing")
        _mk(threads_root, "t_queued"); state._set_status("t_queued", "queued")
        html = render_index("")
        pos = _positions(html, ["t_urgent", "t_proc", "t_queued", "t_new", "t_else"])
        assert pos == sorted(pos), f"not in status order: {pos}"

    def test_age_tiebreak_within_band(self, threads_root):
        _mk(threads_root, "t_old"); _mk(threads_root, "t_new2")
        os.utime(threads_root / "t_old", (1000, 1000))
        os.utime(threads_root / "t_new2", (2000, 2000))  # newer -> first within the else band
        html = render_index("")
        assert html.index("/thread/t_new2") < html.index("/thread/t_old")

    def test_merge_status_has_no_bearing(self, threads_root, monkeypatch):
        monkeypatch.setattr("manage.web.threads._has_unmerged_changes",
                            lambda tid: tid == "t_unmerged")
        _mk(threads_root, "t_unmerged"); _mk(threads_root, "t_urgent")
        state._URGENT.add("t_urgent")
        os.utime(threads_root / "t_unmerged", (2000, 2000))  # newer, but else band
        os.utime(threads_root / "t_urgent", (1000, 1000))    # older, but urgent -> still first
        html = render_index("")
        assert html.index("/thread/t_urgent") < html.index("/thread/t_unmerged")
