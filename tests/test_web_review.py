"""Tests for the diff polish + review page in ``manage/web.py``.

The web app is otherwise tested behaviorally; these tests cover the
small pure-functions added for the review feature so we can refactor
them without breaking the contract:

- ``render_file_diff`` (per-file HTML + inline cap + binary handling)
- ``_format_review_message`` (markdown formatter for submit payloads)
- diff-line classification helpers
- ``POST /thread/{tid}/review`` endpoint contract
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from assist.domain_manager import Change
from manage import web
from manage.web import (
    INLINE_FILE_LINE_CAP,
    _classify_diff_line,
    _diff_stats,
    _format_review_message,
    _is_binary_diff,
    _rename_pair,
    render_file_diff,
)


# --- Helpers ------------------------------------------------------------

def _basic_diff() -> str:
    return (
        "diff --git a/foo.py b/foo.py\n"
        "index abc..def 100644\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,3 +1,4 @@\n"
        " context\n"
        "-deleted\n"
        "+added\n"
        "+more added\n"
        " trailing\n"
    )


# --- Classification helpers ---------------------------------------------

class TestClassification:
    def test_meta_lines(self):
        for line in [
            "diff --git a/x b/x",
            "index abc..def 100644",
            "--- a/x",
            "+++ b/x",
            "rename from old",
            "rename to new",
            "similarity index 95%",
            "new file mode 100644",
            "deleted file mode 100644",
            "\\ No newline at end of file",
        ]:
            assert _classify_diff_line(line) == "meta", line

    def test_hunk(self):
        assert _classify_diff_line("@@ -1,3 +1,4 @@") == "hunk"

    def test_add_del_ctx(self):
        assert _classify_diff_line("+content") == "add"
        assert _classify_diff_line("-content") == "del"
        assert _classify_diff_line(" content") == "ctx"
        assert _classify_diff_line("") == "ctx"

    def test_meta_takes_precedence_over_plus_minus(self):
        # +++/--- start with + or - but must be classified as meta first.
        assert _classify_diff_line("+++ b/x") == "meta"
        assert _classify_diff_line("--- a/x") == "meta"


class TestStats:
    def test_counts_only_content_changes(self):
        diff = _basic_diff()
        # 2 added (+added, +more added), 1 deleted (-deleted).
        # +++/--- headers must NOT be counted.
        assert _diff_stats(diff) == (2, 1)

    def test_zero_for_pure_context(self):
        assert _diff_stats(" ctx\n ctx2\n") == (0, 0)


class TestBinary:
    def test_detects(self):
        diff = (
            "diff --git a/img.png b/img.png\n"
            "Binary files a/img.png and b/img.png differ\n"
        )
        assert _is_binary_diff(diff) is True

    def test_text_diff_not_binary(self):
        assert _is_binary_diff(_basic_diff()) is False


class TestRenamePair:
    def test_detects_rename(self):
        diff = (
            "diff --git a/old.py b/new.py\n"
            "similarity index 95%\n"
            "rename from old.py\n"
            "rename to new.py\n"
        )
        assert _rename_pair(diff) == ("old.py", "new.py")

    def test_returns_none_when_no_rename(self):
        assert _rename_pair(_basic_diff()) is None


# --- render_file_diff ---------------------------------------------------

class TestRenderFileDiff:
    def test_full_emits_data_attributes_for_clickable_rows(self):
        c = Change(path="foo.py", diff=_basic_diff())
        out = render_file_diff(c, 0, full=True)

        # Hunk header is row 5 (1-based: diff--git, index, ---, +++, @@).
        assert 'data-key="foo.py::5"' in out
        assert 'data-file="foo.py"' in out
        assert 'data-row="5"' in out
        # Context, add, del rows clickable too.
        assert 'data-key="foo.py::6"' in out  # " context"
        assert 'data-key="foo.py::8"' in out  # "+added"

    def test_full_does_not_clickify_meta_lines(self):
        c = Change(path="foo.py", diff=_basic_diff())
        out = render_file_diff(c, 0, full=True)
        # File headers shouldn't carry data-key — they're not user-comment-worthy.
        assert 'class="diff-row diff-meta">diff --git a/foo.py b/foo.py</div>' in out
        assert 'data-key="foo.py::1"' not in out  # row 1 is "diff --git ..."

    def test_full_classes_for_each_kind(self):
        c = Change(path="foo.py", diff=_basic_diff())
        out = render_file_diff(c, 0, full=True)
        assert "diff-row diff-add" in out
        assert "diff-row diff-del" in out
        assert "diff-row diff-ctx" in out
        assert "diff-row diff-hunk" in out
        assert "diff-row diff-meta" in out

    def test_inline_omits_data_attributes(self):
        c = Change(path="foo.py", diff=_basic_diff())
        out = render_file_diff(c, 0, full=False)
        # No clickable affordance on the thread page.
        assert "data-key=" not in out
        assert "data-row=" not in out
        # But CSS classes remain so the colors come through.
        assert "diff-row diff-add" in out

    def test_inline_truncates_oversized_file(self):
        # Build a diff larger than INLINE_FILE_LINE_CAP.
        body = "+x\n" * (INLINE_FILE_LINE_CAP + 50)
        diff = (
            "diff --git a/big.py b/big.py\n"
            "--- a/big.py\n"
            "+++ b/big.py\n"
            "@@ -0,0 +1," + str(INLINE_FILE_LINE_CAP + 50) + " @@\n"
            + body
        )
        c = Change(path="big.py", diff=diff)
        out = render_file_diff(c, 3, full=False, tid="thr-1")
        assert "Diff too large to preview" in out
        # Rows themselves must not be inlined or we defeat the purpose.
        assert "diff-row diff-add" not in out
        # The placeholder links to the review page anchored at this file id.
        assert "/thread/thr-1/review#file-3" in out

    def test_inline_does_not_substitute_tid_inside_diff_content(self):
        # Regression: a diff line containing the literal "{tid}" must
        # not be replaced by the thread id.  An earlier draft ran a
        # post-hoc str.replace over the rendered HTML and corrupted
        # content matching the placeholder token.
        diff = (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -1 +1 @@\n"
            "+url = f\"/thread/{tid}/review\"\n"
        )
        c = Change(path="foo.py", diff=diff)
        out = render_file_diff(c, 0, full=False, tid="abc-123")
        assert "{tid}" in out
        assert "abc-123" not in out

    def test_full_renders_oversized_file(self):
        body = "+x\n" * (INLINE_FILE_LINE_CAP + 50)
        diff = (
            "diff --git a/big.py b/big.py\n"
            "--- a/big.py\n"
            "+++ b/big.py\n"
            "@@ -0,0 +1," + str(INLINE_FILE_LINE_CAP + 50) + " @@\n"
            + body
        )
        c = Change(path="big.py", diff=diff)
        out = render_file_diff(c, 3, full=True)
        # Review page never truncates — that's the whole point.
        assert "Diff too large to preview" not in out
        assert "diff-row diff-add" in out

    def test_binary_file_renders_placeholder_with_no_clickable_rows(self):
        c = Change(
            path="img.png",
            diff="diff --git a/img.png b/img.png\n"
                 "Binary files a/img.png and b/img.png differ\n",
        )
        out = render_file_diff(c, 0, full=True)
        assert "Binary file" in out
        assert "data-key=" not in out
        assert "binary" in out  # the badge


# --- _format_review_message ---------------------------------------------

class TestFormatReviewMessage:
    def test_overall_only(self):
        out = _format_review_message("Looks good!", [], [])
        assert out.startswith("## Change review\n")
        assert "I've reviewed the changes and have some comments." in out
        assert "Looks good!" in out
        assert "### Per-line comments" not in out

    def test_lines_only(self):
        comments = [
            {"file": "foo.py", "row": 7, "lineText": "+def hello():", "comment": "rename"},
        ]
        out = _format_review_message("", comments, [])
        assert "## Change review" in out
        assert "I've reviewed the changes and have some comments." in out
        assert "### Per-line comments" in out
        assert "**`foo.py`** at diff line 7:" in out
        assert "+def hello():" in out
        assert "rename" in out

    def test_opener_precedes_overall_and_line_section(self):
        comments = [{"file": "x.py", "row": 1, "lineText": "+x", "comment": "c"}]
        out = _format_review_message("My overall thoughts.", comments, [])
        idx_opener = out.find("I've reviewed the changes")
        idx_overall = out.find("My overall thoughts.")
        idx_section = out.find("### Per-line comments")
        # Opener appears once and sits between header and overall comment.
        assert out.count("I've reviewed the changes") == 1
        assert 0 < idx_opener < idx_overall < idx_section

    def test_overall_and_lines(self):
        comments = [
            {"file": "foo.py", "row": 7, "lineText": "+x", "comment": "first"},
            {"file": "bar.py", "row": 12, "lineText": "-y", "comment": "second"},
        ]
        out = _format_review_message("Top-level", comments, [])
        # Both per-line entries present, separated by ---.
        assert out.count("**`") == 2
        assert "\n---\n" in out
        # Overall sits before the per-line section.
        idx_overall = out.find("Top-level")
        idx_section = out.find("### Per-line comments")
        assert 0 < idx_overall < idx_section

    def test_renamed_file_shows_old_path(self):
        comments = [{"file": "new.py", "row": 1, "lineText": "+x", "comment": "ok"}]
        changes = [
            Change(
                path="new.py",
                diff=(
                    "diff --git a/old.py b/new.py\n"
                    "rename from old.py\n"
                    "rename to new.py\n"
                ),
            ),
        ]
        out = _format_review_message("", comments, changes)
        assert "**`new.py` (renamed from `old.py`)** at diff line 1:" in out

    def test_strips_blank_comments_and_raises_when_only_those(self):
        comments = [{"file": "x.py", "row": 1, "lineText": "+x", "comment": "  "}]
        with pytest.raises(ValueError):
            _format_review_message("", comments, [])

    def test_raises_on_fully_empty(self):
        with pytest.raises(ValueError):
            _format_review_message("", [], [])
        with pytest.raises(ValueError):
            _format_review_message("   ", [], [])

    def test_handles_missing_keys_gracefully(self):
        # Defensive: malformed payload from a stale browser shouldn't 500.
        comments = [{"comment": "still ok"}]
        out = _format_review_message("", comments, [])
        assert "still ok" in out


# --- POST /thread/{tid}/review route ------------------------------------

class TestPostReviewRoute:
    """End-to-end checks on the POST endpoint.

    The route validates the JSON payload, formats the message via
    ``_format_review_message``, and schedules ``_process_message`` as a
    background task.  We stub out the heavy bits (``_get_domain_manager``,
    ``_process_message``) and exercise the FastAPI handler directly.
    """

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        """Point the manager at a tmp dir and create a thread directory.

        ``MANAGER`` is the module-level singleton; it caches DB handles,
        so we patch ``thread_dir`` / ``root_dir`` instead of swapping it
        out wholesale.
        """
        tdir = tmp_path / "thread-1"
        tdir.mkdir()
        monkeypatch.setattr(web.MANAGER, "root_dir", str(tmp_path))
        monkeypatch.setattr(web.MANAGER, "thread_dir", lambda tid: str(tmp_path / tid))
        return TestClient(web.app)

    def test_404_when_thread_dir_missing(self, client):
        r = client.post("/thread/does-not-exist/review",
                        data={"payload": json.dumps({"overall": "x", "lines": []})})
        assert r.status_code == 404

    def test_400_on_malformed_json(self, client):
        r = client.post("/thread/thread-1/review",
                        data={"payload": "not-json"})
        assert r.status_code == 400

    def test_400_on_empty_payload(self, client):
        r = client.post("/thread/thread-1/review",
                        data={"payload": json.dumps({"overall": "", "lines": []})})
        assert r.status_code == 400

    def test_303_and_schedules_background_task(self, client, monkeypatch):
        scheduled: list[tuple[str, str]] = []

        def fake_process(tid, text):
            scheduled.append((tid, text))

        monkeypatch.setattr(web, "_process_message", fake_process)
        monkeypatch.setattr(web, "_get_domain_manager", lambda tid: None)

        payload = {
            "overall": "Looks good",
            "lines": [{"file": "foo.py", "row": 1, "lineText": "+x", "comment": "ok"}],
        }
        r = client.post(
            "/thread/thread-1/review",
            data={"payload": json.dumps(payload)},
            follow_redirects=False,
        )
        assert r.status_code == 303
        assert r.headers["location"] == "/thread/thread-1?reviewed=1"
        assert len(scheduled) == 1
        assert scheduled[0][0] == "thread-1"
        assert scheduled[0][1].startswith("## Change review")
        assert "Looks good" in scheduled[0][1]
        assert "+x" in scheduled[0][1]
