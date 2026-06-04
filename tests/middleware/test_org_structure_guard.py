"""Unit tests for OrgStructureGuardMiddleware's detection (no LLM).

Pins the args-only rule: reject a .org edit_file that adds a new heading
while anchoring on body text (no heading in old_string) — the mid-section
mis-anchor — and allow everything else.  The PASS/FAIL anchor shapes here
are the ones observed in the real eval runs.
"""
from unittest.mock import Mock

from langchain_core.messages import ToolMessage

from assist.middleware.org_structure_guard import (
    OrgStructureGuardMiddleware,
    _would_split,
    _has_heading,
)


def _req(name="edit_file", file_path="roadmap.org", old="", new=""):
    r = Mock()
    r.tool = Mock(name="tool")
    r.tool.name = name
    r.tool_call = {"name": name, "id": "tc1",
                   "args": {"file_path": file_path,
                            "old_string": old, "new_string": new}}
    return r


# --- detection unit (the load-bearing logic) ---

def test_has_heading_distinguishes_bold_from_heading():
    assert _has_heading("* Foo") is True
    assert _has_heading("** Bar\nbody") is True
    assert _has_heading("*Direction.*  Pick a real API.") is False  # bold, not heading
    assert _has_heading("*What landed first as a stopgap* (PR #118)") is False
    assert _has_heading("just body text") is False


def test_would_split_true_when_heading_anchored_on_body():
    # The reproduced failure: new section anchored on a bold/body line.
    old = "*What landed first as a stopgap* (PR #118): per-tool throttles"
    new = old + "\n\n* Moonshot\n** TODO Self-improving agent\nbody"
    assert _would_split({"old_string": old, "new_string": new}) is True


def test_would_split_true_when_toplevel_anchored_on_subheading():
    # The guard's earlier gap: a `*` heading inserted before a `**`
    # sub-heading splits the parent `*` section. old_string HAS a heading,
    # but it's deeper than the new one.
    old = "** TODO Explore a real Search API to replace the scraped DDG"
    new = "* Moonshot\n** TODO Self-improving agent\nbody\n" + old
    assert _would_split({"old_string": old, "new_string": new}) is True


def test_would_split_false_when_anchored_on_heading():
    # Correct insertion: old_string contains a same-level heading to go next to.
    old = "* Errands\nDrop off the package."
    new = "* Errands\nDrop off the package.\n* Reading\nFinish my book."
    assert _would_split({"old_string": old, "new_string": new}) is False


def test_would_split_false_for_subheading_anchored_on_heading():
    old = "** TODO Water the plants\nThe ferns especially need attention."
    new = old + "\n** TODO Review the budget\nThis quarter."
    assert _would_split({"old_string": old, "new_string": new}) is False


def test_would_split_false_for_body_only_edit():
    # No new heading added -> not our concern (typo fix etc.).
    old = "some body text here"
    new = "some corrected body text here"
    assert _would_split({"old_string": old, "new_string": new}) is False


def test_would_split_false_when_heading_already_in_old():
    # Rewriting a section, keeping its heading -> not a NEW heading.
    old = "* Foo\nold body"
    new = "* Foo\nnew body"
    assert _would_split({"old_string": old, "new_string": new}) is False


# --- middleware behavior ---

def test_rejects_mid_section_org_edit_without_running_handler():
    mw = OrgStructureGuardMiddleware()
    old = "*Direction.*  Pick a real API."
    new = old + "\n* Moonshot\nbody"
    called = {"ran": False}

    def handler(_req):
        called["ran"] = True
        return ToolMessage(content="edited", tool_call_id="tc1", name="edit_file")

    result = mw.wrap_tool_call(_req(old=old, new=new), handler)
    assert called["ran"] is False           # edit never applied
    # NOT status="error" + non-error-prefixed message, so LoopDetection does
    # not count the corrective redirect as a loop error (the model retries).
    assert result.status != "error"
    assert "split" in result.content.lower()
    from assist.middleware.loop_detection import _looks_like_error
    assert _looks_like_error(result.content) is False


def test_passes_through_correct_org_edit():
    mw = OrgStructureGuardMiddleware()
    old = "* Errands\nDrop off the package."
    new = old + "\n* Reading\nFinish my book."
    sentinel = ToolMessage(content="ok", tool_call_id="tc1", name="edit_file")
    result = mw.wrap_tool_call(_req(old=old, new=new), lambda r: sentinel)
    assert result is sentinel


def test_ignores_non_org_files():
    mw = OrgStructureGuardMiddleware()
    old = "body line"
    new = "body line\n* Heading\nx"
    sentinel = ToolMessage(content="ok", tool_call_id="tc1", name="edit_file")
    result = mw.wrap_tool_call(
        _req(file_path="notes.md", old=old, new=new), lambda r: sentinel)
    assert result is sentinel  # .md is not guarded


def test_ignores_non_edit_tools():
    mw = OrgStructureGuardMiddleware()
    sentinel = ToolMessage(content="ok", tool_call_id="tc1", name="write_file")
    result = mw.wrap_tool_call(
        _req(name="write_file", old="body", new="body\n* H\nx"), lambda r: sentinel)
    assert result is sentinel
