"""Unit tests for the generic FileEditGuardMiddleware (no LLM).

Tests the generic plumbing with a fake validator + an integration check
with the real OrgHeadingInsertionValidator.
"""
from unittest.mock import Mock

from langchain_core.messages import ToolMessage

from assist.middleware.edit_guard import (
    FileEditGuardMiddleware,
    EditValidator,
)
from assist.middleware.org_structure_guard import OrgHeadingInsertionValidator


def _req(name="edit_file", file_path="notes.txt", old="", new=""):
    r = Mock()
    r.tool = Mock()
    r.tool.name = name
    r.tool_call = {"name": name, "id": "tc1",
                   "args": {"file_path": file_path,
                            "old_string": old, "new_string": new}}
    return r


class _FlagValidator(EditValidator):
    """Flags any edit to a .flag file with a fixed message."""
    def applies(self, file_path):
        return file_path.endswith(".flag")
    def check(self, old_string, new_string):
        return "fix it" if "bad" in new_string else None


def test_rejects_when_validator_flags_without_running_handler():
    mw = FileEditGuardMiddleware([_FlagValidator()])
    ran = {"x": False}
    def handler(_):
        ran["x"] = True
        return ToolMessage(content="edited", tool_call_id="tc1", name="edit_file")
    result = mw.wrap_tool_call(_req(file_path="a.flag", new="bad content"), handler)
    assert ran["x"] is False
    assert result.status == "success"      # not an error -> LoopDetection ignores it
    assert result.content == "fix it"


def test_passes_through_when_validator_allows():
    mw = FileEditGuardMiddleware([_FlagValidator()])
    sentinel = ToolMessage(content="ok", tool_call_id="tc1", name="edit_file")
    result = mw.wrap_tool_call(_req(file_path="a.flag", new="good"), lambda r: sentinel)
    assert result is sentinel


def test_ignores_files_no_validator_applies_to():
    mw = FileEditGuardMiddleware([_FlagValidator()])
    sentinel = ToolMessage(content="ok", tool_call_id="tc1", name="edit_file")
    result = mw.wrap_tool_call(_req(file_path="a.txt", new="bad"), lambda r: sentinel)
    assert result is sentinel             # .txt not matched by the .flag validator


def test_ignores_non_edit_tools():
    mw = FileEditGuardMiddleware([_FlagValidator()])
    sentinel = ToolMessage(content="ok", tool_call_id="tc1", name="write_file")
    result = mw.wrap_tool_call(
        _req(name="write_file", file_path="a.flag", new="bad"), lambda r: sentinel)
    assert result is sentinel


def test_multiple_validators_first_match_wins():
    class _Other(EditValidator):
        def applies(self, fp): return True
        def check(self, o, n): return "other"
    mw = FileEditGuardMiddleware([_FlagValidator(), _Other()])
    # .flag + "bad" -> first validator flags
    r1 = mw.wrap_tool_call(_req(file_path="a.flag", new="bad"), lambda r: None)
    assert r1.content == "fix it"
    # .txt -> flag validator skips, _Other flags
    r2 = mw.wrap_tool_call(_req(file_path="a.txt", new="x"), lambda r: None)
    assert r2.content == "other"


# --- integration with the real org validator ---

def test_org_validator_integration_rejects_mid_section():
    mw = FileEditGuardMiddleware([OrgHeadingInsertionValidator()])
    old = "*Direction.*  Pick a real API."
    ran = {"x": False}
    def handler(_):
        ran["x"] = True
        return ToolMessage(content="edited", tool_call_id="tc1", name="edit_file")
    result = mw.wrap_tool_call(
        _req(file_path="roadmap.org", old=old, new=old + "\n* Moonshot\nbody"), handler)
    assert ran["x"] is False
    assert result.status == "success"
    assert "split" in result.content.lower()


def test_org_validator_integration_allows_correct_edit():
    mw = FileEditGuardMiddleware([OrgHeadingInsertionValidator()])
    old = "* Errands\nDrop off the package."
    sentinel = ToolMessage(content="ok", tool_call_id="tc1", name="edit_file")
    result = mw.wrap_tool_call(
        _req(file_path="roadmap.org", old=old, new=old + "\n* Reading\nx"),
        lambda r: sentinel)
    assert result is sentinel
