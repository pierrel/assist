"""Unit tests for OrgHeadingInsertionValidator's detection (no LLM).

Pins the args-only rule: flag a .org edit that adds a new heading anchored
shallower than every heading in old_string (the mid-section mis-anchor) and
allow everything else.  The PASS/FAIL anchor shapes here are the ones
observed in the real eval runs.
"""
from assist.middleware.org_structure_guard import (
    OrgHeadingInsertionValidator,
    _would_split,
    _has_heading,
)
from assist.middleware.loop_detection import _looks_like_error


# --- detection (the load-bearing logic) ---

def test_has_heading_distinguishes_bold_from_heading():
    assert _has_heading("* Foo") is True
    assert _has_heading("** Bar\nbody") is True
    assert _has_heading("*Direction.*  Pick a real API.") is False  # bold, not heading
    assert _has_heading("*What landed first as a stopgap* (PR #118)") is False
    assert _has_heading("just body text") is False


def test_would_split_true_when_heading_anchored_on_body():
    old = "*What landed first as a stopgap* (PR #118): per-tool throttles"
    new = old + "\n\n* Moonshot\n** TODO Self-improving agent\nbody"
    assert _would_split({"old_string": old, "new_string": new}) is True


def test_would_split_true_when_toplevel_anchored_on_subheading():
    # `*` heading inserted before a `**` sub-heading splits the parent `*`.
    old = "** TODO Explore a real Search API to replace the scraped DDG"
    new = "* Moonshot\n** TODO Self-improving agent\nbody\n" + old
    assert _would_split({"old_string": old, "new_string": new}) is True


def test_would_split_false_when_anchored_on_same_level_heading():
    old = "* Errands\nDrop off the package."
    new = "* Errands\nDrop off the package.\n* Reading\nFinish my book."
    assert _would_split({"old_string": old, "new_string": new}) is False


def test_would_split_false_for_subheading_anchored_on_heading():
    old = "** TODO Water the plants\nThe ferns especially need attention."
    new = old + "\n** TODO Review the budget\nThis quarter."
    assert _would_split({"old_string": old, "new_string": new}) is False


def test_would_split_false_for_body_only_edit():
    assert _would_split({"old_string": "some body text",
                         "new_string": "some corrected body text"}) is False


def test_would_split_false_when_heading_already_in_old():
    assert _would_split({"old_string": "* Foo\nold body",
                         "new_string": "* Foo\nnew body"}) is False


# --- the validator ---

def test_validator_applies_only_to_org():
    v = OrgHeadingInsertionValidator()
    assert v.applies("roadmap.org") is True
    assert v.applies("/a/b/notes.org") is True
    assert v.applies("notes.md") is False


def test_validator_check_returns_message_on_split():
    v = OrgHeadingInsertionValidator()
    old = "*Direction.*  Pick a real API."
    msg = v.check(old, old + "\n* Moonshot\nbody")
    assert msg is not None
    assert "split" in msg.lower()
    # message must not look like an error (so LoopDetection ignores it)
    assert _looks_like_error(msg) is False


def test_validator_check_returns_none_when_ok():
    v = OrgHeadingInsertionValidator()
    old = "* Errands\nDrop off the package."
    assert v.check(old, old + "\n* Reading\nFinish my book.") is None
