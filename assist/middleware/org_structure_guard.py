"""``OrgHeadingInsertionValidator`` — an ``EditValidator`` that stops the
small model from inserting a new org heading mid-section.

Failure mode (2026-06-03 prod threads, reproduced in
``edd/eval/test_org_insertion.py``): asked to add a section to a large
``.org`` file, the model anchors its ``edit_file`` ``old_string`` on a body
line — often an org *bold* line like ``*Direction.*`` that it mistakes for a
heading — so the new heading lands in the MIDDLE of an existing section and
splits it.  Six skill/prompt variants failed to fix this (the model ignores
the guidance on a file this size), so this is a deterministic check; see
``docs/2026-06-03-org-insertion-mid-section.org``.

Detection (purely from the edit args — no file read): the model is *adding a
heading* when ``new_string`` introduces a heading line (``*``… + a SPACE)
not in ``old_string``.  It lands mid-section when that new heading is
SHALLOWER (more top-level) than every heading in ``old_string`` — which
covers both "anchored on body text / a bold line — no heading at all" and
"a ``*`` heading anchored on a ``**`` sub-heading (splitting the parent)".

Plugged into the generic ``FileEditGuardMiddleware``; the rejection is
delivered as a non-error ``status="success"`` redirect (see ``edit_guard``)
so the model keeps re-anchoring instead of being killed by loop detection.
"""
import re

from assist.middleware.edit_guard import EditValidator

# A heading is asterisks followed by a SPACE: "* Foo", "** Bar".  An org
# *bold* line ("*Direction.*", "*Note.* text") has no space after the "*"
# and is body text, NOT a heading — this distinction is the whole bug.
_HEADING_RE = re.compile(r"^\*+ ")


def _level(line: str) -> int | None:
    """Heading depth (``* `` -> 1, ``** `` -> 2), or None if not a heading."""
    st = line.lstrip()
    if _HEADING_RE.match(st):
        return len(st) - len(st.lstrip("*"))
    return None


def _heading_levels(text: str) -> list[int]:
    return [lvl for line in text.splitlines() if (lvl := _level(line)) is not None]


def _has_heading(text: str) -> bool:
    return bool(_heading_levels(text))


def _new_heading_lines(old_string: str, new_string: str) -> list[str]:
    """Heading lines introduced by new_string that aren't in old_string."""
    old_heads = {l for l in old_string.splitlines() if _level(l) is not None}
    return [l for l in new_string.splitlines()
            if _level(l) is not None and l not in old_heads]


def _would_split(args: dict) -> bool:
    """The edit inserts a NEW heading anchored such that it lands mid-section.

    A correct insertion places the new heading immediately before an existing
    heading at the SAME level or shallower (the heading it should precede), or
    at end of file anchored on the last same-level heading.  It splits a
    section when the new heading is SHALLOWER (more top-level) than every
    heading in ``old_string`` — including the case where old_string has NO
    heading at all (anchored on pure body text).  Examples seen in evals:
    ``* Moonshot`` anchored on a body line (no heading) -> split; ``* Moonshot``
    anchored on a ``** TODO`` sub-heading (level 2) -> splits the parent ``*``
    section; ``* Reading`` anchored on ``* Errands`` (same level) -> OK.
    """
    old = args.get("old_string")
    new = args.get("new_string")
    if not isinstance(old, str) or not isinstance(new, str):
        return False
    new_heads = _new_heading_lines(old, new)
    if not new_heads:
        return False  # not adding a heading — none of our concern
    new_min = min(_level(l) for l in new_heads)        # shallowest new heading
    old_levels = _heading_levels(old)
    old_min = min(old_levels) if old_levels else 10 ** 6  # no heading -> "infinitely deep"
    # New heading shallower than the anchor's shallowest heading -> it breaks
    # out of / splits the section the anchor sits inside.
    return new_min < old_min


def _redirect_message() -> str:
    # Deliberately does NOT start with an error prefix ("error:", "cannot",
    # "failed to", …); delivered with status="success" by the guard, so
    # LoopDetectionMiddleware does NOT count it as a loop error and the model
    # keeps re-anchoring (it cycles through anchors toward a valid one).
    return (
        "Adjust the anchor before this edit can apply.  Your `old_string` "
        "anchors the new heading on body text (or a deeper sub-heading), so "
        "the new heading would land in the MIDDLE of an existing section and "
        "split it.\n\n"
        "To add the section, anchor on a HEADING line at the same level or "
        "shallower:\n"
        "Step 1: pick the existing heading your new section should come "
        "immediately BEFORE.\n"
        "Step 2: set `old_string` to that exact heading line.\n"
        "Step 3: set `new_string` to your new heading and its body, then that "
        "same heading line.\n\n"
        "A heading is asterisks then a SPACE (`* Foo`, `** Bar`).  A line like "
        "`*Word.*` (no space after the `*`) is BOLD text, NOT a heading — do "
        "not anchor on it.\n\n"
        "To put the section at the very end, anchor on the LAST top-level "
        "heading and its body, and add yours after."
    )


class OrgHeadingInsertionValidator(EditValidator):
    """Reject a ``.org`` edit that would drop a new heading mid-section."""

    def applies(self, file_path: str) -> bool:
        return file_path.endswith(".org")

    def check(self, old_string: str, new_string: str) -> str | None:
        if _would_split({"old_string": old_string, "new_string": new_string}):
            return _redirect_message()
        return None
