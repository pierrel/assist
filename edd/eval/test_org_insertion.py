"""Eval: adding an item to an org file must not split an existing section,
and must get the anchor right on the FIRST edit (no mid-section thrash).

Reproduces a production failure mode (2026-06-03 threads): asked to add a
new heading to a LARGE org file, the small model anchored its first
``edit_file`` on a body line in the MIDDLE of an unrelated section —
often around an org *bold* line (``*Direction.*``, ``*Concrete next
step.*``) that LOOKS like a heading — dropping the new heading there and
splitting the section, then burning several more edits self-correcting.
The final file sometimes ended up fine, so a final-state-only check
misses it; we assert the FIRST edit is already structurally correct.

The roadmap fixture is a frozen snapshot of the real (tracked, PII-free)
``roadmap.org`` that triggered the failure — ~365 lines, deep nesting,
many bold-looking lines — because smaller synthetic files don't reproduce
it (34/34 passed).  The inbox fixture is synthetic (gtd files have PII).
See ``docs/2026-06-03-org-insertion-mid-section.org``.
"""
import tempfile
from pathlib import Path
from textwrap import dedent
from unittest import TestCase

import pytest

from assist.model_manager import select_assistant_model
from assist.agent import create_agent, AgentHarness

from .utils import read_file, create_filesystem, AgentTestMixin


def _heading_depth(line: str) -> int | None:
    st = line.lstrip()
    if st.startswith("*") and st.lstrip("*").startswith(" "):
        return len(st) - len(st.lstrip("*"))
    return None


def _blocks(text: str, level: int) -> dict[str, list[str]]:
    """Group body lines under headings at ``level`` asterisks or fewer."""
    blocks: dict[str, list[str]] = {}
    cur = None
    for line in text.splitlines():
        d = _heading_depth(line)
        if d is not None and d <= level:
            cur = line.strip()
            blocks.setdefault(cur, [])
        elif cur is not None:
            blocks[cur].append(line)
    return blocks


def _section_intact(text: str, level: int, heading: str,
                    body_lines: list[str]) -> str | None:
    blocks = _blocks(text, level)
    if heading not in blocks:
        return f"section {heading!r} disappeared"
    joined = "\n".join(blocks[heading])
    for bl in body_lines:
        if bl not in joined:
            return f"body line {bl!r} no longer under {heading!r} (section split)"
    return None


def _bracket_body(text: str, level: int, heading: str) -> list[str]:
    """First and last non-blank body lines under ``heading`` — they bracket
    the section, so a heading inserted ANYWHERE inside orphans the last one
    (or the first).  Distinctive full sentences, safe for substring match."""
    block = _blocks(text, level).get(heading, [])
    body = [l.strip() for l in block
            if l.strip() and _heading_depth(l) is None]
    if not body:
        return []
    return [body[0]] if len(body) == 1 else [body[0], body[-1]]


_ROADMAP = (Path(__file__).parent / "fixtures" / "roadmap_sample.org").read_text()

# Dense real sections to keep intact (each has bold-looking body lines).
_ROADMAP_TARGETS = {
    h: _bracket_body(_ROADMAP, 1, h)
    for h in ("* Architecture", "* Research improvements", "* Reliability")
}


class OrgInsertionMixin(AgentTestMixin):
    def create_agent(self, filesystem: dict):
        root = tempfile.mkdtemp(prefix="org_insert_")
        create_filesystem(root, filesystem)
        return AgentHarness(create_agent(self.model, root)), root

    def setUp(self):
        self.model = select_assistant_model(0.1)

    def first_edit_result(self, agent, original: str, basename: str):
        """Apply the FIRST edit_file to ``basename`` to ``original``; return
        the result, or None if the model used write_file / no edit_file."""
        for m in agent.all_messages():
            for tc in (getattr(m, "tool_calls", None) or []):
                a = tc.get("args") or {}
                fp = a.get("file_path") or a.get("path") or ""
                if tc.get("name") == "edit_file" and fp.endswith(basename):
                    old = a.get("old_string") or ""
                    new = a.get("new_string") or ""
                    return original.replace(old, new, 1) if old and old in original else original
                if tc.get("name") == "write_file" and fp.endswith(basename):
                    return None
        return None

    def assertFirstEditClean(self, agent, original, basename, level, targets):
        first = self.first_edit_result(agent, original, basename)
        if first is None:
            return  # whole-file write / no edit — not a mid-anchor failure
        for head, body in targets.items():
            err = _section_intact(first, level, head, body)
            self.assertIsNone(
                err,
                f"FIRST edit split a section ({err}). The model mis-anchored "
                f"its first edit_file mid-section.\n--- first-edit result ---\n{first}")


class TestAddTopLevelSection(OrgInsertionMixin, TestCase):
    """Add a new top-level section to the real ~365-line roadmap (mirrors
    the 'moonshot' prod case).  Vague prompt — does not telegraph 'top-level
    section' or where it goes."""

    # KNOWN FAILING (reproduced bug, fix pending — see the design doc).
    # Baseline 5/5 FAIL; four skill variants (example, procedure, both,
    # append-at-end) all failed 0/3 — the small model mistakes the org *bold*
    # line `*Direction.*` for a heading and anchors there.  strict=False so an
    # occasional pass won't break the suite; remove the marker when a real fix
    # (e.g. a deterministic guard) lands.
    @pytest.mark.xfail(reason="org bold mis-anchor on large files; skill-only "
                              "changes don't fix it — fix pending", strict=False)
    def test_first_edit_does_not_split_section(self):
        agent, root = self.create_agent({"roadmap.org": _ROADMAP})
        agent.message(
            "Add Moonshot to the roadmap — the idea of a self-improving agent "
            "that rewrites its own skills.")
        after = read_file(f"{root}/roadmap.org")
        self.assertRegex(after, r"(?im)^\* .*moonshot",
                         f"Expected a top-level Moonshot heading.\n---\n{after}")
        self.assertFirstEditClean(agent, _ROADMAP, "roadmap.org", 1, _ROADMAP_TARGETS)
        for head, body in _ROADMAP_TARGETS.items():
            self.assertIsNone(_section_intact(after, 1, head, body),
                              f"final state split {head!r}.\n---\n{after}")


_SIMPLE = dedent("""\
    * Groceries
    Milk, eggs, and bread.
    * Chores
    Vacuum the living room.
    * Errands
    Drop off the package.
    """)

_SIMPLE_TARGETS = {
    "* Groceries": ["Milk, eggs, and bread."],
    "* Chores": ["Vacuum the living room."],
    "* Errands": ["Drop off the package."],
}


class TestAddSimpleSection(OrgInsertionMixin, TestCase):
    """Backstop: a SMALL, easy file must keep working too — the fix must not
    be tuned only for the hard real-roadmap case at the expense of simple
    ones.  (Passes at baseline; must still pass after the fix.)"""

    def test_first_edit_does_not_split_section(self):
        agent, root = self.create_agent({"notes.org": _SIMPLE})
        agent.message("Add a section called Reading with a note to finish my book.")
        after = read_file(f"{root}/notes.org")
        self.assertRegex(after, r"(?im)^\* .*reading",
                         f"Expected a Reading section.\n---\n{after}")
        self.assertFirstEditClean(agent, _SIMPLE, "notes.org", 1, _SIMPLE_TARGETS)
        for head, body in _SIMPLE_TARGETS.items():
            self.assertIsNone(_section_intact(after, 1, head, body),
                              f"final state split {head!r}.\n---\n{after}")


_INBOX = dedent("""\
    * Tasks
    ** TODO Buy groceries
    Milk, eggs, and bread for the week.
    ** TODO Call the plumber
    About the kitchen leak under the sink.

    He said to call back after Tuesday.
    ** TODO Renew the library card
    It expires at the end of the month.
    ** TODO Schedule dentist cleaning
    Overdue by a few months now.
    ** TODO Fix the bike tire
    The rear one keeps going flat.

    Patch kit is in the garage somewhere.
    ** TODO Reply to the landlord
    About renewing the lease for another year.
    ** TODO Back up the laptop
    Hasn't been done since the OS upgrade.
    ** TODO Water the plants
    The ferns especially need attention.
    """)


class TestAddTodoUnderHeading(OrgInsertionMixin, TestCase):
    """Add a TODO to a gtd inbox without splitting an existing item (mirrors
    the 'tax review' prod case)."""

    def test_first_edit_does_not_split_item(self):
        agent, root = self.create_agent({"gtd": {"inbox.org": _INBOX}})
        agent.message("Add a task to review the quarterly budget.")
        after = read_file(f"{root}/gtd/inbox.org")
        self.assertRegex(after, r"(?im)^\*\* TODO .*budget",
                         f"Expected a new budget TODO.\n---\n{after}")
        targets = {
            "** TODO Call the plumber": ["He said to call back after Tuesday."],
            "** TODO Fix the bike tire": ["Patch kit is in the garage somewhere."],
        }
        self.assertFirstEditClean(agent, _INBOX, "inbox.org", 2, targets)
        for head, body in targets.items():
            self.assertIsNone(_section_intact(after, 2, head, body),
                              f"final state split {head!r}.\n---\n{after}")
