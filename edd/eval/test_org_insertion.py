"""Eval: adding an item to an org file must not split an existing section.

Reproduces a production failure mode (2026-06-03 threads): asked to add a
new heading/TODO, the small model picked a poor ``edit_file`` anchor — a
body line in the MIDDLE of an unrelated section — and inserted the new
heading there, landing it mid-section and breaking the org hierarchy
(then often having to detect and undo it).  See
``docs/2026-06-03-org-insertion-mid-section.org`` (design doc).

All fixtures are synthetic — real user org files contain PII.

The contract these pin: a new item is added at a structurally valid
location (its own well-formed heading) and NO pre-existing section is
split — every original heading keeps all of its original body lines
together under it.
"""
import tempfile
from textwrap import dedent
from unittest import TestCase

from assist.model_manager import select_assistant_model
from assist.agent import create_agent, AgentHarness

from .utils import read_file, create_filesystem, AgentTestMixin


def _top_level_blocks(text: str) -> dict[str, list[str]]:
    """Map each top-level (``* ``) heading to the body lines beneath it,
    up to the next top-level heading.  Sub-headings (``** ``) and their
    bodies count as body lines of the enclosing top-level section."""
    blocks: dict[str, list[str]] = {}
    cur = None
    for line in text.splitlines():
        if line.startswith("* "):
            cur = line
            blocks[cur] = []
        elif cur is not None:
            blocks[cur].append(line)
    return blocks


def _item_bodies(text: str) -> dict[str, list[str]]:
    """Map each heading at ANY level to the body lines (non-heading)
    directly beneath it, up to the next heading of any level.  Used to
    verify an item's own body wasn't split by an inserted heading."""
    bodies: dict[str, list[str]] = {}
    cur = None
    for line in text.splitlines():
        if line.lstrip().startswith("*") and line.lstrip().lstrip("*").startswith(" "):
            cur = line.strip()
            bodies[cur] = []
        elif cur is not None and line.strip():
            bodies[cur].append(line.strip())
    return bodies


class OrgInsertionMixin(AgentTestMixin):
    def create_agent(self, filesystem: dict):
        root = tempfile.mkdtemp(prefix="org_insert_")
        create_filesystem(root, filesystem)
        return AgentHarness(create_agent(self.model, root)), root

    def setUp(self):
        self.model = select_assistant_model(0.1)

    def assertSectionNotSplit(self, before: str, after: str, heading: str,
                              body_lines: list[str]):
        """Every original body line of ``heading`` must still live together
        under that same heading in ``after`` (no inserted heading split it)."""
        blocks = _top_level_blocks(after)
        self.assertIn(heading, blocks,
                      f"Original section {heading!r} disappeared.\n---\n{after}")
        joined = "\n".join(blocks[heading])
        for bl in body_lines:
            self.assertIn(
                bl, joined,
                f"Body line {bl!r} of {heading!r} is no longer under it — "
                f"a heading was likely inserted mid-section.\n---\n{after}")


class TestAddTopLevelSection(OrgInsertionMixin, TestCase):
    """Add a new top-level section to a multi-section roadmap-style file
    (mirrors the 'moonshot' prod case)."""

    ROADMAP = dedent("""\
        * Alpha
        First note about alpha.
        A second alpha detail that matters.

        * Beta
        A line about beta.
        Another beta detail in the middle.
        A final beta point at the end.

        * Gamma
        Some gamma content here.
        """)

    def test_new_section_does_not_split_existing(self):
        agent, root = self.create_agent({"roadmap.org": self.ROADMAP})
        agent.message(
            "Add a new top-level roadmap item titled 'Delta' with a one-line "
            "note that it's an experimental idea.")
        after = read_file(f"{root}/roadmap.org")

        # A Delta heading was added at top level.
        self.assertRegex(after, r"(?im)^\* .*delta",
                         f"Expected a top-level Delta heading.\n---\n{after}")
        # The multi-line Beta section (the splittable one) stays intact.
        self.assertSectionNotSplit(
            self.ROADMAP, after, "* Beta",
            ["A line about beta.",
             "Another beta detail in the middle.",
             "A final beta point at the end."])
        # Alpha too.
        self.assertSectionNotSplit(
            self.ROADMAP, after, "* Alpha",
            ["First note about alpha.",
             "A second alpha detail that matters."])


class TestAddTodoUnderHeading(OrgInsertionMixin, TestCase):
    """Add a TODO to a gtd inbox without splitting an existing item
    (mirrors the 'tax review' prod case)."""

    INBOX = dedent("""\
        * Tasks
        ** TODO Buy groceries
        Milk, eggs, and bread.
        ** TODO Call the plumber
        About the kitchen leak under the sink.
        ** TODO Renew the library card
        It expires at the end of the month.
        """)

    def test_new_todo_does_not_split_existing_item(self):
        agent, root = self.create_agent({"gtd": {"inbox.org": self.INBOX}})
        agent.message("Add a task to review the quarterly budget.")
        after = read_file(f"{root}/gtd/inbox.org")

        # A new ** TODO was added.
        self.assertRegex(after, r"(?im)^\*\* TODO .*budget",
                         f"Expected a new budget TODO.\n---\n{after}")
        # No existing item's body was split by an inserted heading.
        bodies = _item_bodies(after)
        for head, body in [
            ("** TODO Call the plumber",
             "About the kitchen leak under the sink."),
            ("** TODO Buy groceries", "Milk, eggs, and bread."),
            ("** TODO Renew the library card",
             "It expires at the end of the month."),
        ]:
            self.assertIn(head, bodies,
                          f"Item {head!r} disappeared.\n---\n{after}")
            self.assertIn(
                body, " ".join(bodies[head]),
                f"Body of {head!r} was split off — a heading was inserted "
                f"mid-item.\n---\n{after}")
