"""Evals for the org-format skill, applied via the general agent.

The org-format skill teaches correct .org-mode editing — specifically the
heading-body relationship and the rule for inserting a new heading without
orphaning the previous heading's body content. The skill is loaded by the
general agent (not the read-only context-agent), since only the writer needs
this knowledge.

These evals are the canary for whether the skills mechanism (deepagents'
SkillsMiddleware + progressive disclosure) actually changes behaviour for
the small model on a real, breakable file-edit task.
"""
import os
import tempfile
from textwrap import dedent

from unittest import TestCase

from assist.agent import create_agent, AgentHarness
from assist.model_manager import select_chat_model

from .utils import create_filesystem, read_file


class TestOrgFormatSkill(TestCase):
    """Verifies the general agent applies the org-format skill when editing
    .org files, and does not apply org conventions when editing other files."""

    def create_agent(self, filesystem: dict):
        root = tempfile.mkdtemp()
        create_filesystem(root, filesystem)
        return AgentHarness(create_agent(self.model, root)), root

    def setUp(self):
        self.model = select_chat_model(0.1)

    def test_inserts_heading_without_orphaning_previous_body(self):
        """Insert a new top-level heading between two existing top-level
        headings. The wrong insertion (immediately after the previous
        heading) orphans the previous heading's body content under the new
        heading; the correct insertion goes after the previous heading's
        full body (immediately before the next same-level heading).

        This is the failure mode the org-format skill is designed to
        prevent. The assertion checks the file's resulting structure: the
        previous heading's body must remain attached to it.
        """
        agent, root = self.create_agent({
            "README.org": "My projects are tracked in projects.org",
            "projects.org": dedent("""\
                * Project Alpha
                Alpha is the first project in the portfolio.

                It has multiple paragraphs of description.

                ** Status
                Running well.

                * Project Beta
                Beta is the second project.
                """),
        })

        agent.message(
            "Add a new top-level project 'Project Gamma' between Alpha and "
            "Beta in projects.org. Its description should be 'Gamma is a "
            "new experimental project.'"
        )

        content = read_file(os.path.join(root, "projects.org"))

        alpha_idx = content.find("* Project Alpha")
        gamma_idx = content.find("* Project Gamma")
        beta_idx = content.find("* Project Beta")

        self.assertGreaterEqual(alpha_idx, 0, "Alpha heading should still be present")
        self.assertGreaterEqual(gamma_idx, 0, "Gamma heading should have been inserted")
        self.assertGreaterEqual(beta_idx, 0, "Beta heading should still be present")
        self.assertLess(alpha_idx, gamma_idx, "Gamma should come after Alpha")
        self.assertLess(gamma_idx, beta_idx, "Gamma should come before Beta")

        # The critical body-rule check: everything between * Project Alpha
        # and * Project Gamma is Alpha's body. It MUST still contain Alpha's
        # description and Alpha's ** Status subsection. If the agent
        # inserted Gamma immediately after the * Project Alpha line, this
        # region would be empty (or contain only whitespace) and Alpha's
        # body would be orphaned under Gamma.
        alpha_section = content[alpha_idx:gamma_idx]
        self.assertIn(
            "Alpha is the first project",
            alpha_section,
            "Alpha's description must remain under its own heading — "
            "agent must not insert Gamma before Alpha's body."
        )
        self.assertIn(
            "** Status",
            alpha_section,
            "Alpha's ** Status subsection must remain under * Project Alpha — "
            "agent must not orphan it by inserting Gamma early."
        )

    def test_md_file_uses_markdown_conventions(self):
        """When editing a .md file, the org-format skill should not apply.
        The agent should use markdown conventions (no asterisk-headings) and
        the response should not contain the org-format SKILL.md content."""
        agent, root = self.create_agent({
            "README.md": "Tasks are tracked in tasks.md",
            "tasks.md": dedent("""\
                # Tasks
                - [ ] Fold laundry
                - [ ] Buy new pants (size 31)
                """),
        })

        res = agent.message("I need to add a task about buying groceries")

        # The task should make it into the markdown file in some form.
        content = read_file(os.path.join(root, "tasks.md"))
        self.assertRegex(
            content.lower(),
            "groceries",
            "The grocery task should have been added to tasks.md"
        )

        # Org-format skill content should not show up in the response — the
        # agent should not be loading or surfacing it for markdown files.
        self.assertNotRegex(
            res,
            r"(?i)(\*.*heading|asterisk|orphan)",
            "Org-format skill content should not surface for .md files"
        )
