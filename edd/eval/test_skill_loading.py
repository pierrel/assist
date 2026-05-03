"""Skill-loading evals across three levels of prompt implicitness.

The `org-format` skill is a clean target for measuring whether the
SkillsMiddleware progressive-disclosure mechanism actually fires:

- It is NOT pre-injected into the system prompt (unlike the `dev` skill,
  which the general agent inlines when project indicators are present),
  so the only way the agent can apply its rules is to load `SKILL.md`
  via the `read_file` tool.
- The skill governs a concrete, breakable mechanic (the heading-body
  rule for org-mode), so a side-channel correctness check is possible
  in addition to the tool-call check.

These three tests vary how loud the hint is — from "use this skill
explicitly" to "no mention at all" — to surface failure modes where
the agent would correctly apply the skill only when named, or only
when domain language is present.

Each test asserts the same thing twice over:

1. The agent loaded the skill — i.e. it called `read_file` (or
   the deepagents alias `ls`) on the `/skills/org-format/SKILL.md`
   path that the SkillsMiddleware advertises.
2. The agent applied the skill correctly — i.e. inserting a heading
   did not orphan the previous heading's body. This is the same
   correctness rule covered in test_org_format_skill.py.
"""
import os
import tempfile
from textwrap import dedent

from unittest import TestCase

from langchain_core.messages import AIMessage

from assist.agent import create_agent, AgentHarness
from assist.model_manager import select_chat_model

from .utils import create_filesystem, read_file


_PROJECTS_FIXTURE = dedent("""\
    * Project Alpha
    Alpha is the first project in the portfolio.

    It has multiple paragraphs of description.

    ** Status
    Running well.

    * Project Beta
    Beta is the second project.
    """)


class TestSkillLoading(TestCase):
    """Verifies the agent reaches for the org-format skill across a
    range of prompt implicitness levels."""

    def setUp(self):
        self.model = select_chat_model(0.1)

    def _make_agent(self):
        root = tempfile.mkdtemp()
        create_filesystem(root, {
            "README.org": "My projects are tracked in projects.org",
            "projects.org": _PROJECTS_FIXTURE,
        })
        return AgentHarness(create_agent(self.model, root)), root

    def _skill_was_loaded(self, agent, skill_name: str) -> bool:
        """Return True iff any tool call loaded the skill's body.

        Two recognized signals:

        - ``read_file`` with a path containing ``/skills/{skill_name}/``
        - ``load_skill`` with ``name == skill_name``

        We grep tool-call args rather than tool results because the
        model proves intent the moment it issues the call.
        """
        path_needle = f"/skills/{skill_name}/"
        for m in agent.all_messages():
            if not isinstance(m, AIMessage) or not m.tool_calls:
                continue
            for tc in m.tool_calls:
                args = tc.get("args") or {}
                if tc.get("name") == "load_skill" and args.get("name") == skill_name:
                    return True
                for v in args.values():
                    if isinstance(v, str) and path_needle in v:
                        return True
        return False

    def _assert_alpha_body_preserved(self, root: str):
        content = read_file(os.path.join(root, "projects.org"))
        alpha_idx = content.find("* Project Alpha")
        gamma_idx = content.find("* Project Gamma")
        self.assertGreaterEqual(alpha_idx, 0, "Alpha heading missing")
        self.assertGreaterEqual(gamma_idx, 0, "Gamma heading was not inserted")
        self.assertLess(alpha_idx, gamma_idx, "Gamma should come after Alpha")

        alpha_section = content[alpha_idx:gamma_idx]
        self.assertIn(
            "Alpha is the first project",
            alpha_section,
            "Alpha's body was orphaned — the org-format skill rule was "
            "not applied even though the skill should have been loaded."
        )
        self.assertIn(
            "** Status",
            alpha_section,
            "Alpha's ** Status subsection was orphaned under Gamma — "
            "the skill rule was not applied."
        )

    def test_suggested_skill_request(self):
        """Implicitness level 1 — domain hints, no skill name.

        The user mentions the failure mode the skill prevents
        ("orphaning", "heading body rules") but does not say "skill"
        or "org-format". The agent has to match the description's
        wording against the user's wording without being told to.
        """
        agent, root = self._make_agent()

        agent.message(
            "Add a new top-level project 'Project Gamma' between Alpha "
            "and Beta in projects.org. Its description should be "
            "'Gamma is a new experimental project.' Be careful with "
            "org-mode heading-body rules — I don't want Alpha's body "
            "to end up orphaned under the new heading."
        )

        self.assertTrue(
            self._skill_was_loaded(agent, "org-format"),
            "Agent did not load /skills/org-format/SKILL.md despite "
            "the user's prompt mentioning the exact problem the skill "
            "addresses (heading-body rules, orphaning). The agent "
            "should have matched these hints against the skill "
            "description and loaded SKILL.md."
        )
        self._assert_alpha_body_preserved(root)

    def test_implicit_skill_request(self):
        """Implicitness level 2 — no hints, just the task.

        The user says nothing about skills, formatting, or org-mode
        rules. The only signal is the `.org` file extension, which the
        skill description (``Load before reading, editing, or
        surfacing any .org file``) is supposed to key off of. This is
        the hardest case and the one progressive disclosure is meant
        to handle: the agent must choose to load the skill from the
        description alone.
        """
        agent, root = self._make_agent()

        agent.message(
            "Add a new top-level project 'Project Gamma' between Alpha "
            "and Beta in projects.org. Its description should be "
            "'Gamma is a new experimental project.'"
        )

        self.assertTrue(
            self._skill_was_loaded(agent, "org-format"),
            "Agent did not load /skills/org-format/SKILL.md. The user "
            "asked to edit a .org file — the org-format skill's "
            "description tells the agent to load before editing any "
            ".org file, so the agent should have loaded SKILL.md "
            "from the description alone."
        )
        self._assert_alpha_body_preserved(root)
