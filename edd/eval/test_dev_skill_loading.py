"""Skill-loading evals for the `dev` skill across three implicitness levels.

Mirror of ``test_skill_loading.py`` for the ``dev`` skill. The ``dev``
skill used to be hard-injected into the system prompt when a top-level
project file was detected (a now-removed scan). After the migration the
agent selects the skill on its own — same path as ``org-format``.

Three tests, each varying how loud the hint is:

1. **Explicit** — the user names the skill directly. Sanity check on
   the middleware wiring.
2. **Suggested** — coding-task language ("function", "test", "TDD") but
   no skill name. The agent has to match phrasing against the
   description.
3. **Soft trigger** — a single coding-adjacent word ("code") in the
   message; the actual content the agent must read to answer lives on
   disk. Originally framed as filesystem-only, but the small model
   does not reliably load skills from workspace observations alone
   without workspace-aware skill selection (a structural change
   deferred for later).

Termination: each test patches ``_make_load_skill_tool`` so the
middleware's ``load_skill`` tool returns a STOP sentinel rather than
the real skill body. The model reliably halts when given an explicit
"reply DONE and call no more tools" instruction, and we still see the
``load_skill`` call in the message history. The post-hoc check from
``test_skill_loading.py:71-93`` is what we assert on — the *intent*,
not the result.

We do NOT use exceptions for early termination: langchain's
``ToolNode`` catches tool exceptions and feeds them back to the model
as ``ToolMessage`` content, so the exception would never escape
``agent.message`` (see
``langgraph.prebuilt.tool_node._default_handle_tool_errors``).
"""
import tempfile
from textwrap import dedent
from unittest import TestCase
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from assist.agent import create_agent, AgentHarness
from assist.model_manager import select_chat_model

from .utils import create_filesystem


_PYPROJECT_FIXTURE = dedent("""\
    [project]
    name = "demo"
    version = "0.1.0"
    description = "Tiny demo project for skill-loading evals."
    requires-python = ">=3.10"
    dependencies = []

    [build-system]
    requires = ["setuptools"]
    build-backend = "setuptools.build_meta"
    """)


_MATHLIB_FIXTURE = dedent('''\
    """Tiny math helpers."""


    def add(a: int, b: int) -> int:
        """Return ``a + b``."""
        return a + b
    ''')


_README_FIXTURE = dedent("""\
    # demo

    A tiny Python project used as a fixture for skill-loading evals.
    """)


_STOP_SENTINEL = (
    "Skill load recorded for testing. STOP NOW. Reply with the single "
    "word DONE and call no more tools. Do not call read_file, ls, "
    "edit_file, write_file, task, or any other tool. The test is over."
)


def _make_stop_load_skill_tool(*_args, **_kwargs):
    """Replacement factory for ``_make_load_skill_tool``.

    Returns a ``load_skill`` tool that ignores the skill name, records
    nothing of its own (the eval inspects ``AIMessage.tool_calls``
    directly), and returns a STOP sentinel telling the model to halt.
    """

    @tool
    def load_skill(name: str) -> str:
        """Load the named skill body. (Stubbed in eval — returns a stop
        sentinel so the agent halts after the skill choice is recorded.)
        """
        return _STOP_SENTINEL

    return load_skill


class TestDevSkillLoading(TestCase):
    """Verifies the agent reaches for the ``dev`` skill across a range
    of prompt implicitness levels — without the hard-inject path."""

    def setUp(self):
        self.model = select_chat_model(0.1)

    def _make_agent(self):
        """Create an agent rooted at a temp dir that looks like a Python
        project (pyproject.toml + a tiny module + README).

        ``create_agent`` is called inside the patch context so the
        middleware uses the stop-sentinel ``load_skill`` factory.
        """
        root = tempfile.mkdtemp(prefix="dev_skill_loading_")
        create_filesystem(root, {
            "pyproject.toml": _PYPROJECT_FIXTURE,
            "mathlib.py": _MATHLIB_FIXTURE,
            "README.md": _README_FIXTURE,
        })
        return AgentHarness(create_agent(self.model, root)), root

    def _skill_was_loaded(self, agent, skill_name: str) -> bool:
        """Return True iff any tool call attempted to load *skill_name*.

        Inspects ``AIMessage.tool_calls`` so a ``load_skill(name="dev")``
        invocation counts as the skill being chosen, regardless of what
        the (stubbed) tool returned. Only the ``load_skill`` route is
        recognized here — ``read_file('/skills/...')`` is no longer how
        the agent loads skills, and a fallback that accepted it would
        mask regressions where the model goes off-script.
        """
        for m in agent.all_messages():
            if not isinstance(m, AIMessage) or not m.tool_calls:
                continue
            for tc in m.tool_calls:
                if tc.get("name") != "load_skill":
                    continue
                args = tc.get("args") or {}
                if args.get("name") == skill_name:
                    return True
        return False

    def _attempted_skills(self, agent) -> list[str]:
        """Return the list of names passed to ``load_skill`` (in order)."""
        names: list[str] = []
        for m in agent.all_messages():
            if not isinstance(m, AIMessage) or not m.tool_calls:
                continue
            for tc in m.tool_calls:
                if tc.get("name") == "load_skill":
                    n = (tc.get("args") or {}).get("name")
                    if n:
                        names.append(n)
        return names

    # ------------------------------------------------------------------

    def test_explicit_dev_skill_request(self):
        """Implicitness 0 — the user names the dev skill outright.

        Pure middleware-wiring sanity check. A failure here means the
        skill isn't being advertised at all, the ``load_skill`` tool
        isn't bound, or the prompt template hides the listing.
        """
        with patch(
            "assist.middleware.skills_middleware._make_load_skill_tool",
            _make_stop_load_skill_tool,
        ):
            agent, _ = self._make_agent()
            agent.message(
                "Load the dev skill before doing anything. Then add a "
                "function `subtract(a, b)` to mathlib.py with a test."
            )

            self.assertTrue(
                self._skill_was_loaded(agent, "dev"),
                f"Agent did not call load_skill(name='dev') despite the "
                f"user explicitly naming the skill. "
                f"Skills attempted: {self._attempted_skills(agent)}"
            )

    def test_suggested_dev_skill_request(self):
        """Implicitness 1 — coding-task language, no skill name.

        The user says "function", "test", "TDD" — terms straight out of
        the dev skill's description. The agent has to match the
        description's wording against the user's wording without being
        told to.
        """
        with patch(
            "assist.middleware.skills_middleware._make_load_skill_tool",
            _make_stop_load_skill_tool,
        ):
            agent, _ = self._make_agent()
            agent.message(
                "Add a function `subtract(a, b)` to mathlib.py and write "
                "a test for it. Follow TDD — failing test first, then the "
                "implementation."
            )

            self.assertTrue(
                self._skill_was_loaded(agent, "dev"),
                f"Agent did not call load_skill(name='dev') despite "
                f"coding-task language ('function', 'test', 'TDD') in "
                f"the prompt — all of which are in the dev skill "
                f"description. "
                f"Skills attempted: {self._attempted_skills(agent)}"
            )

    def test_soft_trigger_dev_skill_request(self):
        """Implicitness 2 — single soft trigger word, content on disk.

        The prompt mentions "code" once but says nothing about
        functions, tests, TDD, debugging, or the skill itself. The
        agent has to match that single word against the description
        and load the skill before doing anything else. The actual
        answer ("what is it?") still depends on reading files on disk,
        but the *trigger* lives in the message.

        A fully bare prompt (no trigger words at all) does not
        reliably fire skill loading on the small model — the
        filesystem alone is not enough without workspace-aware skill
        selection (a structural change deferred for later).
        """
        with patch(
            "assist.middleware.skills_middleware._make_load_skill_tool",
            _make_stop_load_skill_tool,
        ):
            agent, _ = self._make_agent()
            agent.message(
                "Hey, take a look at this code and tell me what it is."
            )

            self.assertTrue(
                self._skill_was_loaded(agent, "dev"),
                f"Agent did not call load_skill(name='dev') despite the "
                f"single soft trigger ('this code') in the prompt — "
                f"'code' is in the dev skill description. "
                f"Skills attempted: {self._attempted_skills(agent)}"
            )
