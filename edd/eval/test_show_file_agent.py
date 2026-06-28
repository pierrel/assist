"""Eval: the general agent calls `show_file` when the user asks to SEE a file.

Regression for a live thread (20260627170138-b912f542) where the user said
"Show me the one with the name 'fitness.org'" and the agent ran read_file +
summarized instead of calling show_file.  The focal test mirrors that thread's
shape: a realistic, multi-file personal workspace, then a "show me <named file>"
request.  Spot-checks cover other verbs/extensions for generality.

Real-LLM eval (small model) — run with the deploy venv; partial pass rates are
expected.  See assist/CLAUDE.md eval cadence.
"""
import tempfile
from textwrap import dedent
from unittest import TestCase

from assist.agent import create_agent, AgentHarness
from assist.model_manager import select_assistant_model
from assist.spec import AgentSpec
from assist.tools import show_file

from .utils import create_filesystem


# A realistic personal workspace (subset of the real thread's shape): the
# fitness file lives among many swim/health files AND unrelated notes, so
# "show me fitness.org" is a genuine selection, not the only candidate.
def _personal_workspace() -> dict:
    return {
        "README.org": "Personal notes. Fitness in fitness.org, recipes in recipes.org.",
        "fitness.org": dedent("""\
            * Swimming
            ** 2026
            | date | dist (yd) | time | weight |
            |------+-----------+------+--------|
            | 1/5  |      2600 | 1h   |  168.4 |
            | 1/7  |      2650 | 55m  |  169.8 |
            """),
        "swim-workouts.org": "* Swim Workouts\n** 6/9/26 Mixed Stroke (~3000 yd)\n",
        "swim-prehab-routine.org": "#+TITLE: Prehab\n* Morning\n- 90/90 breathing\n",
        "health.org": "* Wellness visits\n| Test | 2024 |\n|------+------|\n| LDL | 129 |\n",
        "roman-swim.org": "* Log\n** 2025-01-12 @ UCSF\nEntered without issue.\n",
        "references": {
            "swim-workout-best-practices.org": "* Swim Best Practices\n** Warm-Up\n",
        },
        "journal.org": "* 2026\nA normal day.\n",
        "recipes.org": "* Recipes\n** Pancakes\n- flour, eggs\n",
        "french.org": "* French\n- bonjour = hello\n",
        "financial.org": "* Accounts\n- checking\n",
    }


class TestShowFileAgent(TestCase):
    def setUp(self):
        self.model = select_assistant_model(0.1)

    def create_agent(self, filesystem: dict):
        root = tempfile.mkdtemp()
        create_filesystem(root, filesystem)
        spec = AgentSpec(tools=(show_file,))
        return AgentHarness(create_agent(self.model, root, spec=spec)), root

    def _show_file_paths(self, agent) -> list:
        paths = []
        for m in agent.all_messages():
            for c in (getattr(m, "tool_calls", None) or []):
                if c.get("name") == "show_file":
                    paths.append((c.get("args") or {}).get("path", ""))
        return paths

    def test_shows_named_file_in_personal_workspace(self):
        """Example-thread shape: a realistic workspace + 'show me <named file>'
        must call show_file for that file, not read+summarize."""
        agent, _ = self.create_agent(_personal_workspace())
        agent.message("Show me the file with the name fitness.org")
        paths = self._show_file_paths(agent)
        self.assertTrue(
            any("fitness.org" in p for p in paths),
            f"expected show_file called for fitness.org; show_file calls: {paths}",
        )

    def test_opens_file_alternate_phrasing(self):
        """Generality: a different verb ('open') + a different file."""
        agent, _ = self.create_agent(_personal_workspace())
        agent.message("Open my recipes.org")
        paths = self._show_file_paths(agent)
        self.assertTrue(
            any("recipes.org" in p for p in paths),
            f"expected show_file called for recipes.org; show_file calls: {paths}",
        )

    def test_shows_pdf(self):
        """Generality: a PDF + 'view'."""
        fs = _personal_workspace()
        fs["property-tax-receipt.pdf"] = "%PDF-1.4 fake receipt bytes"
        agent, _ = self.create_agent(fs)
        agent.message("View my property-tax-receipt.pdf")
        paths = self._show_file_paths(agent)
        self.assertTrue(
            any("property-tax-receipt.pdf" in p for p in paths),
            f"expected show_file called for the pdf; show_file calls: {paths}",
        )
