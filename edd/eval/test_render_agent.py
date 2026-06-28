"""Eval: the agent emits a ```render block when the user asks to SEE a file.

Regression for thread 20260627170138-b912f542 (agent summarized instead of
showing). The render skill (web-only, assist/web_skills/render) instructs the
model to embed a workspace file via a fenced ``render`` block instead of
read+summarize. Focal test mirrors that thread's shape (realistic multi-file
personal workspace → "show me <named file>"); spot-checks cover other verbs.

Real-LLM eval (small model) — run with the deploy venv; partial pass rates are
expected. The signal here is *emission*: does the model load the render skill
and emit a well-formed render block naming the file? (See the design doc
docs/2026-06-28-render-skill.org — this eval is the gate that chose block-from-
skill over keeping a tool.)
"""
import os
import re
import tempfile
from textwrap import dedent
from unittest import TestCase

from deepagents.backends import FilesystemBackend

from assist.agent import create_agent, AgentHarness
from assist.model_manager import select_assistant_model
from assist.spec import AgentSpec

from .utils import create_filesystem

# The web-only render skill, registered the same way ThreadManager wires it.
_RENDER_SKILLS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assist", "web_skills")

# A render block: a fenced ```render whose body has type: file and the path.
_RENDER_BLOCK = re.compile(r"```render\b(.*?)```", re.S | re.I)


def _personal_workspace() -> dict:
    return {
        "README.org": "Personal notes. Fitness in fitness.org, recipes in recipes.org.",
        "fitness.org": dedent("""\
            * Swimming
            ** 2026
            | date | dist (yd) | time | weight |
            |------+-----------+------+--------|
            | 1/5  |      2600 | 1h   |  168.4 |
            """),
        "swim-workouts.org": "* Swim Workouts\n** 6/9/26 Mixed Stroke (~3000 yd)\n",
        "health.org": "* Wellness visits\n| Test | 2024 |\n|------+------|\n| LDL | 129 |\n",
        "roman-swim.org": "* Log\n** 2025-01-12 @ UCSF\nEntered without issue.\n",
        "journal.org": "* 2026\nA normal day.\n",
        "recipes.org": "* Recipes\n** Pancakes\n- flour, eggs\n",
        "french.org": "* French\n- bonjour = hello\n",
        "financial.org": "* Accounts\n- checking\n",
    }


class TestRenderAgent(TestCase):
    def setUp(self):
        self.model = select_assistant_model(0.1)

    def create_agent(self, filesystem: dict):
        root = tempfile.mkdtemp()
        create_filesystem(root, filesystem)
        skills = {"/render-skill/": FilesystemBackend(root_dir=_RENDER_SKILLS_DIR,
                                                      virtual_mode=True)}
        return AgentHarness(create_agent(self.model, root,
                                         spec=AgentSpec(skill_sources=skills)))

    def _render_block_paths(self, agent) -> list:
        """Bodies of render blocks the agent emitted in its message content."""
        bodies = []
        for m in agent.all_messages():
            content = m.content if isinstance(m.content, str) else ""
            bodies.extend(b for b in _RENDER_BLOCK.findall(content) if "type:" in b.lower())
        return bodies

    def test_emits_render_block_for_named_file(self):
        """Example-thread shape: 'show me <named file>' in a realistic workspace
        emits a render block naming that file (not read+summarize)."""
        agent = self.create_agent(_personal_workspace())
        agent.message("Show me the file with the name fitness.org")
        blocks = self._render_block_paths(agent)
        self.assertTrue(
            any("fitness.org" in b for b in blocks),
            f"expected a render block for fitness.org; render blocks: {blocks}",
        )

    def test_emits_render_block_alternate_verb(self):
        """Generality: a different verb ('open') + a different file."""
        agent = self.create_agent(_personal_workspace())
        agent.message("Open my recipes.org")
        blocks = self._render_block_paths(agent)
        self.assertTrue(
            any("recipes.org" in b for b in blocks),
            f"expected a render block for recipes.org; render blocks: {blocks}",
        )
