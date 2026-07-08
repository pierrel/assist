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
        """Bodies of render blocks the AGENT emitted — only AIMessage content, so
        the loaded skill's own example blocks (a ToolMessage) don't count."""
        from langchain_core.messages import AIMessage
        bodies = []
        for m in agent.all_messages():
            if not isinstance(m, AIMessage):
                continue
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

    def test_emits_line_range(self):
        """Section by line: 'show me lines X-Y of <file>' carries a lines: range."""
        fs = dict(_personal_workspace())
        fs["log.org"] = "* Log\n" + "".join(f"- entry {i}\n" for i in range(1, 60))
        agent = self.create_agent(fs)
        agent.message("Show me lines 10 to 20 of log.org")
        blocks = self._render_block_paths(agent)
        self.assertTrue(
            any("log.org" in b and "lines:" in b.lower() for b in blocks),
            f"expected a render block for log.org with a lines: range; blocks: {blocks}",
        )

    def test_resolves_described_section_to_line_range(self):
        """The common case: 'show me the section about X' (no explicit numbers) —
        the agent must read the file, locate the section, and emit a lines: range
        (description resolved to numbers, not left as prose)."""
        fs = dict(_personal_workspace())
        fs["config.org"] = (
            "* Intro\nsome intro text\nmore intro\n"
            "* Backups\nback up to the NAS nightly\nkeep three copies offsite\n"
            "* Networking\nwifi is on channel 6\nrouter in the closet\n")
        agent = self.create_agent(fs)
        agent.message("Show me the section about backups in config.org")
        blocks = self._render_block_paths(agent)
        self.assertTrue(
            any("config.org" in b and "lines:" in b.lower() for b in blocks),
            f"expected a render block for config.org with a resolved lines: range; "
            f"blocks: {blocks}",
        )

    def _write_paths(self, agent) -> list:
        """Paths the agent wrote/edited this turn (write_file / edit_file tool calls)."""
        from langchain_core.messages import AIMessage
        paths = []
        for m in agent.all_messages():
            if not isinstance(m, AIMessage):
                continue
            for tc in (getattr(m, "tool_calls", None) or []):
                if tc.get("name") in ("write_file", "edit_file"):
                    args = tc.get("args") or tc.get("arguments") or {}
                    if isinstance(args, dict):
                        p = args.get("file_path") or args.get("path") or ""
                        if p:
                            paths.append(str(p))
        return paths

    def test_shows_unsupported_file_instead_of_summarizing(self):
        """The 2026-07-08 regression: asked to SHOW a file whose extension isn't
        .org/.md/.pdf (here a .txt.j2 template), the agent must render it — either
        the file directly (it now renders as text) OR a converted /tmp/*.md copy —
        NOT fall back to reading + summarizing it."""
        fs = dict(_personal_workspace())
        fs["research_prompt.txt.j2"] = (
            "You are a dedicated researcher. Conduct thorough research and cite sources. "
            "MARKER-QZX7 always cite provenance.")
        agent = self.create_agent(fs)
        agent.message("Show me the research_prompt.txt.j2 file")
        blocks = self._render_block_paths(agent)
        writes = self._write_paths(agent)
        # a render block for the original unsupported file (text fallback) OR for a
        # /tmp/*.md the agent wrote from it — either way it chose to SHOW, not summarize.
        showed_original = any("research_prompt.txt.j2" in b for b in blocks)
        showed_tmp_copy = (any("/tmp/" in b and ".md" in b.lower() for b in blocks)
                           and any("/tmp/" in w and w.lower().endswith(".md") for w in writes))
        self.assertTrue(
            showed_original or showed_tmp_copy,
            f"expected a render block for the .txt.j2 (direct text) or a /tmp/*.md copy "
            f"of it; render blocks: {blocks}; writes: {writes}",
        )

    def test_converts_unsupported_file_to_tmp_md_when_asked_formatted(self):
        """The copy/modify-for-render path specifically: asked to show an unsupported
        file NICELY FORMATTED, the agent writes a converted /tmp/*.md and renders that
        (using the now-persistent /tmp scratch dir)."""
        fs = dict(_personal_workspace())
        fs["notes.py"] = "# TODO: MARKER-PY42\nprint('backups run nightly')\n"
        agent = self.create_agent(fs)
        agent.message("Show me notes.py nicely formatted as markdown")
        blocks = self._render_block_paths(agent)
        writes = self._write_paths(agent)
        wrote_tmp_md = any("/tmp/" in w and w.lower().endswith(".md") for w in writes)
        rendered_tmp = any("/tmp/" in b and ".md" in b.lower() for b in blocks)
        self.assertTrue(
            wrote_tmp_md and rendered_tmp,
            f"expected the agent to write a /tmp/*.md copy AND render it; "
            f"writes: {writes}; render blocks: {blocks}",
        )

    def test_emits_page_range(self):
        """Section by page: 'show page N of <pdf>' carries a pages: range."""
        from pypdf import PdfWriter
        root = tempfile.mkdtemp()  # throwaway, same as create_agent's pattern
        create_filesystem(root, _personal_workspace())
        w = PdfWriter()
        for _ in range(6):
            w.add_blank_page(width=200, height=200)
        with open(os.path.join(root, "report.pdf"), "wb") as f:
            w.write(f)
        skills = {"/render-skill/": FilesystemBackend(root_dir=_RENDER_SKILLS_DIR,
                                                      virtual_mode=True)}
        agent = AgentHarness(create_agent(self.model, root,
                                          spec=AgentSpec(skill_sources=skills)))
        agent.message("Show me page 3 of report.pdf")
        blocks = self._render_block_paths(agent)
        self.assertTrue(
            any("report.pdf" in b and "pages:" in b.lower() for b in blocks),
            f"expected a render block for report.pdf with a pages: range; blocks: {blocks}",
        )
