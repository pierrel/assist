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
import shutil
import tempfile
from textwrap import dedent
from unittest import TestCase

from assist.agent import create_agent, AgentHarness
from assist.model_manager import select_assistant_model
from assist.sandbox_manager import SandboxManager
from assist.spec import AgentSpec
from assist.thread_manager import _web_skill_sources

from .utils import (agent_tool_calls, complete_web_main_tasks, create_filesystem,
                    prompt_rewrite_web_main_spec,
                    stub_research_subagent)

# A render block: a fenced ```render whose body has type: file and the path.
_RENDER_BLOCK = re.compile(r"```render\b(.*?)```", re.S | re.I)

# Complete 1x1 transparent PNG used when the graph already exists. Keeping a
# real image here prevents routing evals from accepting a broken browser image.
_ONE_PIXEL_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360606060000000050001a5f645400000000049454e44"
    "ae426082"
)


def _personal_workspace() -> dict:
    return {
        "README.org": "Personal notes. Fitness in fitness.org, recipes in recipes.org.",
        "fitness.org": dedent("""\
            * Swimming
            ** 2026
            | date | dist (yd) | time | weight |
            |------+-----------+------+--------|
            | 6/1  |      3200 | 1h10m |  169.4 |
            | 6/8  |      3400 | 1h12m |  168.6 |
            | 6/15 |      3300 | 1h10m |  167.4 |
            | 6/24 |      3500 | 1h15m |  166.4 |
            | 7/1  |      3600 | 1h15m |  167.4 |
            | 7/23 |      3700 | 1h20m |  166.6 |
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
        return AgentHarness(create_agent(
            self.model, root, spec=prompt_rewrite_web_main_spec()))

    def create_sandbox_agent(self, filesystem: dict):
        """Production-shaped agent: real Docker execute + persistent sibling /tmp."""
        thread_root = tempfile.mkdtemp(prefix="render_graph_eval_")
        workspace = os.path.join(thread_root, "domain")
        os.mkdir(workspace)
        create_filesystem(workspace, filesystem)
        sandbox = SandboxManager.get_sandbox_backend(workspace)
        if sandbox is None:
            shutil.rmtree(thread_root)
            self.skipTest("Docker sandbox unavailable — is assist-sandbox built?")
        self.addCleanup(shutil.rmtree, thread_root, True)
        self.addCleanup(SandboxManager.cleanup, workspace)
        agent = AgentHarness(create_agent(
            self.model, workspace, sandbox_backend=sandbox,
            spec=AgentSpec(skill_sources=_web_skill_sources())))
        return agent, thread_root

    def _render_block_paths(self, agent, message_count: int = 0) -> list[str]:
        """Bodies of render blocks the AGENT emitted — only AIMessage content, so
        the loaded skill's own example blocks (a ToolMessage) don't count."""
        from langchain_core.messages import AIMessage
        bodies = []
        for m in agent.all_messages()[message_count:]:
            if not isinstance(m, AIMessage):
                continue
            content = m.content if isinstance(m.content, str) else ""
            bodies.extend(b for b in _RENDER_BLOCK.findall(content) if "type:" in b.lower())
        return bodies

    def _png_path(self, blocks: list[str], thread_root: str) -> str | None:
        """Host path named by the last PNG file render block."""
        for body in reversed(blocks):
            match = re.search(r"(?mi)^path:\s*(.+?\.png)\s*$", body)
            if not match:
                continue
            path = match.group(1)
            if path.startswith("/tmp/"):
                return os.path.join(thread_root, "tmp", path.removeprefix("/tmp/"))
            if path.startswith("/workspace/"):
                return os.path.join(thread_root, "domain",
                                    path.removeprefix("/workspace/"))
        return None

    def _assert_no_skill_file_detour(self, calls: list[dict]):
        """The named skill tool replaces filesystem discovery of SKILL.md."""
        detours = [
            call for call in calls
            if call.get("name") != "load_skill"
            and any(marker in str(call.get("args") or {}).lower()
                    for marker in ("skill.md", "/render-skill"))
        ]
        self.assertFalse(detours, f"render skill read as an ordinary file: {detours}")

    def _put_png_in_workspace(self, thread_root: str, name: str):
        path = os.path.join(thread_root, "domain", name)
        with open(path, "wb") as image:
            image.write(_ONE_PIXEL_PNG)

    def _assert_valid_png(self, blocks: list[str], thread_root: str):
        path = self._png_path(blocks, thread_root)
        self.assertIsNotNone(path, f"no PNG path in render blocks: {blocks}")
        self.assertTrue(os.path.isfile(path), f"rendered PNG does not exist: {path}")
        with open(path, "rb") as image:
            self.assertEqual(image.read(8), b"\x89PNG\r\n\x1a\n")

    def test_emits_render_block_for_named_file(self):
        """Example-thread shape: 'show me <named file>' in a realistic workspace
        emits a render block naming that file (not read+summarize)."""
        # Rendering a known local file is not a research eval.  Build inside the
        # stub because the graph captures the research worker at construction.
        with stub_research_subagent():
            agent = self.create_agent(_personal_workspace())
            agent.message("Show me the file with the name fitness.org")
            complete_web_main_tasks(agent)
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
        paths = []
        for call in agent_tool_calls(agent):
            if call.get("name") in ("write_file", "edit_file"):
                args = call.get("args") or call.get("arguments") or {}
                if isinstance(args, dict):
                    path = args.get("file_path") or args.get("path") or ""
                    if path:
                        paths.append(str(path))
        return paths

    def _render_skill_was_loaded(self, agent) -> bool:
        """True only for the load_skill tool contract, never a SKILL.md path read."""
        return any(
            (call.get("args") or {}).get("name") == "render"
            for call in agent_tool_calls(agent, "load_skill")
        )

    def test_creates_and_renders_requested_graph(self):
        """Production regression: asking to SEE a graph should load the web render
        skill and finish with an image render block, not create a PNG and summarize
        it or claim that a text-only model cannot display it."""
        agent, thread_root = self.create_sandbox_agent(_personal_workspace())
        agent.message("Can you show me a graph of how my weight has changed over the last 2 months?")
        blocks = self._render_block_paths(agent)
        self.assertTrue(self._render_skill_was_loaded(agent), "render skill should load")
        self.assertTrue(
            any("type: file" in b.lower() and ".png" in b.lower() for b in blocks),
            f"expected a file render block naming a PNG; render blocks: {blocks}",
        )
        self._assert_valid_png(blocks, thread_root)

    def test_render_existing_graph_followup_loads_skill(self):
        """Generality: an unrelated existing visual is placed in the conversation."""
        fs = dict(_personal_workspace())
        agent, thread_root = self.create_sandbox_agent(fs)
        self._put_png_in_workspace(thread_root, "weekly_laps.png")
        agent.message("Put the picture at /workspace/weekly_laps.png in this conversation.")
        blocks = self._render_block_paths(agent)
        self.assertTrue(self._render_skill_was_loaded(agent), "render skill should load")
        self.assertTrue(
            any("type: file" in b.lower() and "weekly_laps.png" in b for b in blocks),
            f"expected a file render block for weekly_laps.png; render blocks: {blocks}",
        )
        self._assert_valid_png(blocks, thread_root)

    def test_named_render_skill_request_uses_load_skill(self):
        """An explicit named-skill correction must call load_skill, not search for
        and read the mounted SKILL.md as an ordinary workspace file."""
        fs = dict(_personal_workspace())
        agent, thread_root = self.create_sandbox_agent(fs)
        self._put_png_in_workspace(thread_root, "pace_trend.png")
        before = len(agent_tool_calls(agent))
        agent.message("Use the render skill to put /workspace/pace_trend.png in the conversation.")
        calls = agent_tool_calls(agent)[before:]
        blocks = self._render_block_paths(agent)
        self.assertTrue(self._render_skill_was_loaded(agent), "render skill should load")
        self._assert_no_skill_file_detour(calls)
        self.assertTrue(
            any("type: file" in b.lower() and "pace_trend.png" in b for b in blocks),
            f"expected a file render block for pace_trend.png; render blocks: {blocks}",
        )
        self._assert_valid_png(blocks, thread_root)

    def test_graph_render_followup_sequence(self):
        """Production flow through resolution: create/show graph, then the terse
        render follow-up that originally failed. Stop once the request succeeds;
        standalone named-skill coverage pins the separate correction contract."""
        agent, thread_root = self.create_sandbox_agent(_personal_workspace())
        before_graph = len(agent.all_messages())
        agent.message("Can you show me a graph of how my weight has changed over the last 2 months?")
        graph_calls = agent_tool_calls(agent)
        graph_blocks = self._render_block_paths(agent, before_graph)

        before_render = len(agent.all_messages())
        agent.message("Please render the graph here.")
        render_followup_calls = agent_tool_calls(agent)[len(graph_calls):]
        blocks_after_render = self._render_block_paths(agent, before_render)

        loaded_initially = any(
            call.get("name") == "load_skill"
            and (call.get("args") or {}).get("name") == "render"
            for call in graph_calls
        )
        self.assertTrue(loaded_initially,
                        f"initial graph request did not load render: {graph_calls}")
        self.assertTrue(
            any("type: file" in block.lower() and ".png" in block.lower()
                for block in blocks_after_render),
            f"render follow-up emitted no new PNG block; calls: "
            f"{render_followup_calls}; blocks: {blocks_after_render}",
        )
        self._assert_no_skill_file_detour(render_followup_calls)
        self._assert_valid_png(graph_blocks, thread_root)
        self._assert_valid_png(blocks_after_render, thread_root)

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
