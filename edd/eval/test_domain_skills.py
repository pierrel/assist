"""End-to-end evals for in-repo *domain* skills.

A domain repo a thread operates in can ship its own skills at
``<working_dir>/.claude/skills/<name>/SKILL.md`` (the agent-agnostic
agentskills.io path, also read natively by Claude Code / the Agent SDK).
``create_agent`` auto-discovers that directory and the agent loads those
skills exactly like the built-in ones.  See
docs/2026-06-06-in-repo-domain-skills.org.

What these evals pin:

1. *Discovery + use, not just discovery.*  Mirroring the bar in
   ``test_skill_loading.py``, each model test asserts BOTH that the agent
   loaded the domain skill AND that the skill changed the output — the skill
   body mandates a sentinel tag the agent would never emit on its own, so the
   tag's presence is proof the rules were applied (loaded != used).

2. *Generalization, not lexical proximity* (per CLAUDE.md).  The fixture
   skill's ``description`` is written around the *shape* of the task (checking
   that a spreadsheet's rows add up to its stated total).  The prompts share no
   distinctive skill vocabulary — no "ledger", "audit", or the sentinel tag —
   so a pass means the small model matched the description to the task, not the
   prompt to the description.  Two implicitness rungs (domain hint vs.
   task-only) probe the match.

3. *Both filesystem and sandbox resolution.*  The no-sandbox tests exercise the
   local ``FilesystemBackend`` path; the sandbox test is non-redundant because
   the sandbox backend prefixes every path with ``work_dir`` (``/workspace``),
   so ``/.claude/skills/...`` resolves to ``/workspace/.claude/skills/...`` —
   the production path, exercised end-to-end only here.

4. *Agent-agnosticism is a tested contract, not a claim.*  The fixture
   ``SKILL.md`` (the very bytes the agent loads) is validated against the
   agentskills.io core schema, so it stays the same artifact Claude Code reads.
"""
import os
import re
import shutil
import subprocess
import tempfile
from textwrap import dedent
from unittest import TestCase

import yaml
from langchain_core.messages import AIMessage

from assist.agent import create_agent, AgentHarness
from assist.model_manager import select_assistant_model
from assist.sandbox_manager import SandboxManager

from .utils import create_filesystem


_SKILL_NAME = "ledger-audit"          # NOT a built-in name (no collision)
_SENTINEL = "[LEDGER-CHECK]"          # mandated by the skill body, absent from prompts

# Single source of truth for the fixture skill — the bytes the agent loads AND
# the bytes the agent-agnostic conformance test validates.  Frontmatter is the
# open-standard core only (name + description), so the same file is valid for
# Claude Code / the Agent SDK unchanged.
_SKILL_MD = dedent("""\
    ---
    name: ledger-audit
    description: Confirming that the itemized rows in a billing or expense spreadsheet add up to its stated grand total, and pinpointing the line that throws the sum off.
    ---

    # Ledger audit

    When you are asked to check whether a spreadsheet's line items match its
    stated total, follow this procedure:

    1. Begin your reply with the exact tag `[LEDGER-CHECK]` on its own first
       line. It records that this audit procedure was applied.
    2. Add up the value column across every itemized row yourself.
    3. Compare your sum against the file's stated total row.
    4. Name the specific row whose value makes the two disagree, and state the
       size of the gap.
    """)

# Rows sum to 205.00; the stated TOTAL says 200.00 — off by 5.00.  The exact
# arithmetic is NOT what these tests assert (that would be the calculate
# skill's job); the sentinel tag is the behavior-change signal.
_CSV = dedent("""\
    item,amount
    Hosting,120.00
    Domains,40.00
    Email,15.00
    Storage,30.00
    TOTAL,200.00
    """)


def _fixture() -> dict:
    """A domain working tree: the in-repo skill plus the file the prompt asks
    about."""
    return {
        ".claude": {"skills": {_SKILL_NAME: {"SKILL.md": _SKILL_MD}}},
        "billing-2026.csv": _CSV,
    }


def _skill_was_loaded(agent, skill_name: str) -> bool:
    """True iff a tool call loaded the named skill's body.

    Recognizes ``load_skill(name=skill_name)`` (the small-model tool) and the
    upstream ``read_file`` path containing ``/skills/<name>/``.  Local to this
    suite, mirroring the other skill evals so they can drift independently.
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


def _cleanup_workspace(path: str) -> None:
    """Remove a sandbox workspace, using Docker to delete root-owned files.

    Mirrors the helper in test_calculate_skill.py / test_dev_agent.py.
    """
    try:
        subprocess.run(
            ['docker', 'run', '--rm', '-v', f'{path}:/cleanup',
             'alpine', 'sh', '-c',
             'chmod -R 777 /cleanup 2>/dev/null; rm -rf /cleanup/*'],
            check=False, timeout=60,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
    shutil.rmtree(path, ignore_errors=True)


class TestDomainSkillFrontmatterIsAgentAgnostic(TestCase):
    """No model / no Docker — guards that the fixture skill stays a portable,
    open-standard artifact (the contract behind "Claude Code can leverage it").
    """

    def test_fixture_uses_open_standard_core_only(self):
        frontmatter = yaml.safe_load(_SKILL_MD.split("---")[1])
        # Open-standard core keys only — no assist-specific frontmatter that
        # would make the file non-portable to Claude Code / the Agent SDK.
        core = {"name", "description", "license", "compatibility",
                "metadata", "allowed-tools"}
        self.assertTrue(
            set(frontmatter) <= core,
            f"non-open-standard frontmatter keys: {set(frontmatter) - core}",
        )
        # agentskills.io constraints: name == dir name, ≤64, lowercase-alnum
        # with single hyphens; description ≤1024.
        self.assertEqual(frontmatter["name"], _SKILL_NAME)
        self.assertLessEqual(len(frontmatter["name"]), 64)
        self.assertRegex(frontmatter["name"], r"^[a-z0-9]+(-[a-z0-9]+)*$")
        self.assertLessEqual(len(frontmatter["description"]), 1024)


class TestDomainSkillLoadingLocal(TestCase):
    """No-sandbox rung: domain skill discovered via the local FilesystemBackend.
    Asserts the skill loaded AND its mandated sentinel reached the output.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def _make_agent(self):
        root = tempfile.mkdtemp(prefix="domain_skill_eval_")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        create_filesystem(root, _fixture())
        return AgentHarness(create_agent(self.model, root))

    def test_loads_with_domain_hint(self):
        """Implicitness rung 1 — the failure mode is described, but no skill
        vocabulary and no sentinel."""
        agent = self._make_agent()
        response = agent.message(
            "I exported `billing-2026.csv`. The individual charges are "
            "supposed to sum to the TOTAL line at the bottom, but the books "
            "don't balance — one of the rows must be wrong. Which one is off?"
        )
        self.assertTrue(
            _skill_was_loaded(agent, _SKILL_NAME),
            "agent did not load the in-repo ledger-audit skill despite the "
            "prompt describing exactly the task its description covers",
        )
        self.assertIn(
            _SENTINEL, response,
            "agent loaded the skill but did not apply it — the mandated "
            f"{_SENTINEL} tag is absent, so loaded != used",
        )

    def test_loads_task_only(self):
        """Implicitness rung 2 — just the file and a vague ask; the agent must
        match the skill description to the file's shape on its own."""
        agent = self._make_agent()
        response = agent.message(
            "Take a look at `billing-2026.csv` and tell me if anything's off."
        )
        self.assertTrue(
            _skill_was_loaded(agent, _SKILL_NAME),
            "agent did not load the in-repo ledger-audit skill from the "
            "task alone",
        )
        self.assertIn(
            _SENTINEL, response,
            f"skill loaded but mandated {_SENTINEL} tag absent (loaded != used)",
        )


class TestDomainSkillLoadingSandbox(TestCase):
    """Sandbox rung (production path): the sandbox backend prefixes paths with
    ``work_dir``, so this is the only test that proves
    ``/.claude/skills/...`` -> ``/workspace/.claude/skills/...`` resolves
    end-to-end.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def setUp(self):
        self.workspace = tempfile.mkdtemp(prefix="domain_skill_sandbox_eval_")
        self.sandbox = SandboxManager.get_sandbox_backend(self.workspace)
        if self.sandbox is None:
            self.skipTest(
                "Docker sandbox unavailable — is Docker running and "
                "assist-sandbox built?"
            )

    def tearDown(self):
        SandboxManager.cleanup(self.workspace)
        _cleanup_workspace(self.workspace)

    def test_loads_with_domain_hint_in_sandbox(self):
        create_filesystem(self.workspace, _fixture())
        agent = AgentHarness(create_agent(
            self.model, self.workspace, sandbox_backend=self.sandbox))
        response = agent.message(
            "I exported `billing-2026.csv`. The individual charges are "
            "supposed to sum to the TOTAL line at the bottom, but the books "
            "don't balance — one of the rows must be wrong. Which one is off?"
        )
        self.assertTrue(
            _skill_was_loaded(agent, _SKILL_NAME),
            "agent did not load the in-repo ledger-audit skill inside the "
            "sandbox (path-prefix resolution may be broken)",
        )
        self.assertIn(
            _SENTINEL, response,
            f"skill loaded but mandated {_SENTINEL} tag absent (loaded != used)",
        )
