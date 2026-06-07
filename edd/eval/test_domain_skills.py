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
   loaded the domain skill AND that it carried out the procedure the skill
   describes — it reports the reconciled figure (210.00), which appears nowhere
   in the fixture file (the file shows the parts and a *wrong* total), so its
   presence proves the agent actually added the entries rather than echoing a
   number.  (An earlier draft mandated a sentinel tag instead; the small model
   reliably did the reconciliation work but unreliably emitted the cosmetic
   tag — the AGENTS.md "guidance is unreliable for this model" shape — so the
   derived figure is the stabler, more honest "used" signal.)

2. *Generalization, not lexical proximity* (per CLAUDE.md).  The fixture
   skill's ``description`` is written around the *shape* of the task
   (reconciling the parts of a spreadsheet to its stated bottom-line figure).
   The prompts deliberately avoid the description's own verbs ("reconcile",
   "sum", "total") and the skill's identity ("ledger", "audit") — so a pass
   means the small model mapped task-shape to the description, not prompt-words
   to description-words.  Two implicitness rungs (domain hint vs.
   a reconciliation question) probe the match, and an anti-test pins the
   description against firing on an unrelated task over the same file.

3. *Both filesystem and sandbox resolution.*  The no-sandbox tests exercise the
   local ``FilesystemBackend`` path; the sandbox test is non-redundant because
   the sandbox backend prefixes every path with ``work_dir`` (``/workspace``),
   so ``/.claude/skills/...`` resolves to ``/workspace/.claude/skills/...`` —
   the production path, exercised end-to-end only here.

4. *Agent-agnosticism is a tested contract, not a claim.*  The fixture
   ``SKILL.md`` (the very bytes the agent loads) is validated against the
   agentskills.io core schema, so it stays the same artifact Claude Code reads.
"""
import shutil
import tempfile
from textwrap import dedent
from unittest import TestCase

import yaml

from assist.agent import create_agent, AgentHarness
from assist.model_manager import select_assistant_model
from assist.sandbox_manager import SandboxManager

from .utils import create_filesystem, skill_was_loaded, cleanup_workspace


_SKILL_NAME = "ledger-audit"          # NOT a built-in name (no collision)

# Single source of truth for the fixture skill — the bytes the agent loads AND
# the bytes the agent-agnostic conformance test validates.  Frontmatter is the
# open-standard core only (name + description), so the same file is valid for
# Claude Code / the Agent SDK unchanged.
_SKILL_MD = dedent("""\
    ---
    name: ledger-audit
    description: For a spreadsheet of expenses or charges where a bottom-line figure should equal the parts above it — verify the parts actually reconcile to that figure, and identify the single entry responsible when they don't.
    ---

    # Ledger audit

    When the parts of a spreadsheet are supposed to match a stated bottom-line
    figure, follow this procedure:

    1. Add up the amount column across every entry yourself.
    2. Compare your computed figure against the spreadsheet's stated bottom
       line, and state both numbers explicitly.
    3. State the size of the gap between them.
    """)

# The entries add up to 210.00 but the stated TOTAL says 190.00 — a gap of
# 20.00, with 210.00 the correct figure.  Round numbers keep the arithmetic
# trivial for the small model.  210.00 appears NOWHERE in the file, so the
# agent reporting it is proof it actually summed the entries (did the work).
_CSV = dedent("""\
    item,amount
    Hosting,100.00
    Domains,50.00
    Email,20.00
    Storage,40.00
    TOTAL,190.00
    """)
_CORRECT_SUM = "210"  # entries reconcile to 210.00; the stated 190.00 is wrong
# Digit-boundary match so a wrong larger number that contains "210" (e.g. 2100)
# can't false-pass a plain substring check; tolerates "210"/"210.0"/"210.00".
_CORRECT_SUM_RE = r"\b210(?:\.\d{1,2})?\b"


def _fixture() -> dict:
    """A domain working tree: the in-repo skill plus the file the prompt asks
    about."""
    return {
        ".claude": {"skills": {_SKILL_NAME: {"SKILL.md": _SKILL_MD}}},
        "billing-2026.csv": _CSV,
    }


class TestDomainSkillFrontmatterIsAgentAgnostic(TestCase):
    """No model / no Docker — guards that the fixture skill stays a portable,
    open-standard artifact (the contract behind "Claude Code can leverage it").
    """

    def test_fixture_uses_open_standard_core_only(self):
        # Limit to the first two delimiters so a stray `---` in the body
        # (e.g. a horizontal rule) can't shift the parsed segment.
        frontmatter = yaml.safe_load(_SKILL_MD.split("---", 2)[1])
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
    Asserts the skill loaded AND that the agent did the reconciliation (the
    derived figure, absent from the file, reaches the output).
    """

    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def _make_agent(self):
        root = tempfile.mkdtemp(prefix="domain_skill_eval_")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        create_filesystem(root, _fixture())
        return AgentHarness(create_agent(self.model, root))

    def _assert_loaded_and_applied(self, agent, response):
        """Loaded AND used: the domain skill was discovered + loaded, AND the
        agent carried out the reconciliation it describes — it reports the
        derived figure (210.00, absent from the file), proving it summed the
        entries rather than echoing a stated number or returning a stub."""
        self.assertTrue(
            skill_was_loaded(agent, _SKILL_NAME),
            "agent did not load the in-repo ledger-audit skill",
        )
        self.assertRegex(
            response, _CORRECT_SUM_RE,
            f"skill loaded but the reconciliation wasn't done — the derived "
            f"figure {_CORRECT_SUM}.00 is absent from the reply (loaded != used)",
        )

    def test_loads_with_domain_hint(self):
        """Implicitness rung 1 — the failure mode is described, but no skill
        vocabulary, no description verbs, and no sentinel."""
        agent = self._make_agent()
        response = agent.message(
            "I exported `billing-2026.csv`. The bottom line is supposed to "
            "match the charges above it, but the books don't balance — one "
            "entry must be wrong. Which one?"
        )
        self._assert_loaded_and_applied(agent, response)

    def test_loads_task_only(self):
        """Implicitness rung 2 — a reconciliation question with no skill
        vocabulary; the agent must map it to the skill description itself."""
        agent = self._make_agent()
        response = agent.message(
            "Take a look at `billing-2026.csv` — does the bottom-line figure "
            "actually match the entries above it?"
        )
        self._assert_loaded_and_applied(agent, response)

    def test_does_not_load_on_unrelated_file_task(self):
        """Anti-test — a non-reconciliation task over the SAME file must NOT
        trip the skill, so its description isn't merely firing on any mention
        of a billing spreadsheet (pins it against loose drift)."""
        agent = self._make_agent()
        agent.message(
            "Rename `billing-2026.csv` to `invoices.csv` for me."
        )
        self.assertFalse(
            skill_was_loaded(agent, _SKILL_NAME),
            "ledger-audit loaded on a pure file-rename task — its description "
            "is firing on the billing file rather than the reconciliation task",
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
        cleanup_workspace(self.workspace)

    def test_loads_with_domain_hint_in_sandbox(self):
        create_filesystem(self.workspace, _fixture())
        agent = AgentHarness(create_agent(
            self.model, self.workspace, sandbox_backend=self.sandbox))
        response = agent.message(
            "I exported `billing-2026.csv`. The bottom line is supposed to "
            "match the charges above it, but the books don't balance — one "
            "entry must be wrong. Which one?"
        )
        # Same loaded-and-applied bar as the local rung; the sandbox variant's
        # job is to prove /.claude/skills/ -> /workspace/.claude/skills/
        # resolution, so it shares the assertions rather than redefining them.
        self.assertTrue(
            skill_was_loaded(agent, _SKILL_NAME),
            "agent did not load the in-repo ledger-audit skill inside the "
            "sandbox (path-prefix resolution may be broken)",
        )
        self.assertRegex(
            response, _CORRECT_SUM_RE,
            f"skill loaded but reconciliation not done — derived figure "
            f"{_CORRECT_SUM}.00 absent (loaded != used)",
        )
