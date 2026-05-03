"""Skill-loading + behavior evals for the (planned) ``calculate`` skill.

The calculate skill is for throwaway, answer-the-question Python work:
arithmetic, statistics, simulations, financial projections.  It is
deliberately distinct from the ``dev`` skill — there is no plan/red/green
ceremony, the goal is just to give the user a correct answer.  Code
should still be tested (the agent must trust its own answer), and any
script left behind should land somewhere sensible in the workspace
(typically ``scripts/``); ad-hoc one-off scripts should be cleaned up.

These evals run the general agent inside a Docker sandbox — ``calculate``
needs a real ``execute`` for Python — and walk a ladder of implicitness:

1. ``test_explicit_calculate_request`` — user names the skill.
2. ``test_compound_growth_calculation`` — multi-step financial math, no
   skill word.  Implicit-loading rung.
3. ``test_throwaway_cleanup`` — checks the skill's "don't litter" rule
   on a stdev question.
4. ``test_does_not_load_on_non_math_prompt`` — anti-test.  Off-topic
   prompt; calculate must not load.
5. ``test_does_not_load_on_planning_prompt`` — anti-test, narrower.
   A finance-adjacent prompt that does not ask for a number; pins the
   description against drift in the loose direction (e.g. a stray
   trigger word like ``retirement`` firing on "thinking about
   retirement").

Each math test asserts: (a) the agent loaded the calculate skill,
(b) the response contains a numerically-correct answer.  The two
anti-tests assert calculate did not load.  Test 3 adds a side-channel
whole-tree cleanup check.
"""
import os
import re
import subprocess
import tempfile
import shutil

from unittest import TestCase

from langchain_core.messages import AIMessage, ToolMessage

from assist.agent import create_agent, AgentHarness
from assist.model_manager import select_chat_model
from assist.sandbox_manager import SandboxManager

from .utils import create_filesystem


def _cleanup_workspace(path: str) -> None:
    """Remove workspace directory, using Docker to delete root-owned files.

    Mirrors the helper in test_dev_agent.py.  Sandbox commands (pip,
    pytest, etc.) write files as root inside the bind mount; plain
    shutil.rmtree fails on those without an intermediate chmod.
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


class TestCalculateSkill(TestCase):
    """Evals for the calculate skill.

    Each test gets a fresh workspace and sandbox container so any files
    the agent leaves behind (or fails to clean up) are scoped to one
    test only.  The sandbox is required: the calculate skill is supposed
    to make the agent run real Python and check its own answer, which
    is impossible without a working ``execute`` tool.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = select_chat_model(0.1)

    def setUp(self):
        self.workspace = tempfile.mkdtemp(prefix="calculate_skill_eval_")
        self.sandbox = SandboxManager.get_sandbox_backend(self.workspace)
        if self.sandbox is None:
            self.skipTest(
                "Docker sandbox unavailable — is Docker running and "
                "assist-sandbox built?"
            )

    def tearDown(self):
        SandboxManager.cleanup(self.workspace)
        _cleanup_workspace(self.workspace)

    # ------------------------------------------------------------------
    # Helpers — small subset of test_dev_agent.py's inspection helpers,
    # plus a calculate-specific skill-loaded probe shared with the org
    # tests.
    # ------------------------------------------------------------------

    def _create_agent(self, filesystem: dict | None = None):
        if filesystem:
            create_filesystem(self.workspace, filesystem)
        return AgentHarness(create_agent(
            self.model,
            self.workspace,
            sandbox_backend=self.sandbox,
        ))

    def _skill_was_loaded(self, agent, skill_name: str) -> bool:
        """True iff a tool call loaded the named skill's body.

        Recognizes both routes the SkillsMiddleware exposes:

        - ``load_skill(name=skill_name)`` — the small-model tool we
          register in ``SmallModelSkillsMiddleware``.
        - ``read_file`` / ``read`` with a path containing
          ``/skills/<skill_name>/`` — the upstream deepagents path.

        Mirrors ``_skill_was_loaded`` in test_skill_loading.py — kept
        local so the two suites can drift independently.
        """
        path_needle = f"/skills/{skill_name}/"
        for m in agent.all_messages():
            if not isinstance(m, AIMessage) or not m.tool_calls:
                continue
            for tc in m.tool_calls:
                args = tc.get("args") or {}
                if (tc.get("name") == "load_skill"
                        and args.get("name") == skill_name):
                    return True
                for v in args.values():
                    if isinstance(v, str) and path_needle in v:
                        return True
        return False

    def _executed_commands(self, agent) -> list[str]:
        """Command strings from every ``execute`` tool call."""
        commands = []
        for m in agent.all_messages():
            if not isinstance(m, AIMessage) or not m.tool_calls:
                continue
            for tc in m.tool_calls:
                if tc.get("name") == "execute":
                    args = tc.get("args") or {}
                    cmd = args.get("command", "")
                    if cmd:
                        commands.append(cmd)
        return commands

    def _ran_python(self, agent) -> bool:
        """True iff at least one execute call ran Python.

        Catches ``python``/``python3`` invocations, ``-c``-style inline
        scripts, and ``script.py`` invocations.  Does NOT count merely
        writing a .py file — the calculate skill's contract is that the
        agent runs the code, not just authors it.
        """
        for cmd in self._executed_commands(agent):
            if re.search(r"\bpython3?\b", cmd):
                return True
            if re.search(r"\b\w+\.py\b", cmd):
                return True
        return False

    def _written_py_paths(self, agent) -> list[str]:
        """Workspace-relative paths of ``.py`` files written by the agent.

        Used by ``test_throwaway_cleanup`` to confirm the skill's "don't
        litter the workspace root with one-off scripts" rule.
        """
        paths = []
        for m in agent.all_messages():
            if not isinstance(m, AIMessage) or not m.tool_calls:
                continue
            for tc in m.tool_calls:
                if tc.get("name") not in ("write_file", "write"):
                    continue
                args = tc.get("args") or {}
                p = args.get("file_path") or args.get("path") or ""
                if p.endswith(".py"):
                    paths.append(p)
        return paths

    # ------------------------------------------------------------------
    # Ladder — least to most implicit.
    # ------------------------------------------------------------------

    def test_compound_growth_calculation(self):
        """Implicitness 1 — multi-step financial arithmetic, no math words.

        Compound interest at 6.5% APR compounded monthly on $10,000 for
        25 years: 10000 * (1 + 0.065/12)**(12*25) ≈ $50,495.42.  We
        only consider dollar-formatted candidates (``$50,495.42`` /
        ``$50495.42``) — the bare-number regex was too eager and would
        pick up arbitrary tokens (years, interest rate) that happened
        to fall near the truth at high slack.  1% tolerance absorbs
        minor rounding without admitting unrelated numbers.

        Failure modes this catches:

        - Skill not loaded → agent guesses a number with the wrong
          formula (simple interest, wrong compounding period).
        - Skill loaded but execute not run → small model sometimes
          outputs a confident-looking but wrong figure.
        """
        agent = self._create_agent()

        res = agent.message(
            "If I invest $10,000 at 6.5% annual return compounded "
            "monthly, what is the balance after 25 years?"
        )

        self.assertTrue(
            self._skill_was_loaded(agent, "calculate"),
            "Agent did not load the calculate skill for a compound-"
            "interest projection.",
        )
        self.assertTrue(
            self._ran_python(agent),
            "Multi-step financial math should be checked by running "
            "Python — eyeballed answers are not acceptable.",
        )

        # Match only $-prefixed numbers; the bare-number sweep was too
        # noisy (any 25, 6.5, 12, etc. floated in).  Excludes the seed
        # $10,000 from the prompt so we don't false-pass on echo-back.
        truth = 10000 * (1 + 0.065 / 12) ** (12 * 25)  # ≈ 50495.42
        dollar_tokens = re.findall(
            r"\$\s?([\d]{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)", res
        )
        nums = []
        for tok in dollar_tokens:
            try:
                v = float(tok.replace(",", ""))
            except ValueError:
                continue
            if v == 10000:  # the seed echoed back; don't count it.
                continue
            nums.append(v)
        self.assertTrue(
            any(abs(n - truth) / truth < 0.01 for n in nums),
            f"Response should contain a $-formatted balance close to "
            f"${truth:,.2f}.  Dollar amounts found: {nums}.  "
            f"Response: {res[:500]}",
        )

    def test_throwaway_cleanup(self):
        """Implicitness 2 + side-channel rule check.

        Standard deviation of [12, 15, 18, 22, 30, 45, 67, 89, 102]:
        - population stdev ≈ 31.79
        - sample stdev ≈ 33.71
        Tight tolerance (1%) — both ``statistics.stdev`` and
        ``statistics.pstdev`` return exact values, so any agent that
        actually runs Python can hit either of these to two decimals.
        A wide slack here would let an eyeballed answer slip through.

        Side-channel: the skill's "throwaway cleanup" rule says one-off
        scripts should be cleaned up after the answer is delivered.
        A single stdev question has no reason to leave a script behind
        anywhere in the workspace.  We walk the whole tree, not just
        the root: an agent that hides ``scripts/calc.py`` is failing
        the same rule as one that drops ``calc.py`` at the top.

        If, in the future, we want a "kept-code" sub-rule (skill says:
        save under ``scripts/`` only when the user would benefit), the
        test prompt should make that benefit explicit; this prompt
        does not.
        """
        agent = self._create_agent()

        res = agent.message(
            "What is the standard deviation of "
            "12, 15, 18, 22, 30, 45, 67, 89, and 102?"
        )

        self.assertTrue(
            self._skill_was_loaded(agent, "calculate"),
            "Agent did not load the calculate skill for a stdev question.",
        )
        self.assertTrue(
            self._ran_python(agent),
            "Stdev should be computed by running Python (statistics or "
            "numpy), not estimated by eye.",
        )

        # Tight tolerance: both pstdev and stdev are exact, so 1% is
        # plenty of slack and keeps eyeballed answers out.
        candidates = (31.79, 33.71)
        nums = [float(x.replace(",", ""))
                for x in re.findall(r"\d+(?:\.\d+)?", res)]
        self.assertTrue(
            any(any(abs(n - c) / c < 0.01 for c in candidates) for n in nums),
            f"Response should contain a stdev near {candidates}. "
            f"Numbers found: {nums}.  Response: {res[:500]}",
        )

        # Cleanup: walk the whole workspace.  No throwaway script
        # for a single stdev question should survive — anywhere.
        leftover = []
        for dirpath, _dirnames, filenames in os.walk(self.workspace):
            for fname in filenames:
                if fname.endswith(".py"):
                    rel = os.path.relpath(
                        os.path.join(dirpath, fname), self.workspace,
                    )
                    leftover.append(rel)

        self.assertEqual(
            leftover, [],
            "Throwaway scripts should be cleaned up.  This is a one-off "
            "stdev question — there is no reason for any .py file to "
            f"remain anywhere in the workspace.  Found: {leftover}.",
        )

    def test_does_not_load_on_non_math_prompt(self):
        """Anti-test — calculate must NOT load when there is no math.

        Pins the description against drift in the loose direction.  If
        a future SKILL.md change makes "calculate" fire on every prompt
        (e.g. the description picks up generic words like "amount" or
        "estimate"), this test catches it before the loose description
        ships and starts costing latency on unrelated turns.

        The prompt is purely informational — no numbers, no projection
        request, no quantification — so the agent should answer it from
        local context / research, never from calculate.
        """
        agent = self._create_agent()

        agent.message("What is the capital of France?")

        self.assertFalse(
            self._skill_was_loaded(agent, "calculate"),
            "Calculate skill loaded on a non-math prompt — the "
            "description is too aggressive.  Tighten its trigger words "
            "so it stays scoped to computation, not general questions.",
        )

    def test_does_not_load_on_planning_prompt(self):
        """Anti-test — narrower.  Calculate must NOT load when the prompt
        contains a finance-adjacent trigger word but does not actually
        ask for a number.

        The first anti-test ("capital of France") is fully off-topic and
        easy to keep negative.  This one is closer to the trigger margin:
        the prompt mentions "retirement", which is a word the description
        explicitly avoided as a bare trigger because it false-fires on
        prompts like this.  If a future loosening of the description
        adds bare ``retirement`` (or ``invest``, ``balance``, etc.), this
        test catches the regression before the loose description ships
        and starts costing latency on unrelated turns.
        """
        agent = self._create_agent()

        agent.message(
            "I'm thinking about retirement — what should I be considering?"
        )

        self.assertFalse(
            self._skill_was_loaded(agent, "calculate"),
            "Calculate skill loaded on a finance-adjacent planning prompt "
            "with no numeric question.  The MUST-load clause is keyed on "
            "'answer involves a number' precisely to keep this case "
            "negative; check the trigger word list for over-broad tokens "
            "(retirement, invest, balance, growth, interest).",
        )
