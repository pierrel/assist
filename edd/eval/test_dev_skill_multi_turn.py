"""Multi-turn dev evals — the canary for the dev-agent → dev-skill migration.

These tests target capabilities that are *architecturally blocked* by the
deepagents subagent model (each `task` call resets messages to the
description; no cross-call state preservation), and *should pass* once the
general agent loads the `dev` skill and handles code work in its own
conversation thread.

Two scenarios:
1. Multi-turn TDD with explicit approvals between phases.
2. Mid-task clarification — user adds requirements while the agent is
   working.

The ASSERTIONS focus on *context preservation*, not on dev-correctness — we
care that turn N+1 has access to the conversation from turn N. The dev
skill takes care of the workflow inside each turn.
"""
import os
import tempfile
import shutil

from unittest import TestCase

from assist.thread import ThreadManager
from assist.sandbox_manager import SandboxManager


class TestDevSkillMultiTurn(TestCase):
    """Multi-turn dev work via the general agent + dev skill."""

    def setUp(self):
        self.workspace = tempfile.mkdtemp()
        # Project indicator so the general agent's prompt pre-loads the dev
        # skill body (small-model reliability path).
        with open(os.path.join(self.workspace, "pyproject.toml"), "w") as f:
            f.write('[project]\nname = "test-project"\n')
        self.sandbox = SandboxManager.get_sandbox_backend(self.workspace)
        if self.sandbox is None:
            self.skipTest("Docker sandbox unavailable — is Docker running and assist-sandbox built?")
        self.thread_manager = ThreadManager(self.workspace)

    def tearDown(self):
        SandboxManager.cleanup(self.workspace)
        if os.path.exists(self.workspace):
            shutil.rmtree(self.workspace, ignore_errors=True)

    def _new_thread(self):
        return self.thread_manager.new(sandbox_backend=self.sandbox)

    def test_multi_turn_tdd_with_approvals(self):
        """Three user turns — feature request, approve plan, approve tests.

        With the subagent architecture, this scenario is broken in two
        ways: each task call resets state, AND the dev-agent collapses
        all phases into one task to work around statelessness. With the
        general agent loading the dev skill, the conversation thread
        carries context across turns and the agent can honour the
        explicit pause points in the workflow.
        """
        thread = self._new_thread()

        # Turn 1: feature request. Agent should produce a plan and pause.
        resp1 = thread.message(
            "Add an `add(a, b)` function to a new file `calculator.py` "
            "in the workspace, with a unit test. Use the TDD workflow."
        )
        # We don't strictly assert the agent stops here — just capture
        # response shape. The harder assertion is below: turn 2 must have
        # context from turn 1.

        # Turn 2: approve. Agent should know to continue TDD without
        # being re-told about calculator.py / add().
        resp2 = thread.message("approved, please proceed")
        self.assertRegex(
            resp2.lower(),
            r"(calculator|add|test|implementation)",
            "Turn 2 should reference the work from turn 1 — context must be preserved. "
            f"Turn 2 response: {resp2[:500]}"
        )

        # Turn 3: continue.
        resp3 = thread.message("looks good, continue")
        # By turn 3, calculator.py should exist OR the agent should have
        # written tests. Either is evidence the multi-turn flow worked.
        ws_files = []
        for r, d, fs in os.walk(self.workspace):
            for f in fs:
                if f.endswith(".py") or f.endswith(".md"):
                    ws_files.append(os.path.relpath(os.path.join(r, f), self.workspace))
        self.assertTrue(
            any("calculator" in p or "test" in p.lower() or "plan" in p.lower() for p in ws_files),
            f"After 3 turns, expected calculator.py / a test file / a plan file in the workspace. "
            f"Found: {ws_files[:20]}"
        )

    def test_mid_task_clarification(self):
        """User adds a requirement mid-task. Agent should incorporate it.

        With the subagent architecture, turn 2 starts a fresh dev-agent
        with no idea what 'add' was — only 'also add subtract' as
        context. With the skill on the general agent, turn 2 lands in
        the same conversation and the agent picks up where it left off.

        Assertion targets the SANDBOX FILE STATE rather than response
        text — the agent's narration is unreliable on the small model
        (it sometimes summarises tersely with "I saved the output...").
        File state is what we care about: did both functions land in
        calculator.py?
        """
        thread = self._new_thread()

        # Turn 1: start the task.
        thread.message(
            "Create `calculator.py` in the workspace with an `add(a, b)` "
            "function. Use TDD."
        )

        # Turn 2: add a requirement. We do NOT re-explain the original
        # task; turn 2 must rely on turn-1 context.
        thread.message(
            "Actually, while you're at it, also add a `subtract(a, b)` function."
        )

        # File-state assertion: read calculator.py from the sandbox.
        result = self.sandbox.execute("cat /workspace/calculator.py")
        content = result.output if hasattr(result, "output") else str(result)

        self.assertIn(
            "def add",
            content,
            f"calculator.py must contain `def add` from turn 1's request. "
            f"File contents: {content[:500]}"
        )
        self.assertIn(
            "def subtract",
            content,
            f"calculator.py must contain `def subtract` from turn 2's "
            f"clarification — proves context from turn 1 carried into "
            f"turn 2 (otherwise the agent would not know about the "
            f"calculator.py file at all). File contents: {content[:500]}"
        )
