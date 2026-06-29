"""Eval: the `time` skill makes the agent answer date questions with `date`.

Real-LLM eval (small model), inside a Docker sandbox — the time skill's whole
premise is that the agent runs the `date` command via `execute` instead of
guessing the day/date from memory. Each test asserts: (a) the time skill loaded,
(b) the agent actually ran `date`, (c) the answer is right. Correct answers are
derived from `date` itself (not hard-coded) so the eval is year-independent.

Prompts deliberately avoid the SKILL.md EXAMPLES verbatim (probe generalization).
"""
import re
import tempfile
from unittest import TestCase

from assist.agent import create_agent, AgentHarness
from assist.model_manager import select_assistant_model
from assist.sandbox_manager import SandboxManager

from .utils import create_filesystem, skill_was_loaded, executed_commands, cleanup_workspace


class TestTimeAgent(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def setUp(self):
        self.workspace = tempfile.mkdtemp(prefix="time_skill_eval_")
        self.sandbox = SandboxManager.get_sandbox_backend(self.workspace)
        if self.sandbox is None:
            self.skipTest("Docker sandbox unavailable — is Docker running and "
                          "assist-sandbox built?")

    def tearDown(self):
        SandboxManager.cleanup(self.workspace)
        cleanup_workspace(self.workspace)

    def _agent(self):
        return AgentHarness(create_agent(self.model, self.workspace,
                                         sandbox_backend=self.sandbox))

    def _ran_date(self, agent) -> bool:
        # `date` at command position — NOT `datetime.date` (the . is a \b, which the
        # calculate-skill Python habit could otherwise sneak past this proxy).
        return any(re.search(r"(?:^|[\s;&|()])date\b", cmd)
                   for cmd in executed_commands(agent))

    def _weekday(self, expr: str) -> str:
        """The weekday `date -d <expr>` resolves to IN THE SANDBOX — same clock and
        TZ the agent uses. Fail fast on empty output (else assertIn('', reply) would
        vacuously pass and hide a broken `date`)."""
        out = (self.sandbox.execute(f"date -d '{expr}' +%A").output or "").strip()
        self.assertTrue(out, f"sandbox `date -d {expr}` produced no output")
        return out

    def test_weekday_of_a_date(self):
        agent = self._agent()
        reply = str(agent.message("What day of the week does the 5th of July land on?") or "").lower()
        self.assertTrue(skill_was_loaded(agent, "time"), "time skill should load")
        self.assertTrue(self._ran_date(agent), "agent should run the date command")
        self.assertIn(self._weekday("7/5").lower(), reply)

    def test_relative_date(self):
        agent = self._agent()
        reply = str(agent.message("What's the date on the upcoming Thursday?") or "").lower()
        self.assertTrue(skill_was_loaded(agent, "time"), "time skill should load")
        self.assertTrue(self._ran_date(agent), "agent should run the date command")
        self.assertIn("thursday", reply)   # next Thursday is a Thursday

    def test_today(self):
        agent = self._agent()
        reply = str(agent.message("Remind me what the date is right now.") or "").lower()
        self.assertTrue(skill_was_loaded(agent, "time"), "time skill should load")
        self.assertTrue(self._ran_date(agent), "agent should run the date command")
        self.assertIn(self._weekday("today").lower(), reply)

    def test_does_not_load_on_non_date_prompt(self):
        # Anti-test: an off-topic prompt must not trip the time skill.
        agent = self._agent()
        agent.message("Write a haiku about the ocean.")
        self.assertFalse(skill_was_loaded(agent, "time"),
                         "time skill should NOT load for a non-date prompt")
