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

from .utils import skill_was_loaded, executed_commands, cleanup_workspace


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

    def _date_parts(self, expr: str):
        """(weekday, month, day) that `date -d <expr>` resolves to IN THE SANDBOX —
        same clock/TZ the agent uses. Fail fast on empty (an assertIn against an
        empty string would vacuously pass)."""
        out = (self.sandbox.execute(f"date -d '{expr}' '+%A|%B|%-d'").output or "").strip()
        self.assertIn("|", out, f"sandbox `date -d {expr}` produced no usable output")
        wk, mon, day = out.split("|")
        return wk.lower(), mon.lower(), day

    def _asserts_full_date(self, reply, expr):
        """Reply names the resolved weekday AND month AND day — i.e. an actual
        date, not just a weekday (the prompt asks for the date)."""
        wk, mon, day = self._date_parts(expr)
        self.assertIn(wk, reply)
        self.assertIn(mon, reply)
        self.assertRegex(reply, rf"\b{day}\b")   # day number, word-bounded (not '2' in '2026')

    def test_weekday_of_a_date(self):
        agent = self._agent()
        reply = str(agent.message("What day of the week does the 5th of July land on?") or "").lower()
        self.assertTrue(skill_was_loaded(agent, "time"), "time skill should load")
        self.assertTrue(self._ran_date(agent), "agent should run the date command")
        self.assertIn(self._date_parts("7/5")[0], reply)   # the prompt asks for the weekday

    def test_relative_date(self):
        agent = self._agent()
        reply = str(agent.message("What's the date on the upcoming Thursday?") or "").lower()
        self.assertTrue(skill_was_loaded(agent, "time"), "time skill should load")
        self.assertTrue(self._ran_date(agent), "agent should run the date command")
        self.assertIn("thursday", reply)              # next Thursday is a Thursday
        self._asserts_full_date(reply, "next Thursday")  # ...and the actual date

    def test_today(self):
        agent = self._agent()
        reply = str(agent.message("Remind me what the date is right now.") or "").lower()
        self.assertTrue(skill_was_loaded(agent, "time"), "time skill should load")
        self.assertTrue(self._ran_date(agent), "agent should run the date command")
        self._asserts_full_date(reply, "today")

    def test_does_not_load_on_non_date_prompt(self):
        # Anti-test: an off-topic prompt must not trip the time skill.
        agent = self._agent()
        agent.message("Write a haiku about the ocean.")
        self.assertFalse(skill_was_loaded(agent, "time"),
                         "time skill should NOT load for a non-date prompt")
