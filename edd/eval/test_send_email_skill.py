"""Natural acceptance eval for the web-only email send skill.

The prompt asks for the user-visible outcome without naming a skill, tool, approval
mechanism, sender, or CC. The graph pauses before delivery, so no provider request runs.
"""
import shutil
import tempfile
from unittest import TestCase

from assist.agent import AgentHarness, create_agent
from assist.events.email import EMAIL_INTERRUPT_ON, email_tools
from assist.model_manager import select_assistant_model
from assist.spec import AgentSpec
from assist.thread_manager import _web_skill_sources

from .utils import skill_was_loaded


def _proposed_email(harness) -> dict | None:
    snapshot = harness.agent.get_state({"configurable": {"thread_id": harness.thread_id}})
    for interrupt in getattr(snapshot, "interrupts", None) or ():
        for action in (interrupt.value or {}).get("action_requests", []):
            if action.get("name") == "send_email":
                return action.get("args")
    return None


class TestSendEmailSkill(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="send_email_eval_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_natural_request_loads_skill_and_proposes_one_email(self):
        agent = AgentHarness(create_agent(
            self.model, self.root,
            spec=AgentSpec(skill_sources=_web_skill_sources(), tools=email_tools(),
                           interrupt_on=EMAIL_INTERRUPT_ON)))

        agent.message(
            "Could you let Robin know at robin@example.test that the Tuesday meeting is "
            "moving to 2pm, and ask whether that still works?"
        )

        proposal = _proposed_email(agent)
        self.assertTrue(skill_was_loaded(agent, "send-email"))
        self.assertIsNotNone(proposal, "agent did not propose an email for the request")
        assert proposal is not None
        self.assertEqual(proposal.get("to"), "robin@example.test")
        self.assertIn("Tuesday", proposal.get("body", ""))
        self.assertTrue(proposal.get("subject", "").strip())
