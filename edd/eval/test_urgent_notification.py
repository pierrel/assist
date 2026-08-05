"""Natural acceptance coverage for the web-only urgent-notification tool.

The fixture provides only the normal web tool surface and never configures SMS, so these
real-model trials measure when the agent emits ``notify`` without sending a text.
"""
from __future__ import annotations

import shutil
import tempfile
from unittest import TestCase

from langchain_core.messages import AIMessage

from assist.agent import AgentHarness, create_agent
from assist.events.notify import notify_tools
from assist.model_manager import select_assistant_model
from assist.spec import AgentSpec

from .utils import stub_research_subagent


class TestUrgentNotification(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="urgent_notification_eval_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _agent(self) -> AgentHarness:
        with stub_research_subagent():
            return AgentHarness(create_agent(
                self.model, self.root,
                spec=AgentSpec(tools=notify_tools(lambda _tid: None))))

    @staticmethod
    def _notify_calls(agent: AgentHarness) -> list[dict]:
        return [call for message in agent.all_messages()
                if isinstance(message, AIMessage)
                for call in (message.tool_calls or [])
                if call.get("name") == "notify"]

    @staticmethod
    def _all_calls(agent: AgentHarness) -> list[dict]:
        return [call for message in agent.all_messages()
                if isinstance(message, AIMessage)
                for call in (message.tool_calls or [])]

    def test_imminent_deadline_is_flagged_with_a_message(self):
        agent = self._agent()
        with stub_research_subagent():
            agent.message(
                "I won't see this chat again until this evening. The landlord just said I "
                "must answer about the lease before 4pm today or the apartment goes to "
                "someone else. Please make sure I don't miss it.")
        calls = self._notify_calls(agent)
        self.assertEqual(len(calls), 1,
                         f"expected exactly one notify; calls: {calls}")
        all_calls = self._all_calls(agent)
        self.assertTrue(all_calls and all_calls[0].get("name") == "notify",
                        f"notify was not the first effect: {all_calls}")
        message = (calls[0].get("args") or {}).get("message", "")
        self.assertTrue(message.strip(), f"notify omitted its message: {calls}")
        self.assertIn("4", message, f"notification lacks the deadline: {calls}")

    def test_routine_response_is_not_flagged(self):
        agent = self._agent()
        with stub_research_subagent():
            agent.message("I finished reading a novel this weekend. Can you help me summarize it?")
        calls = self._notify_calls(agent)
        self.assertEqual(calls, [], f"routine request should not notify: {calls}")
