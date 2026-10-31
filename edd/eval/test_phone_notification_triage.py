"""Natural SMS-subscription attention coverage using the actual reduced web tools."""
import shutil
import tempfile
from unittest import TestCase, mock

from assist.agent import AgentHarness, create_agent
from assist.spec import AgentSpec
from assist.events.model import Subscription
from assist.events.notify import valid_reason
from assist.model_manager import select_assistant_model
from assist import thread_manager
from manage.web import state
from .utils import agent_tool_calls, stub_research_subagent


class TestPhoneNotificationTriage(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def test_time_sensitive_message_queues_a_bounded_reason(self):
        directory = tempfile.mkdtemp(prefix='phone-notify-triage-')
        queued = []
        try:
            spec = AgentSpec(
                tools=thread_manager._web_triage_tools,
                skill_sources=thread_manager._triage_skill_sources(),
                async_subagent_tools=(), web_main=False, main_guidance_skills=False,
                interrupt_on=thread_manager._web_triage_interrupt_on)
            with stub_research_subagent():
                agent = AgentHarness(create_agent(self.model, directory, spec=spec))
            subscription = Subscription(
                id='synthetic', thread_id='synthetic', sender_regexp='.*',
                template='A text arrived from {sender}: {text}\nLet me know about messages that need attention today.')
            prompt = subscription.render('+15555550100',
                'The clinic needs your confirmation before 3pm today or the appointment slot will be released.')
            with mock.patch.object(state, '_mark_urgent',
                                   side_effect=lambda tid, reason: queued.append((tid, reason)) or 'queued'), stub_research_subagent():
                agent.message(prompt)
            self.assertEqual(len(agent_tool_calls(agent, 'notify')), 1)
            self.assertEqual(len(queued), 1)
            self.assertTrue(valid_reason(queued[0][1]))
        finally:
            shutil.rmtree(directory, ignore_errors=True)
