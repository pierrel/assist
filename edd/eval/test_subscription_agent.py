"""Natural acceptance eval for bundled subscription-tool disclosure."""
import tempfile
from unittest import TestCase

from langchain_core.messages import AIMessage

from assist.agent import AgentHarness, create_agent
from assist.events.store import SubscriptionStore
from assist.events.tools import subscription_tools
from assist.model_manager import select_assistant_model
from assist.spec import AgentSpec

from .utils import skill_was_loaded, stub_research_subagent


class TestSubscriptionAgent(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def test_natural_request_loads_skill_and_creates_subscription(self):
        store = SubscriptionStore(tempfile.mkdtemp(prefix="subscription_eval_store_"))
        with stub_research_subagent():
            agent = AgentHarness(
                create_agent(
                    self.model,
                    tempfile.mkdtemp(prefix="subscription_eval_workspace_"),
                    spec=AgentSpec(tools=tuple(subscription_tools(store))),
                ),
                thread_id="subscription-eval",
            )
            agent.message(
                "Whenever a text comes in from a 415 number, check whether it is "
                "asking to schedule something and draft a short reply for me to review."
            )

        calls = [call for message in agent.all_messages()
                 if isinstance(message, AIMessage)
                 for call in (message.tool_calls or [])]
        self.assertTrue(skill_was_loaded(agent, "subscribe-events"))
        self.assertTrue(any(call.get("name") == "create_subscription"
                            for call in calls), calls)
