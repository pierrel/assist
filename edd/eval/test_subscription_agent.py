"""Natural acceptance eval for bundled subscription-tool disclosure."""
import os
import tempfile
from unittest import TestCase, mock

from assist.agent import AgentHarness, create_agent
from assist.events.store import SubscriptionStore
from assist.events.tools import subscription_tools
from assist.model_manager import select_assistant_model
from assist.spec import AgentSpec

from .utils import (agent_tool_calls, prompt_rewrite_web_main_spec,
                    skill_was_loaded, stub_research_subagent)


class TestSubscriptionAgent(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def test_natural_request_loads_skill_and_creates_subscription(self):
        with tempfile.TemporaryDirectory(prefix="subscription_eval_store_") as store_root, \
                tempfile.TemporaryDirectory(prefix="subscription_eval_workspace_") as workspace_root:
            os.makedirs(os.path.join(store_root, "subscription-eval"))
            store = SubscriptionStore(store_root)
            with stub_research_subagent():
                agent = AgentHarness(
                    create_agent(
                        self.model,
                        workspace_root,
                        spec=AgentSpec(tools=tuple(subscription_tools(store))),
                    ),
                    thread_id="subscription-eval",
                )
                agent.message(
                    "Whenever a text comes in from a 415 number, check whether it is "
                    "asking to schedule something and draft a short reply for me to review."
                )

            calls = agent_tool_calls(agent)
        self.assertTrue(skill_was_loaded(agent, "subscribe-events"))
        self.assertTrue(any(call.get("name") == "create_subscription"
                            for call in calls), calls)


class TestPromptRewriteSubscriptionOutcome(TestCase):
    """Natural web-main comparison with a persisted event-subscription outcome."""

    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def test_creates_requested_subscription(self):
        thread_id = "subscription-eval"
        with tempfile.TemporaryDirectory(prefix="subscription_web_main_store_") as store_root, \
                tempfile.TemporaryDirectory(prefix="subscription_web_main_workspace_") as workspace_root:
            os.makedirs(os.path.join(store_root, thread_id))
            store = SubscriptionStore(store_root)
            with mock.patch("assist.tools.requests.get",
                            side_effect=AssertionError("subscription eval must not fetch URLs")) as get, \
                 stub_research_subagent():
                agent = AgentHarness(create_agent(
                    self.model, workspace_root,
                    spec=prompt_rewrite_web_main_spec(
                        tools=tuple(subscription_tools(store))),
                ), thread_id=thread_id)
                reply = agent.message(
                    "Whenever a text comes in from a 415 number, check whether it is "
                    "asking to schedule something and draft a short reply for me to review.")
            get.assert_not_called()
            saved = store.for_thread(thread_id)

        self.assertEqual(len(saved), 1, saved)
        self.assertIn("415", saved[0].sender_regexp)
        self.assertIn("schedule", saved[0].template.lower())
        self.assertIn("reply", saved[0].template.lower())
        self.assertRegex(str(reply).lower(), r"(subscribed|subscription|whenever)")
