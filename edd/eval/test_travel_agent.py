"""Eval: the agent calls the `travel` tool for travel-time / distance questions.

Real-LLM eval (small model) — run with the deploy venv against the live MOTIS
(ASSIST_ROUTING_URL in .dev.env). The signal is *invocation*: does the model load
the travel skill and call `travel(origin, destination)` for "how long from A to B"
/ "faster to bike or drive" questions (rather than answering from memory)?
"""
import tempfile
from unittest import TestCase

from assist.agent import create_agent, AgentHarness
from assist.model_manager import select_assistant_model

from .utils import create_filesystem


class TestTravelAgent(TestCase):
    def setUp(self):
        self.model = select_assistant_model(0.1)

    def _agent(self):
        root = tempfile.mkdtemp()
        create_filesystem(root, {"README.org": "Personal notes."})
        return AgentHarness(create_agent(self.model, root))  # travel is built-in

    def _called_travel(self, agent) -> bool:
        for m in agent.all_messages():
            for c in (getattr(m, "tool_calls", None) or []):
                if c.get("name") == "travel":
                    return True
        return False

    def test_calls_travel_for_distance_question(self):
        agent = self._agent()
        agent.message("How long does it take to get from the Ferry Building to "
                      "Oakland City Hall?")
        self.assertTrue(self._called_travel(agent),
                        "expected the agent to call the travel tool")

    def test_calls_travel_for_mode_comparison(self):
        agent = self._agent()
        agent.message("Is it faster to bike or drive from Civic Center to the "
                      "Mission in San Francisco?")
        self.assertTrue(self._called_travel(agent),
                        "expected the agent to call the travel tool")
