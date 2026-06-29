"""Eval: the agent DISCRIMINATES between `travel` and `directions`.

Real-LLM eval (small model) — deploy venv, live MOTIS (ASSIST_ROUTING_URL in
.dev.env). Both tools live behind ONE travel skill (Pierre's call), so the risk is
mis-selection: the model must call `directions` (+ a mode) for "how do I get there
/ which bus" prompts, and `travel` for "how long / how far" prompts — and NOT the
other. Prompts deliberately avoid the skill's example wording (probe generalization,
not lexical proximity).
"""
import tempfile
from unittest import TestCase

from assist.agent import create_agent, AgentHarness
from assist.model_manager import select_assistant_model

from .utils import create_filesystem


class TestDirectionsAgent(TestCase):
    def setUp(self):
        self.model = select_assistant_model(0.1)

    def _agent(self):
        root = tempfile.mkdtemp()
        create_filesystem(root, {"README.org": "Personal notes."})
        return AgentHarness(create_agent(self.model, root))  # travel + directions built-in

    def _calls(self, agent):
        names = []
        for m in agent.all_messages():
            for c in (getattr(m, "tool_calls", None) or []):
                if c.get("name") in ("travel", "directions"):
                    names.append((c["name"], c.get("args") or {}))
        return names

    # --- directions-shaped prompts -> directions, NOT travel ---
    def test_step_by_step_calls_directions(self):
        agent = self._agent()
        agent.message("Give me step-by-step directions from Coit Tower to Oracle "
                      "Park by car.")
        calls = self._calls(agent)
        self.assertTrue(any(n == "directions" for n, _ in calls),
                        f"expected directions; got {[n for n, _ in calls]}")
        self.assertFalse(any(n == "travel" for n, _ in calls),
                         "directions-shaped prompt should not call travel")

    def test_which_train_calls_directions_transit(self):
        agent = self._agent()
        agent.message("Which train do I take to get from Civic Center to the "
                      "Embarcadero?")
        calls = self._calls(agent)
        dir_calls = [a for n, a in calls if n == "directions"]
        self.assertTrue(dir_calls, f"expected directions; got {[n for n, _ in calls]}")
        # mode should resolve to transit for a "which train" ask
        self.assertTrue(any(str(a.get("mode", "")).lower() == "transit" for a in dir_calls),
                        f"expected mode=transit; got {dir_calls}")

    # --- travel-shaped prompt -> travel, NOT directions (no traffic-stealing) ---
    def test_how_long_calls_travel_not_directions(self):
        agent = self._agent()
        agent.message("Roughly how long is the drive from the Ferry Building to "
                      "Oakland City Hall?")
        calls = self._calls(agent)
        self.assertTrue(any(n == "travel" for n, _ in calls),
                        f"expected travel; got {[n for n, _ in calls]}")
        self.assertFalse(any(n == "directions" for n, _ in calls),
                         "travel-shaped prompt should not call directions")
