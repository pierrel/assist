"""Eval: the agent DISCRIMINATES between `travel` and `directions`.

Real-LLM eval (small model) — deploy venv, live MOTIS (ASSIST_ROUTING_URL in
.dev.env). Both tools live behind ONE travel skill (Pierre's call), so the risk is
mis-selection: the model must call `directions` (+ a mode) for "how do I get there
/ which bus" prompts, and `travel` for "how long / how far" prompts — and NOT the
other. Prompts deliberately avoid the skill's example wording (probe generalization,
not lexical proximity).
"""
import tempfile
from unittest import TestCase, mock

from assist.agent import create_agent, AgentHarness
from assist.model_manager import select_assistant_model

from .utils import (agent_tool_calls, create_filesystem,
                    prompt_rewrite_web_main_spec, skill_was_loaded,
                    stub_research_subagent)


def _motis_fixture(path: str, params: dict):
    """Minimal local transit resource for the prompt-comparison outcome."""
    if path == "/api/v1/geocode":
        query = str(params["text"]).removeprefix("the ").strip()
        hits = {
            "Civic Center": {"name": "Civic Center", "lat": 37.7793, "lon": -122.4142},
            "Embarcadero": {"name": "Embarcadero", "lat": 37.7929, "lon": -122.3971},
        }
        place = next((name for name in hits if query.startswith(name)), None)
        if place is None:
            raise AssertionError(f"unexpected geocoder query: {query}")
        return [hits[place]]
    if path == "/api/v1/plan" and params.get("transitModes") == "TRANSIT":
        return {"itineraries": [{
            "duration": 21 * 60,
            "startTime": "2026-08-13T15:00:00Z",
            "endTime": "2026-08-13T15:21:00Z",
            "legs": [
                {"mode": "WALK", "duration": 3 * 60,
                 "from": {"name": "Civic Center"}, "to": {"name": "Civic Center BART"}},
                {"mode": "SUBWAY", "duration": 15 * 60,
                 "from": {"name": "Civic Center BART"}, "to": {"name": "Embarcadero"},
                 "routeShortName": "BART", "headsign": "Dublin/Pleasanton",
                 "intermediateStops": [{}, {}], "startTime": "2026-08-13T15:03:00Z"},
            ],
        }]}
    raise AssertionError(f"unexpected routing request: {path} {params}")


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
        self.assertTrue(skill_was_loaded(agent, "travel"))
        calls = self._calls(agent)
        self.assertTrue(any(n == "directions" for n, _ in calls),
                        f"expected directions; got {[n for n, _ in calls]}")
        self.assertFalse(any(n == "travel" for n, _ in calls),
                         "directions-shaped prompt should not call travel")

    def test_which_train_calls_directions_transit(self):
        agent = self._agent()
        agent.message("Which train do I take to get from Civic Center to the "
                      "Embarcadero?")
        self.assertTrue(skill_was_loaded(agent, "travel"))
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
        self.assertTrue(skill_was_loaded(agent, "travel"))
        calls = self._calls(agent)
        self.assertTrue(any(n == "travel" for n, _ in calls),
                        f"expected travel; got {[n for n, _ in calls]}")
        self.assertFalse(any(n == "directions" for n, _ in calls),
                         "travel-shaped prompt should not call directions")


class TestPromptRewriteTransitDirectionsOutcome(TestCase):
    """Natural web-main comparison using the real directions tool over local transit data."""

    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def test_reports_canned_train_directions(self):
        with tempfile.TemporaryDirectory(prefix="directions_web_main_") as root:
            create_filesystem(root, {"README.org": "Personal notes."})
            with mock.patch.dict("os.environ", {
                    "ASSIST_ROUTING_URL": "http://routing-fixture",
                    "ASSIST_GEOCODER_URL": "",
                 }), \
                 mock.patch("assist.tools._motis_get", side_effect=_motis_fixture), \
                 mock.patch("assist.tools.requests.get",
                            side_effect=AssertionError("directions eval must not fetch URLs")) as get, \
                 stub_research_subagent():
                agent = AgentHarness(create_agent(
                    self.model, root, spec=prompt_rewrite_web_main_spec()))
                reply = agent.message("Which train do I take to get from Civic Center to the "
                                      "Embarcadero?")

            get.assert_not_called()
            calls = agent_tool_calls(agent, "directions")
            self.assertTrue(calls, agent.all_messages())
            self.assertIn("bart", str(reply).lower())
