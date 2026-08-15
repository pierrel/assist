"""Eval: the agent calls the `travel` tool for travel-time / distance questions.

Real-LLM eval (small model) — run with the deploy venv against the live MOTIS
(ASSIST_ROUTING_URL in .dev.env). The signal is *invocation*: does the model load
the travel skill and call `travel(origin, destination)` for "how long from A to B"
/ "faster to bike or drive" questions (rather than answering from memory)?
"""
import tempfile
from unittest import TestCase, mock

from assist.agent import create_agent, AgentHarness
from assist.model_manager import select_assistant_model

from .utils import (agent_tool_calls, create_filesystem,
                    prompt_rewrite_web_main_spec, skill_was_loaded,
                    stub_research_subagent)


def _motis_fixture(path: str, params: dict):
    """Minimal local routing resource for the prompt-comparison outcome."""
    if path == "/api/v1/geocode":
        place = str(params["text"]).removeprefix("the ").strip()
        hits = {
            "Ferry Building": {"name": "Ferry Building", "lat": 37.7955, "lon": -122.3937},
            "Oakland City Hall": {"name": "Oakland City Hall", "lat": 37.8054, "lon": -122.2727},
        }
        # The natural question already establishes San Francisco.  Supplying
        # that city again is an equivalent, valid geocoding request, not an
        # outcome difference.
        for canonical, hit in hits.items():
            if place.startswith(canonical):
                return [hit]
        raise AssertionError(f"unexpected fixture geocode: {place!r}")
    if path != "/api/v1/plan":
        raise AssertionError(f"unexpected routing path: {path}")
    if params.get("transitModes") == "TRANSIT":
        return {"itineraries": [{"duration": 29 * 60}]}
    seconds, meters = {
        "CAR": (18 * 60, 13_200),
        "BIKE": (43 * 60, 11_800),
        "WALK": (155 * 60, 11_200),
    }[params["directModes"]]
    return {"direct": [{"duration": seconds, "legs": [{"distance": meters}]}]}


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
        self.assertTrue(skill_was_loaded(agent, "travel"),
                        "expected the agent to load the travel skill")
        self.assertTrue(self._called_travel(agent),
                        "expected the agent to call the travel tool")

    def test_calls_travel_for_mode_comparison(self):
        agent = self._agent()
        agent.message("Is it faster to bike or drive from Civic Center to the "
                      "Mission in San Francisco?")
        self.assertTrue(skill_was_loaded(agent, "travel"),
                        "expected the agent to load the travel skill")
        self.assertTrue(self._called_travel(agent),
                        "expected the agent to call the travel tool")


class TestPromptRewriteTravelOutcome(TestCase):
    """Natural web-main comparison using real travel formatting over local routing data."""

    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def test_reports_canned_travel_time(self):
        with tempfile.TemporaryDirectory(prefix="travel_web_main_") as root:
            create_filesystem(root, {"README.org": "Personal notes."})
            with mock.patch.dict("os.environ", {
                    "ASSIST_ROUTING_URL": "http://routing-fixture",
                    "ASSIST_GEOCODER_URL": "",
                 }), \
                 mock.patch("assist.tools._motis_get", side_effect=_motis_fixture), \
                 mock.patch("assist.tools.requests.get",
                            side_effect=AssertionError("travel eval must not fetch URLs")) as get, \
                 stub_research_subagent():
                agent = AgentHarness(create_agent(
                    self.model, root, spec=prompt_rewrite_web_main_spec()))
                reply = agent.message("How long does it take to get from the Ferry Building "
                                      "to Oakland City Hall?")

            get.assert_not_called()
            calls = agent_tool_calls(agent, "travel")
            self.assertTrue(calls, agent.all_messages())
            text = str(reply).lower()
            self.assertIn("18", text)
            self.assertIn("min", text)
