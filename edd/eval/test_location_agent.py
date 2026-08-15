"""Behavioral coverage for deliberate current-location travel.

This is capability coverage, not a legacy-vs-candidate prompt comparison: the
opaque handle and web-only tool are new. The user asks naturally for walking
directions from here; the fixture supplies only a private browser snapshot and
local routing data. A pass means the model loads travel, asks for location, and
routes with the opaque handle without any live fetch.
"""
from datetime import datetime, timezone
import tempfile
from unittest import TestCase, mock

from langchain_core.messages import HumanMessage

from assist.agent import AgentHarness, create_agent
from assist.checkpoint_rollback import invoke_with_rollback
from assist.location import (CURRENT_LOCATION, LOCATION_CONTEXT_KEY, LocationSnapshot,
                             get_location)
from assist.model_manager import select_assistant_model
from assist.spec import AgentSpec

from .utils import agent_tool_calls, create_filesystem, skill_was_loaded, stub_research_subagent


def _routing_fixture(path: str, params: dict):
    if path == "/api/v1/geocode":
        place = str(params["text"]).removeprefix("the ")
        assert place.startswith("Ferry Building")
        return [{"name": "Ferry Building", "lat": 37.7955, "lon": -122.3937}]
    assert path == "/api/v1/plan"
    # The route's origin reaches MOTIS only after the tool resolves the private
    # handle. The model never receives these exact coordinates.
    assert params["fromPlace"] == "37.7749,-122.4194"
    if params.get("transitModes") == "TRANSIT":
        return {"itineraries": [{"duration": 29 * 60}]}
    return {"direct": [{"duration": 18 * 60, "legs": [{"distance": 1_200,
        "steps": [{"streetName": "Market Street", "distance": 1_200,
                   "polyline": {"points": "??AA", "precision": 7}}]}]}]}


def _message_with_location(agent: AgentHarness, text: str) -> str:
    """Invoke the real graph with the same private run-config seam as web."""
    snapshot = LocationSnapshot(37.7749, -122.4194, datetime.now(timezone.utc))
    result = invoke_with_rollback(
        agent.agent,
        {"messages": [HumanMessage(content=text)]},
        {"configurable": {"thread_id": agent.thread_id,
                            LOCATION_CONTEXT_KEY: snapshot},
         "recursion_limit": 5000},
    )
    return result["messages"][-1].content


class TestCurrentLocationTravel(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def test_walks_from_here_via_opaque_location_handle(self):
        with tempfile.TemporaryDirectory(prefix="current_location_travel_") as root:
            create_filesystem(root, {"README.org": "Personal notes."})
            with mock.patch.dict("os.environ", {
                    "ASSIST_ROUTING_URL": "http://routing-fixture",
                    "ASSIST_GEOCODER_URL": "",
                 }), \
                 mock.patch("assist.tools._motis_get", side_effect=_routing_fixture), \
                 mock.patch("assist.tools.requests.get",
                            side_effect=AssertionError("location travel eval must not fetch URLs")) as get, \
                 stub_research_subagent():
                agent = AgentHarness(create_agent(
                    self.model, root, spec=AgentSpec(tools=(get_location,))))
                reply = _message_with_location(
                    agent, "How do I walk from here to the Ferry Building?")

            get.assert_not_called()
            self.assertTrue(skill_was_loaded(agent, "travel"), agent.all_messages())
            self.assertTrue(agent_tool_calls(agent, "get_location"), agent.all_messages())
            routes = agent_tool_calls(agent, "directions")
            self.assertTrue(routes, agent.all_messages())
            self.assertEqual(routes[-1]["args"].get("origin"), CURRENT_LOCATION)
            self.assertIn("walk", str(reply).lower())
            self.assertIn("Ferry Building", str(reply))
