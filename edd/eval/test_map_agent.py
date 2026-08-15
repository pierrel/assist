"""GATE eval (Pierre): can the small model emit a well-formed `type: map` render
block, given the right guidance (the render skill's Maps section) and accurate
coordinates from a tool? Run BEFORE building any render plumbing — it decides
whether the model can COMPOSE the block (map_data returns coords, model formats
the pin:/path: lines) or whether the tool must return the whole block instead.

`map_data` is now a real general-agent built-in (assist/tools.py) — the eval
does NOT wire a tool, it exercises the real registration. No render/web plumbing
exists yet (a `type: map` block currently renders as a code block). The signal is
EMISSION: does the model load the (extended) render skill, call map_data, and emit
a well-formed map block with valid `pin:`/`path:` lines?

Real-LLM eval (small model) — deploy venv; partial pass rates expected.
"""
import os
import re
import shutil
import tempfile
from unittest import TestCase, mock

from deepagents.backends import FilesystemBackend

from assist.agent import create_agent, AgentHarness
from assist.model_manager import select_assistant_model
from assist.spec import AgentSpec

from .utils import (agent_tool_calls, create_filesystem, AgentTestMixin,
                    prompt_rewrite_web_main_spec, stub_research_subagent)

_RENDER_SKILLS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assist", "web_skills")

# A ```render block whose body declares type: map.
_MAP_BLOCK = re.compile(r"```render\b(.*?)```", re.S | re.I)
# A pin line: `pin: [origin] <lat>,<lon> <label>` — an OPTIONAL leading `origin` marker.
_PIN = re.compile(r"^\s*pin:\s*(?:origin\s+)?(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s+\S",
                  re.M | re.I)
# An origin pin = the user's location (the renderer colors it green).
_ORIGIN_PIN = re.compile(r"^\s*pin:\s*origin\s+-?\d+\.\d+\s*,\s*-?\d+\.\d+", re.M | re.I)
_PATH = re.compile(r"^\s*path:\s*\S+\s+\S", re.M)


def _map_geocoder_fixture(path: str, params: dict):
    """Local geocoding resource for the real map_data comparison path."""
    if path != "/api/v1/geocode":
        raise AssertionError(f"unexpected mapping request: {path} {params}")
    query = str(params["text"]).lower()
    places = {
        "four barrel": ("Four Barrel Coffee", 37.7609, -122.4210),
        "ritual": ("Ritual Coffee", 37.7615, -122.4217),
        "haus": ("Haus Coffee", 37.7597, -122.4202),
    }
    match = next((item for needle, item in places.items() if needle in query), None)
    if match is None:
        raise AssertionError(f"unexpected geocoder query: {query}")
    name, lat, lon = match
    return [{"name": name, "lat": lat, "lon": lon}]


class TestMapAgent(AgentTestMixin, TestCase):
    def setUp(self):
        self.model = select_assistant_model(0.1)

    def _agent(self):
        root = tempfile.mkdtemp()
        create_filesystem(root, {"README.org": "Personal assistant workspace."})
        skills = {"/render-skill/": FilesystemBackend(root_dir=_RENDER_SKILLS_DIR,
                                                      virtual_mode=True)}
        # map_data is a real general-agent built-in now (no tools= wiring here).
        return AgentHarness(create_agent(self.model, root,
                                         spec=AgentSpec(skill_sources=skills)))

    def _map_blocks(self, agent) -> list:
        from langchain_core.messages import AIMessage
        blocks = []
        for m in agent.all_messages():
            if isinstance(m, AIMessage) and isinstance(m.content, str):
                blocks.extend(b for b in _MAP_BLOCK.findall(m.content)
                              if re.search(r"type:\s*map", b, re.I))
        return blocks

    def _assert_wellformed(self, blocks, need_path=False):
        self.assertTrue(blocks, f"expected a type:map render block; got none")
        pins = [b for b in blocks if _PIN.search(b)]
        self.assertTrue(pins, f"expected >=1 valid `pin: lat,lon label` line; blocks: {blocks}")
        # coords must be plausible (in range) — the model copied them, didn't invent
        for lat, lon in _PIN.findall(pins[0]):
            self.assertTrue(-90 <= float(lat) <= 90 and -180 <= float(lon) <= 180)
        if need_path:
            self.assertTrue(any(_PATH.search(b) for b in blocks),
                            f"expected a `path:` line for the route; blocks: {blocks}")

    def test_emits_map_block_for_places(self):
        """The core gate: 'show me these places on a map' → the model calls
        map_data then emits a well-formed type:map block with pin: lines."""
        agent = self.message_with_mocked_research(self._agent,
                      "Show me these coffee shops on a map: Four Barrel, Ritual "
                      "Coffee, and Haus Coffee — all on or near Valencia Street in "
                      "San Francisco.")
        self._assert_wellformed(self._map_blocks(agent))

    def test_emits_map_with_route(self):
        """Pins + a path: 'how far to walk, map it' → a pin per place AND a path
        line for the route."""
        agent = self.message_with_mocked_research(self._agent,
                      "How far is it to walk from Fellow Barber on Valencia Street "
                      "to Haus Coffee in San Francisco? Show it on a map.")
        self._assert_wellformed(self._map_blocks(agent), need_path=True)

    def test_emits_map_alternate_wording(self):
        """Generality — wording the skill doesn't telegraph ('plot', no 'map')."""
        agent = self.message_with_mocked_research(self._agent,
                      "Plot Dolores Park and the Ferry Building in San Francisco "
                      "so I can see where they are relative to each other.")
        self._assert_wellformed(self._map_blocks(agent))

    def test_recommendation_renders_map(self):
        """The updated guidance: when RECOMMENDING places (NOT explicitly asked to map),
        the model still renders a map. Places are named so no research runs. The
        older coordinate-in-context origin assertion is intentionally retired:
        current location now comes only from the deliberate travel-tool lookup."""
        agent = self.message_with_mocked_research(self._agent,
            "Which of these is best for a laptop session for a couple hours: Four Barrel "
            "Coffee, Ritual Coffee, or Haus Coffee? They're all near Valencia Street, SF.")
        blocks = self._map_blocks(agent)
        self._assert_wellformed(blocks)     # a map appeared though the user never said "map it"


class TestPromptRewriteMapOutcome(TestMapAgent):
    """Natural web-main comparison for a visible map rendered from local coordinates."""

    test_emits_map_block_for_places = None
    test_emits_map_with_route = None
    test_emits_map_alternate_wording = None
    test_recommendation_renders_map = None

    def _agent(self):
        root = tempfile.mkdtemp(prefix="map_web_main_")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        create_filesystem(root, {"README.org": "Personal assistant workspace."})
        return AgentHarness(create_agent(
            self.model, root, spec=prompt_rewrite_web_main_spec()))

    def test_renders_canned_places_map(self):
        with mock.patch.dict("os.environ", {
                "ASSIST_ROUTING_URL": "http://routing-fixture",
                "ASSIST_GEOCODER_URL": "",
             }), \
             mock.patch("assist.tools._motis_get", side_effect=_map_geocoder_fixture), \
             mock.patch("assist.tools.requests.get",
                        side_effect=AssertionError("map eval must not fetch URLs")) as get, \
             stub_research_subagent():
            agent = self._agent()
            agent.message("Show me these coffee shops on a map: Four Barrel, Ritual "
                          "Coffee, and Haus Coffee — all on or near Valencia Street in "
                          "San Francisco.")

        get.assert_not_called()
        self.assertTrue(agent_tool_calls(agent, "map_data"), agent.all_messages())
        self._assert_wellformed(self._map_blocks(agent))
