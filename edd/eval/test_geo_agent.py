"""Eval: the agent uses the geo tools to answer coverage and resolve a region to add.

Real-LLM eval (small model) — deploy venv. Tests increment 2's read-only behavior with
fixture stores (no MOTIS/geocoder): NorCal is the only loaded region; the catalog holds a
handful of downloadable ones. The contract:

- a COVERAGE question → call ``list_regions`` and answer from it (don't guess).
- an out-of-region ask ("cover Seattle?") → the model states what's covered, resolves the
  region by NAME via ``find_regions`` (it infers Seattle → Washington; it must NOT type a
  Geofabrik id itself — B1), and OFFERS to add it.
- an already-covered ask ("San Francisco?") → answers yes from ``list_regions``, does NOT
  over-offer a download (the typo/over-propose damper — A3).

Prompts avoid the SKILL's example wording (probe generalization, not lexical proximity).
The propose/approve step is a later increment, so these assert up to the OFFER only.
"""
import json
import tempfile
from unittest import TestCase

from assist.agent import AgentHarness, create_agent
from assist.geo.catalog import CATALOG_FILE, Catalog
from assist.geo.model import Region, STATE_READY
from assist.geo.registry import RegionRegistry
from assist.geo.tools import geo_tools
from assist.model_manager import select_assistant_model
from assist.spec import AgentSpec

# A small deterministic catalog (subset of the real Geofabrik index ids/bboxes).
_CATALOG = [
    {"slug": "norcal", "display_name": "Northern California",
     "bbox": [-124.5, 36.0, -119.0, 42.1], "url": "https://download.geofabrik.de/norcal.osm.pbf"},
    {"slug": "socal", "display_name": "Southern California",
     "bbox": [-121.0, 32.0, -114.0, 35.9], "url": "https://download.geofabrik.de/socal.osm.pbf"},
    {"slug": "us/washington", "display_name": "Washington",
     "bbox": [-124.8, 45.5, -116.9, 49.1], "url": "https://download.geofabrik.de/washington.osm.pbf"},
    {"slug": "us/oregon", "display_name": "Oregon",
     "bbox": [-124.6, 41.9, -116.4, 46.3], "url": "https://download.geofabrik.de/oregon.osm.pbf"},
]


class TestGeoAgent(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def _agent(self):
        """An agent with the geo tools over fixture stores: only NorCal loaded."""
        root = tempfile.mkdtemp()
        geo_dir = tempfile.mkdtemp()
        reg = RegionRegistry(geo_dir)
        reg.put(Region(slug="norcal", display_name="Northern California",
                       bbox=(-124.5, 36.0, -119.0, 42.1), has_transit=True, state=STATE_READY))
        with open(f"{geo_dir}/{CATALOG_FILE}", "w") as f:
            json.dump(_CATALOG, f)
        spec = AgentSpec(tools=tuple(geo_tools(reg, Catalog(geo_dir))))
        return AgentHarness(create_agent(self.model, root, spec=spec))

    def _calls(self, agent):
        out = []
        for m in agent.all_messages():
            for c in (getattr(m, "tool_calls", None) or []):
                if c.get("name") in ("list_regions", "find_regions"):
                    out.append((c["name"], c.get("args") or {}))
        return out

    @staticmethod
    def _text(msg) -> str:
        c = getattr(msg, "content", "")
        if isinstance(c, str):
            return c
        # a list of content blocks → concatenate the text parts
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)

    def test_coverage_question_calls_list_regions(self):
        agent = self._agent()
        reply = agent.message("Which parts of the world can you actually look up trips "
                              "and routes for right now?")
        calls = self._calls(agent)
        self.assertTrue(any(n == "list_regions" for n, _ in calls),
                        f"expected list_regions; got {[n for n, _ in calls]}")
        self.assertIn("california", self._text(reply).lower())   # from the loaded set, not invented

    def test_out_of_region_resolves_and_offers(self):
        agent = self._agent()
        reply = agent.message("I'm going to be wandering around Seattle next week — can "
                              "you help me get around there?")
        calls = self._calls(agent)
        finds = [a for n, a in calls if n == "find_regions"]
        self.assertTrue(finds, f"expected find_regions to resolve the region; got "
                               f"{[n for n, _ in calls]}")
        # the model passes a NAME (Washington/Seattle), NEVER a Geofabrik id like us/washington
        for a in finds:
            q = str(a.get("query", "")).lower()
            self.assertNotIn("us/", q, f"model typed a region id instead of a name: {a}")
        text = self._text(reply).lower()
        self.assertTrue("washington" in text or "add" in text or "download" in text,
                        f"expected an offer to add Washington; got: {self._text(reply)[:300]}")

    def test_already_covered_does_not_over_offer(self):
        agent = self._agent()
        reply = agent.message("If I'm in San Francisco, can you give me a hand getting "
                              "around?")
        text = self._text(reply).lower()
        # NorCal covers SF → should say yes, not propose downloading a region
        self.assertNotIn("download", text,
                         f"should not offer a download for an already-covered area: {text[:300]}")
