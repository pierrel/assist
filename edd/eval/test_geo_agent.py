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

- the user CONFIRMS the offer → ``propose_region_download`` with the EXACT id from the
  ``find_regions`` result (the B1 sink contract) + the reply says it awaits approval —
  a proposal is not a download.

Prompts avoid the SKILL's example wording (probe generalization, not lexical
proximity). The size HEAD is mocked (no real network). The web approve/import flow is
a later increment; the agent's contract ends at the recorded proposal.
"""
import json
import tempfile
from unittest import TestCase
from unittest.mock import patch

from assist.agent import AgentHarness, create_agent
from assist.geo.catalog import CATALOG_FILE, Catalog
from assist.geo.model import Region, STATE_READY
from assist.geo.proposals import ProposalStore
from assist.geo.registry import RegionRegistry
from assist.geo.tools import geo_tools
from assist.model_manager import select_assistant_model
from assist.spec import AgentSpec

from .utils import stub_research_subagent


class _FakeHead:
    """No real network in the eval: a canned HEAD (on-host, 700 MB)."""
    def __init__(self, url, **kw):
        self.url = url
        self.headers = {"Content-Length": "700000000"}

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
        """An agent with the geo tools over fixture stores: only NorCal loaded.
        Returns (harness, proposal_store) so tests can assert what got recorded."""
        root = tempfile.mkdtemp()
        geo_dir = tempfile.mkdtemp()
        reg = RegionRegistry(geo_dir)
        reg.put(Region(slug="norcal", display_name="Northern California",
                       bbox=(-124.5, 36.0, -119.0, 42.1), has_transit=True, state=STATE_READY))
        with open(f"{geo_dir}/{CATALOG_FILE}", "w") as f:
            json.dump(_CATALOG, f)
        props = ProposalStore(geo_dir)
        spec = AgentSpec(tools=tuple(geo_tools(reg, Catalog(geo_dir), props)))
        return AgentHarness(create_agent(self.model, root, spec=spec)), props

    def _calls(self, agent):
        out = []
        for m in agent.all_messages():
            for c in (getattr(m, "tool_calls", None) or []):
                if c.get("name") in ("list_regions", "find_regions",
                                     "propose_region_download"):
                    out.append((c["name"], c.get("args") or {}))
        return out

    # AgentHarness.message() returns the final message's .content directly (a str, or a
    # list of content blocks for the reasoning model) — NOT the message object.
    _CATALOG_SLUGS = frozenset({"norcal", "socal", "us/washington", "us/oregon"})

    @staticmethod
    def _text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in content)
        return str(content or "")

    def test_coverage_question_calls_list_regions(self):
        with stub_research_subagent():   # coverage prompts shouldn't research; mock per policy
            agent, _ = self._agent()
            reply = agent.message("Which parts of the world can you actually look up trips "
                                  "and routes for right now?")
        calls = self._calls(agent)
        self.assertTrue(any(n == "list_regions" for n, _ in calls),
                        f"expected list_regions; got {[n for n, _ in calls]}")
        self.assertIn("california", self._text(reply).lower())   # from the loaded set, not invented

    def test_out_of_region_resolves_and_offers(self):
        # Portland → Oregon: a city the SKILL does NOT name (it worked-examples Seattle /
        # San Diego), so this probes the model's own city→state inference, not recall.
        with stub_research_subagent():
            agent, _ = self._agent()
            reply = agent.message("I'm going to be spending a week in Portland and I'll "
                                  "need help getting around town — can you do that?")
        calls = self._calls(agent)
        finds = [a for n, a in calls if n == "find_regions"]
        self.assertTrue(finds, f"expected find_regions to resolve the region; got "
                               f"{[n for n, _ in calls]}")
        # B1: the model passes a NAME ("Oregon"/"Portland"), never a Geofabrik id — a
        # query EQUAL to a catalog slug (e.g. "us/oregon") means it typed an id.
        for a in finds:
            q = str(a.get("query", "")).strip().lower()
            self.assertNotIn(q, self._CATALOG_SLUGS,
                             f"model typed a region id instead of a name: {a}")
        # the resolved region name must actually appear (not just a magic word like "add")
        self.assertIn("oregon", self._text(reply).lower(),
                      f"expected it to resolve + offer Oregon; got: {self._text(reply)[:300]}")

    def test_user_confirms_then_propose_with_exact_id(self):
        # Two turns: the offer, then the user's yes → the model must call
        # propose_region_download with the EXACT id find_regions returned (us/oregon),
        # and present the result as awaiting approval — not as a started download.
        # Eugene (not a SKILL example; unambiguously Oregon) so the confirm flow isn't
        # confounded by a legit "which Portland, OR or ME?" disambiguation.
        with stub_research_subagent(), \
             patch("assist.geo.tools.requests.head", _FakeHead):
            agent, props = self._agent()
            agent.message("I'm going to be spending a week in Eugene and I'll "
                          "need help getting around town — can you do that?")
            reply = agent.message("Yes please, go ahead and set that up.")
        proposes = [a for n, a in self._calls(agent) if n == "propose_region_download"]
        self.assertTrue(proposes, f"expected propose_region_download after the user's "
                                  f"yes; got {[n for n, _ in self._calls(agent)]}")
        self.assertEqual(str(proposes[-1].get("region_id", "")).strip(), "us/oregon",
                         f"expected the exact find_regions id; got {proposes[-1]}")
        self.assertIsNotNone(props.get("us/oregon"), "proposal was not recorded")
        self.assertIn("approv", self._text(reply).lower(),
                      f"reply should say it awaits approval; got: {self._text(reply)[:300]}")

    def test_already_covered_does_not_over_offer(self):
        with stub_research_subagent():
            agent, _ = self._agent()
            reply = agent.message("If I'm in San Francisco, can you give me a hand getting "
                                  "around?")
        text = self._text(reply).lower()
        # NorCal covers SF → should say yes, not propose downloading a region
        self.assertNotIn("download", text,
                         f"should not offer a download for an already-covered area: {text[:300]}")
