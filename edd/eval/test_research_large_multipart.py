"""Research handles a LARGE multi-part brief without degenerating.

The 2026-07-09 finding (prod thread 20260709060922): on a big multi-part query
("tell me about peptides" → 10+ distinct peptides), the lean orchestrator handed the
WHOLE brief to a single searcher, which over-searched/over-read across every part —
77 model calls, 65+ min, context bloat, no report — dramatically worse than a focused
query. The flow must SIZE the research to the brief: a brief spanning many distinct
things → bounded work per thing, not one searcher drowning. Small focused briefs
(test_research_partial_results, test_research_reliability) must keep working — this eval
guards the OTHER end of the range.

Mocked search + fetch (per the always-mock rule): the mock answers each subtopic from
canned data AND counts retrieval calls, so we can assert the flow stays bounded. The
query has 6 distinct parts, each with a UNIQUE fact that can only reach the report if the
part was actually researched — so coverage is verifiable and can't be faked from memory.

Asserts the outcome, not the mechanism (decompose vs bounded-single are both fine):
  1. Completes and writes a report (a degenerate run drowns / hits the recursion cap).
  2. The report covers ALL 6 parts (each part's unique fact is present).
  3. Total retrieval (search + fetch) stays BOUNDED — no over-search runaway.
"""
import os
import tempfile
from unittest.mock import patch

from assist.agent import create_research_agent, AgentHarness
from assist.model_manager import select_assistant_model

os.environ.setdefault("ASSIST_MODEL_URL", "http://127.0.0.1:8000/v1")
_SEARCH_HOST = "http://searxng.test"

# 6 distinct "things", each with a UNIQUE sentinel fact. Generic (a fictional product
# line) so nothing here is telegraphed by the research prompts, and the sentinels are
# invented tokens that can only come from a source the agent actually read.
_PARTS = {
    "aurora": ("The Aurora tier uses a QUASAR-9 cooling loop.", "quasar-9"),
    "borealis": ("The Borealis tier ships with a TIDELOCK power cell.", "tidelock"),
    "cascade": ("The Cascade tier runs the FERNWEH-3 controller.", "fernweh-3"),
    "drift": ("The Drift tier is rated for MARLINSPIKE certification.", "marlinspike"),
    "ember": ("The Ember tier includes a GLIMMERWICK sensor array.", "glimmerwick"),
    "frost": ("The Frost tier is built on the SLIPSTREAM-7 chassis.", "slipstream-7"),
}
_PROMPT = ("I'm evaluating the Helio product line. Tell me about each tier — Aurora, "
           "Borealis, Cascade, Drift, Ember, and Frost — and what's distinctive about each.")


class _Resp:
    def __init__(self, *, json_payload=None, text=""):
        self._json, self.text, self.status_code = json_payload, text, 200

    def json(self):
        return self._json

    def raise_for_status(self):
        pass


def _results_for(query):
    """Canned search hits: any part-name in the query yields that part's sentinel fact,
    each behind its own URL (so read_url is never needed but is available)."""
    q = query.lower()
    hits = []
    for name, (fact, _tok) in _PARTS.items():
        if name in q:
            hits.append({"title": f"Helio {name.title()} tier — spec sheet",
                         "url": f"https://helio-specs.test/{name}",
                         "content": fact})
    if not hits:  # a broad query with no part name still returns the overview
        hits = [{"title": "Helio product line overview",
                 "url": "https://helio-specs.test/overview",
                 "content": "The Helio line has six tiers: Aurora, Borealis, Cascade, Drift, Ember, Frost."}]
    return hits


def _run(counters):
    def mock_get(url, **kw):
        if url.startswith(_SEARCH_HOST):
            counters["search"] += 1
            q = (kw.get("params") or {}).get("q", "")
            return _Resp(json_payload={"results": _results_for(q), "unresponsive_engines": []})
        counters["fetch"] += 1
        # read_url on a part URL → that part's fact as page text
        name = url.rstrip("/").split("/")[-1]
        fact = _PARTS.get(name, ("Helio overview.",))[0]
        return _Resp(text=f"<article>{fact}</article>")

    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "references"), exist_ok=True)
    agent = AgentHarness(create_research_agent(select_assistant_model(0.1), root))
    with patch.dict(os.environ, {"ASSIST_SEARCH_URL": _SEARCH_HOST}), \
         patch("assist.tools.requests.get", mock_get), \
         patch("assist.tools._host_throttle", lambda host: None):
        res = str(agent.message(_PROMPT))
    return res, root


def _report_text(root, res):
    parts = [res]
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.endswith((".org", ".md")):
                with open(os.path.join(dp, f)) as fh:
                    parts.append(fh.read())
    return "\n".join(parts).lower()


# Bound on total retrieval ops for a 6-part brief. Healthy: ~1-3 searches/part (whether
# decomposed into 6 searchers or covered in a bounded sequence) + a few reads ≈ well under
# 30. The degenerate single-searcher grind did 77+ model calls with far more retrieval.
_RETRIEVAL_BOUND = 30


class TestResearchLargeMultipart:
    def test_covers_all_parts_and_stays_bounded(self):
        counters = {"search": 0, "fetch": 0}
        res, root = _run(counters)
        report = _report_text(root, res)
        total = counters["search"] + counters["fetch"]
        missing = [tok for (_f, tok) in _PARTS.values() if tok not in report]
        print(f"\n[large-multipart] searches={counters['search']} fetches={counters['fetch']} "
              f"total={total} missing_parts={missing}\n  {res[:300]!r}\n")
        # (1)+(2): a report that covers every part's unique fact
        assert not missing, f"report must cover all 6 parts; missing sentinels: {missing}"
        # (3): no over-search runaway — the degeneracy this eval exists to catch
        assert total <= _RETRIEVAL_BOUND, (
            f"retrieval runaway: {total} search+fetch calls (bound {_RETRIEVAL_BOUND}) — "
            f"the brief was handed to an unbounded searcher instead of being sized to its parts")
