"""Research handles a LARGE multi-part brief without degenerating (CITATION-CHASING fixture).

The 2026-07-09 finding (prod thread 20260709060922, peptides): a single searcher read ~61
distinct real medical pages for a 10-part brief — 77 model calls, 65+ min, no report.
Clean AND messy mocks both failed to reproduce it (main == V2 == 16 ops), because the real
driver is CITATION CHASING: real papers cite dozens of others, and the searcher follows the
chain into an ever-expanding frontier of DISTINCT pages (the re-read breaker only catches
re-reading the SAME url, not chasing new ones).

This fixture reproduces that: each tier's page states the fact then lists ~12 citation URLs;
each citation page expands into 8 more. A disciplined searcher reads the tier page, takes the
fact, and stops; a chaser follows citations and explodes. 8 parts.

Baseline against MAIN first (reference), then the branch. Asserts: covers the parts + total
retrieval stays bounded (no citation-chase runaway) + completes.
"""
import os
import tempfile
from unittest.mock import patch

from assist.agent import create_research_agent, AgentHarness
from assist.model_manager import select_assistant_model

os.environ.setdefault("ASSIST_MODEL_URL", "http://127.0.0.1:8000/v1")
_HOST = "http://searxng.test"          # search host (mock patches requests.get)
_SPECS = "https://helio-specs.test"    # page host

_PARTS = {
    "aurora":   ("uses a QUASAR-9 cooling loop",         "quasar-9"),
    "borealis": ("ships with a TIDELOCK power cell",     "tidelock"),
    "cascade":  ("runs the FERNWEH-3 controller",        "fernweh-3"),
    "drift":    ("holds MARLINSPIKE certification",      "marlinspike"),
    "ember":    ("includes a GLIMMERWICK sensor array",  "glimmerwick"),
    "frost":    ("is built on the SLIPSTREAM-7 chassis", "slipstream-7"),
    "gale":     ("boots the HALCYON-2 firmware",          "halcyon-2"),
    "harbor":   ("carries a KESTREL-5 telemetry unit",   "kestrel-5"),
}
_NAMES = list(_PARTS)
_PROMPT = ("I'm evaluating the Helio product line. Tell me about each tier — Aurora, "
           "Borealis, Cascade, Drift, Ember, Frost, Gale, and Harbor — and what's "
           "distinctive about each one.")


class _Resp:
    def __init__(self, *, json_payload=None, text=""):
        self._json, self.text, self.status_code = json_payload, text, 200

    def json(self):
        return self._json

    def raise_for_status(self):
        pass


def _search_results(query):
    q = query.lower()
    named = [n for n in _NAMES if n in q] or _NAMES[:3]
    hits = [{"title": f"Helio {n.title()} — peer-reviewed technical review",
             "url": f"{_SPECS}/tier/{n}",
             "content": f"Peer-reviewed technical review of the Helio {n.title()} tier and its hardware."}
            for n in named[:3]]
    hits += [{"title": "Helio product line — buyer's guide",
              "url": f"{_SPECS}/guide", "content": "Overview of the eight Helio tiers."},
             {"title": "Helio forum — tier comparison", "url": f"{_SPECS}/forum",
              "content": "Users compare Helio tiers; opinions vary."}]
    return hits


def _page_text(url):
    path = url.split(".test", 1)[-1]
    parts = [p for p in path.split("/") if p]
    if parts[:1] == ["tier"]:
        name = parts[1]
        fact = _PARTS[name][0]
        cites = "".join(f"\n- {_SPECS}/cite/{name}/{i}" for i in range(12))
        filler = "This review surveys the design context and prior art. " * 20
        return (f"<article><h1>Helio {name.title()} — technical review</h1>"
                f"<p>The Helio {name.title()} tier {fact}. It targets professional use.</p>"
                f"<p>{filler}</p><h2>References</h2>{cites}</article>")
    if parts[:1] == ["cite"]:
        depth = len(parts) - 1  # how deep in the citation chain
        more = "".join(f"\n- {_SPECS}{path}/{j}" for j in range(8)) if depth < 4 else ""
        filler = "The cited study details methodology, cohort, and adjacent findings. " * 18
        return (f"<article><p>Cited study {'/'.join(parts[1:])} — related background, no tier spec here.</p>"
                f"<p>{filler}</p><h2>Further reading</h2>{more}</article>")
    return ("<article>Helio buyer's guide: the line spans eight tiers — "
            + ", ".join(n.title() for n in _NAMES) + ".</article>")


def _run(counters):
    def mock_get(url, **kw):
        if url.startswith(_HOST):
            counters["search"] += 1
            q = (kw.get("params") or {}).get("q", "")
            return _Resp(json_payload={"results": _search_results(q), "unresponsive_engines": []})
        counters["fetch"] += 1
        return _Resp(text=_page_text(url))

    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "references"), exist_ok=True)
    agent = AgentHarness(create_research_agent(select_assistant_model(0.1), root))
    with patch.dict(os.environ, {"ASSIST_SEARCH_URL": _HOST}), \
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


_RETRIEVAL_BOUND = 50  # calibrate to main's baseline; a citation-chase grind blows past it


class TestResearchLargeMultipart:
    def test_covers_parts_and_stays_bounded(self):
        counters = {"search": 0, "fetch": 0}
        res, root = _run(counters)
        report = _report_text(root, res)
        total = counters["search"] + counters["fetch"]
        covered = [tok for (_f, tok) in _PARTS.values() if tok in report]
        missing = [tok for (_f, tok) in _PARTS.values() if tok not in report]
        print(f"\n[large-multipart] searches={counters['search']} fetches={counters['fetch']} "
              f"total={total} covered={len(covered)}/8 missing={missing}\n  {res[:300]!r}\n")
        assert len(covered) >= 6, f"report should cover most tiers; only {len(covered)}/8, missing {missing}"
        assert total <= _RETRIEVAL_BOUND, (
            f"retrieval runaway: {total} search+fetch calls (bound {_RETRIEVAL_BOUND}) — "
            f"the searcher chased citations instead of taking each tier's fact and moving on")
