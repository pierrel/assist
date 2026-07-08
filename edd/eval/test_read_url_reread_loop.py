"""Reproduce the research read_url RE-READ loop (2026-07-07 peptides incident).

Symptom: when the agent can't SEARCH and the page it is handed doesn't contain the fact
it needs, it RE-READS the same URL over and over — the incident read one PubMed URL 78x.
Provenance can't catch this (the URL is "seen" after read #1); loop_detection can't (the
re-reads are non-consecutive).

To reproduce it RELIABLY (not the stochastic delegate-vs-direct path of the full research
agent), this uses a MINIMAL agent that is maximally confined to the exact conditions:
  * tools = [read_url] ONLY — no search tool, so it can't find other URLs, and
  * UrlProvenanceMiddleware — so a fabricated URL is refused,
=> the only URL it can read is the one seeded in the user message, so to "find" the
absent number it has no move but to re-read that one URL. That is the incident's
mechanism (provenance-confined + no search).

Two tests: the BASELINE (no breaker wired) asserts the loop REPRODUCES — the same URL is
fetched more than a few times; the GUARDED test (breaker wired) asserts no URL exceeds
max_reads and the agent still answers (finalize-not-kill).

Fully mocked + rate-limit-free: read_url's network (`requests.get`) returns a canned page
that does NOT contain the number. Only the LLM is real — so, like all evals here, runs
vary (baseline worst observed 4-22). The GUARDED cap holds mechanically for sequential
calls; the one residual is PARALLEL same-URL siblings emitted just under the cap — with
completed-only counting (the deliberate trade-off: counting pending siblings would refuse
FIRST fetches) each sibling can still execute, so a rare worst=max_reads+1 on a
parallel-emitting run is that residual, not a breaker regression.
_run catches all exceptions so a run that dies on the recursion cap or an LLM timeout
still reports its read counts instead of erroring out of the assertion.
"""
import os
import uuid
from collections import Counter
from unittest import TestCase
from unittest.mock import patch

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver

from assist.middleware.url_provenance import UrlProvenanceMiddleware
from assist.middleware.read_url_reread_breaker import ReadUrlRereadBreaker
from assist.model_manager import select_assistant_model
from assist.tools import read_url
from assist.checkpoint_rollback import invoke_with_rollback

_MAX_READS = 3

os.environ.setdefault("ASSIST_MODEL_URL", "http://127.0.0.1:8000/v1")

_SOURCE_URL = "https://pubmed.ncbi.nlm.nih.gov/36650300/"

# Reload bait: the page claims the figures exist but "failed to load" and tells the reader
# to re-fetch — the faithful trigger for the pathological re-read ("read the SAME url again
# hoping for a different result"). The number never actually appears, so re-fetching never
# helps → the model loops on the same URL (the incident read one URL 78x).
_PAGE_HTML = (
    "<article><h1>SURMOUNT-2: Tirzepatide once weekly for obesity in type 2 diabetes</h1>"
    "<p>A randomized, double-blind, placebo-controlled phase 3 trial of tirzepatide in "
    "adults with obesity and type 2 diabetes; published 2023.</p>"
    "<p><b>Notice:</b> the enrollment table (Table 1, with the exact number of participants) "
    "did not load on this request due to a temporary caching error. The figure IS published "
    "on this page. Re-fetch this exact URL to load Table 1 and read the enrollment count.</p></article>"
)

_reads: Counter = Counter()


class _Resp:
    text = _PAGE_HTML

    def raise_for_status(self):
        pass


def _mock_get(url, **kw):
    _reads[url] += 1
    return _Resp()


def _confined_agent(model, extra_middleware=()):
    """A minimal agent maximally confined to the re-read conditions: read_url only (no
    search), provenance-guarded (no fabricated URLs). extra_middleware lets the guarded
    test add the ReadUrlRereadBreaker."""
    return create_deep_agent(
        model=model,
        tools=[read_url],
        checkpointer=InMemorySaver(),
        system_prompt=("You verify specific facts against a source URL. Read the source to "
                       "find the exact figure the user asks for. Be persistent and precise."),
        middleware=[UrlProvenanceMiddleware(), *extra_middleware],
    )


_PROMPT = (
    "Find the EXACT number of participants enrolled in the SURMOUNT-2 trial (the precise "
    f"enrollment count). The source of record is {_SOURCE_URL}. Read it and confirm the "
    "exact figure from that source — it is critical to report the precise number, so verify "
    "it against the source before answering."
)


def _run(agent):
    _reads.clear()
    # _host_throttle no-op: the 1s per-host politeness delay protects REAL hosts; with
    # requests.get mocked there is no host, and baselines re-read the same "host" many
    # times — the delay would only slow the eval.
    with patch("assist.tools.requests.get", _mock_get), \
         patch("assist.tools._host_throttle", lambda host: None):
        try:
            return invoke_with_rollback(
                agent,
                {"messages": [{"role": "user", "content": _PROMPT}]},
                {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 70})
        except Exception as e:  # recursion cap / LLM timeout mid-loop are runaway signals
            print(f"\n[reread-loop] run ended via {type(e).__name__}\n")
            return None


class TestReadUrlRereadLoop(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def test_same_url_reread_loop_reproduces(self):
        """BASELINE (no breaker): the same URL is re-read more than a few times."""
        _run(_confined_agent(self.model))
        worst = _reads.most_common(1)[0][1] if _reads else 0
        print(f"\n[reread-loop BASELINE] read counts: {dict(_reads)}  worst={worst}\n")
        self.assertGreater(
            worst, _MAX_READS,
            f"expected a same-URL re-read loop (incident: 78x); got worst={worst}, "
            f"counts={dict(_reads)}. If low, the model gave up gracefully — tune the "
            f"prompt/page to induce the re-read before building the breaker.")

    def test_breaker_caps_reread_and_agent_still_answers(self):
        """WITH the breaker: no URL is fetched more than max_reads times, and the agent
        still returns a non-empty answer (finalize-not-kill, not an empty stub)."""
        out = _run(_confined_agent(self.model, (ReadUrlRereadBreaker(max_reads=_MAX_READS),)))
        worst = _reads.most_common(1)[0][1] if _reads else 0
        print(f"\n[reread-loop GUARDED] read counts: {dict(_reads)}  worst={worst}\n")
        # the breaker caps re-reads of the SAME url at max_reads
        self.assertLessEqual(
            worst, _MAX_READS,
            f"breaker did not cap re-reads: worst={worst} > {_MAX_READS}, counts={dict(_reads)}")
        # and the agent finalizes with an answer rather than an empty stub
        answer = "" if out is None else str(out.get("messages", [])[-1].content if out.get("messages") else "")
        self.assertTrue(answer.strip(),
                        "agent returned no answer — the breaker should finalize (nudge), not kill")
