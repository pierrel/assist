"""Research reliability evals: URL provenance + dead-fetch recovery.

The 2026-06-24 threads.db forensics showed research's dominant failure mode was
the agent *fetching guessed/constructed URLs* (e.g. casio.com/watch/<model>) and
re-fetching dead ones. These evals pin the contract:

- Every URL the agent fetches must have come from a search result (no guessing).
- A failed fetch must not be retried; the agent picks a different result and
  still produces a report.

Real model, but search + fetch are served from a CANNED fixture via
ResearchToolSpy — provenance/recovery are about what the model does with search
results, not live search quality, and the live SearXNG engines rate-limit /
CAPTCHA unpredictably (a live run burned 26 min on dead-search retries). Canned
fixtures make the eval deterministic and fast. The spy patches the tools where
the research sub-agents bind them, so the inner sub-agent's calls are captured
(``all_messages()`` only exposes the top-level thread).
"""
import logging
import sys
import tempfile
from unittest import TestCase

from assist.agent import create_research_agent, AgentHarness
from assist.model_manager import select_assistant_model

from .utils import (create_filesystem, files_in_directory, normalize_url,
                    ResearchToolSpy)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

_RESEARCH_Q = ("What kids' digital watches does Casio currently sell, and what "
               "are their model numbers? Save the result to casio.org")

# Canned search results: model numbers (F-91W, LA670, DW-5600) are present in the
# snippets, so the agent CAN answer from these — a guessing-prone model will
# nonetheless construct per-model casio.com URLs that are NOT in this set, which
# is exactly what the provenance test catches.
_CASIO_FIXTURE = [
    {"title": "Casio F-91W Standard Digital Watch",
     "url": "https://www.casio.com/us/watches/casio/product.F-91W-1/",
     "content": "The F-91W is Casio's classic compact digital watch, a popular "
                "first watch for kids: daily alarm, stopwatch, water resistant."},
    {"title": "Best Casio watches for kids in 2026",
     "url": "https://www.example-watchguide.com/best-casio-kids-watches",
     "content": "Top kid-friendly Casio models: the F-91W, the LA670, and the "
                "tough G-Shock DW-5600."},
    {"title": "Casio LA670WA compact digital watch",
     "url": "https://www.example-retailer.com/casio-la670wa",
     "content": "The LA670WA is a small-wrist digital watch with alarm, calendar "
                "and LED light — often recommended for children."},
    {"title": "G-Shock DW-5600 review",
     "url": "https://www.example-watchguide.com/g-shock-dw-5600",
     "content": "The DW-5600 is a tough, kid-proof digital G-Shock: 200m water "
                "resistance and shock resistance."},
]


class TestResearchReliability(TestCase):
    def setUp(self):
        self.model = select_assistant_model(0.1)

    def _agent(self):
        root = tempfile.mkdtemp()
        create_filesystem(root, {"references": {}})
        return AgentHarness(create_research_agent(self.model, root)), root

    def test_only_fetches_search_result_urls(self):
        """Every fetched URL came from a search result — none were guessed."""
        with ResearchToolSpy(search_fixture=_CASIO_FIXTURE) as spy:
            agent, _root = self._agent()  # build INSIDE the patch so sub-agent
            agent.message(_RESEARCH_Q)    # tools bind to the spy, not the real fns
        self.assertTrue(spy.fetched,
                        "agent fetched no URLs — cannot assess provenance")
        guessed = spy.guessed_fetches()
        self.assertEqual(
            guessed, [],
            f"agent fetched {len(guessed)} URL(s) from NO search result "
            f"(guessed/constructed): {guessed} — searched {len(spy.searched)}x "
            f"yielding {len(spy.search_results)} result URLs; "
            f"fetched {len(spy.fetched)} total.")

    def test_recovers_from_dead_fetch(self):
        """A failed fetch isn't retried; the agent tries a different result and
        still writes a report."""
        with ResearchToolSpy(search_fixture=_CASIO_FIXTURE, fail_first=1) as spy:
            agent, root = self._agent()  # build INSIDE the patch (see above)
            agent.message(_RESEARCH_Q)
        self.assertTrue(spy.failed_first,
                        "no fetch occurred, so the dead-fetch path was never hit")
        dead = spy.failed_first[0]
        self.assertEqual(
            spy.fetch_count(dead), 1,
            f"agent re-fetched the dead URL {dead} {spy.fetch_count(dead)}x "
            f"instead of moving on")
        others = {normalize_url(u) for u in spy.fetched} - {dead}
        self.assertTrue(
            others,
            "agent fetched no other URL after the dead fetch — it gave up "
            "instead of trying a different search result")
        self.assertTrue(
            files_in_directory(f"{root}/references"),
            "agent produced no report after recovering from the dead fetch")
