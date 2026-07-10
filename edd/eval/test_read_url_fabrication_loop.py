"""Reproduce the research read_url FABRICATION-404 loop (2026-07-08 b37c712b incident).

Symptom: under search-unavailable (no real URLs in context), the research ORCHESTRATOR
GUESSES plausible-looking URLs, read_url 404s them, and loops — "Let me try completely
different sources now" -> guess another URL -> 404 -> repeat. Thread 20260708090812:
108,278 "404 Not Found" results, 10 successes, 0 successful searches, ~90 min on the slot.
LoopDetection can't catch it (read_url is read-only -> transparent to its error-pattern;
and the model VARIES the guessed URL each call, so the same-args pattern never fires); the
reread breaker can't (the URLs are distinct).

The guard that DOES catch it is UrlProvenanceMiddleware: it refuses a read_url whose URL
appears in no prior tool/user message — exactly a fabrication. It was wired on the searcher
+ fact-checker sub-agents but NOT the orchestrator; the fix (docs/2026-07-08-research-
reread-runaway.org) adds it there too.

Confined reproduction (same approach as test_read_url_reread_loop): a minimal agent with
read_url ONLY (no search tool) and NO seeded URL, so the ONLY way to call read_url is to
fabricate a URL from memory. read_url's network is mocked to 404 every URL. BASELINE (no
provenance) fabricates and 404-loops; GUARDED (provenance) refuses every fabricated fetch,
so ZERO network reads happen and the agent finalizes by reporting it couldn't find sources.

Only the LLM is real, so runs vary. _run catches all exceptions (a recursion-cap death mid-
loop is itself a runaway signal) so the counts are still reported.
"""
import os
import uuid
from collections import Counter
from unittest import TestCase
from unittest.mock import patch

import requests
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver

from assist.middleware.url_provenance import UrlProvenanceMiddleware
from assist.model_manager import select_assistant_model
from assist.tools import read_url
from assist.checkpoint_rollback import invoke_with_rollback

os.environ.setdefault("ASSIST_MODEL_URL", "http://127.0.0.1:8000/v1")

# A factual lookup with no source in context. To read anything the model must GUESS a URL
# (there is no search tool) — the exact condition that produced the b37c712b fabrication
# loop. The persona/domain is synthetic (no real PII).
_PROMPT = (
    "Find out where the musician Rikki Vale is buried — the exact cemetery and city. "
    "Read the primary sources and confirm the location before answering; it is important "
    "to cite the specific page you found it on."
)

_reads: Counter = Counter()


class _Resp404:
    status_code = 404
    text = "Not Found"

    def raise_for_status(self):
        raise requests.exceptions.HTTPError("404 Client Error: Not Found for url")


def _mock_get(url, **kw):
    # Every fabricated URL 404s — re-creates the incident's "guess -> dead link" condition.
    _reads[url] += 1
    return _Resp404()


def _confined_agent(model, guarded):
    """Minimal agent confined to the fabrication conditions: read_url ONLY (no search), no
    seeded URL. ``guarded`` wires UrlProvenanceMiddleware (the fix); baseline omits it."""
    return create_deep_agent(
        model=model,
        tools=[read_url],
        checkpointer=InMemorySaver(),
        system_prompt=(
            "You research factual questions by reading primary-source web pages. Only pass "
            "read_url a URL that appeared verbatim in a search result. NEVER type or guess a "
            "URL from memory — a guessed URL is a dead link. To reach another page, search "
            "again. If you have no real URL to read, say you could not find sources and stop."),
        middleware=[UrlProvenanceMiddleware()] if guarded else [],
    )


def _run(agent):
    _reads.clear()
    with patch("assist.tools.requests.get", _mock_get), \
         patch("assist.tools._host_throttle", lambda host: None):
        try:
            return invoke_with_rollback(
                agent,
                {"messages": [{"role": "user", "content": _PROMPT}]},
                {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 60})
        except Exception as e:  # recursion cap / timeout mid-loop is itself a runaway signal
            print(f"\n[fabrication-loop] run ended via {type(e).__name__}\n")
            return None


class TestReadUrlFabricationLoop(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def test_fabrication_loop_reproduces_without_provenance(self):
        """BASELINE (no provenance): with no real URL and only read_url, the model
        fabricates URLs and 404-loops — several distinct guessed URLs get fetched."""
        _run(_confined_agent(self.model, guarded=False))
        total = sum(_reads.values())
        print(f"\n[fabrication BASELINE] fetched {total} times across "
              f"{len(_reads)} distinct fabricated URLs: {dict(_reads)}\n")
        self.assertGreater(
            total, 3,
            f"expected a fabricate-404 loop (incident: 108k 404s); got {total} fetches, "
            f"counts={dict(_reads)}. If low, the model gave up on its own — tune the prompt "
            "to induce URL-guessing before relying on the guard.")

    def test_provenance_blocks_fabrication_and_agent_finalizes(self):
        """GUARDED (provenance): every fabricated read_url is refused BEFORE the network,
        so ZERO fetches happen, and the agent finalizes with a non-empty answer rather than
        looping on 404s (finalize-not-kill: the correction tells it to report + stop)."""
        out = _run(_confined_agent(self.model, guarded=True))
        total = sum(_reads.values())
        print(f"\n[fabrication GUARDED] fetched {total} times: {dict(_reads)}\n")
        # The core win: provenance refuses the fabricated URLs before the tool runs, so the
        # 404 flood (and the 101 GB / 90-min runaway) is unreachable by construction.
        self.assertEqual(
            total, 0,
            f"provenance should block every fabricated fetch (0 network reads); got "
            f"{total}, counts={dict(_reads)}")
        # And the agent finalizes with an answer (honest "couldn't find sources"), not an
        # empty stub or a recursion-cap death.
        answer = "" if out is None else str(
            out.get("messages", [])[-1].content if out.get("messages") else "")
        self.assertTrue(
            answer.strip(),
            "agent returned no answer — with fabrications blocked it should report it "
            "couldn't find sources and stop, not spin to the recursion cap")
