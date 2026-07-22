"""Reproduce the research search RUNAWAY (2026-07-18 coding-toy incident).

Symptom: the searcher issued THREE obscure product-spec queries ~80 times each, ALL
returning healthy-empty results (SearXNG up, genuinely no hits), 346 searches in 33 min.
loop_detection can't catch it (the three queries cycle A/B/C — non-consecutive); the
search-unavailable breaker can't (the backend is UP, results are just empty). See the
search-runaway design doc.

To reproduce it reliably (not the stochastic full research flow) this uses a MINIMAL agent
confined to the exact conditions: tools=[search_internet, read_url] against a MOCKED SearXNG
that returns healthy-empty (``{"results": [], "unresponsive_engines": []}``) for the obscure
item — so the model searches, gets nothing, and (pre-fix) re-searches the same dead query.

Fully mocked + Tavily-free by construction: ``requests.get`` (SearXNG) is mocked to empty;
healthy-empty never reaches the Tavily branch (``saw_down`` stays False), AND ``requests.post``
(Tavily) is asserted never-called, AND TAVILY_API_KEY is cleared. Only the LLM is real, so
runs vary (like all evals here). The BASELINE (no breaker) asserts the loop REPRODUCES; the
GUARDED tests assert the breaker bounds it and the agent still answers (finalize-not-kill);
the REGRESSION test asserts legit broad research (many DISTINCT result-returning queries) is
NOT clipped.
"""
import os
import uuid
from collections import Counter
from unittest import TestCase
from unittest.mock import patch

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage

from assist.middleware.search_runaway_breaker import (
    SearchRunawayBreakerMiddleware, _normalize_query, _SAME_QUERY_CAP, _TOTAL_CAP,
)
from assist.model_manager import select_assistant_model
from assist.tools import read_url, search_internet
from assist.checkpoint_rollback import invoke_with_rollback

os.environ.setdefault("ASSIST_MODEL_URL", "http://127.0.0.1:8000/v1")

# Synthetic, obscure items a healthy search genuinely can't find — no real products, no PII
# (the incident's real items were niche coding toys). Three of them, to recreate the
# incident's A/B/C-cycling multi-product comparison pressure that drove the runaway.
_ITEM = "Zorblax QX-9 coding toy robot"
_ITEMS = ("Zorblax QX-9", "Glimworks Tuktuk", "Nabbo Codepal")

_get_queries: Counter = Counter()   # backend SearXNG GETs, keyed by normalized query
_post_calls: list = []              # Tavily POSTs — must stay empty


class _EmptyResp:
    """Healthy SearXNG with genuinely no results (never triggers the Tavily fallback)."""
    def json(self):
        return {"results": [], "unresponsive_engines": []}

    def raise_for_status(self):
        pass


class _ResultsResp:
    def __init__(self, query):
        self._query = query

    def json(self):
        return {"results": [
            {"title": f"{self._query} — overview", "url": f"https://example.test/{abs(hash(self._query)) % 9999}",
             "content": f"Reference material about {self._query}."}],
            "unresponsive_engines": []}

    def raise_for_status(self):
        pass


def _mock_get_empty(url, **kw):
    _get_queries[_normalize_query(kw.get("params", {}).get("q", ""))] += 1
    return _EmptyResp()


def _mock_get_results(url, **kw):
    q = kw.get("params", {}).get("q", "")
    _get_queries[_normalize_query(q)] += 1
    return _ResultsResp(q)


def _mock_post(*a, **kw):  # Tavily — record any call so the test can assert it never happens
    _post_calls.append(a)
    raise AssertionError("Tavily (requests.post) must not be called in this eval")


def _confined_agent(model, extra_middleware=()):
    """Minimal searcher: search + read only, no orchestrator. extra_middleware wires the
    breaker for the guarded tests."""
    return create_deep_agent(
        model=model,
        tools=[search_internet, read_url],
        checkpointer=InMemorySaver(),
        system_prompt=("You are a research worker. Find the requested facts by searching the "
                       "web. Report only what your sources say; be persistent and precise."),
        middleware=[*extra_middleware],
    )


def _prompt(items):
    names = ", ".join(f"the '{i}'" for i in items)
    return (f"Research and compare the exact product specifications — battery type, "
            f"dimensions, weight, and price — of these coding toys for kids: {names}. All are "
            f"real products sold online. It is important to report PRECISE specs for each; "
            f"search thoroughly from multiple angles and keep trying for each product before "
            f"giving up. Report the specs you find for each.")


def _search_emissions(out):
    """search_internet tool_calls the model EMITTED, by normalized query (includes ones the
    breaker short-circuited/refused — those still appear as tool_calls in history)."""
    c: Counter = Counter()
    if not out:
        return c
    for m in out.get("messages", []):
        for tc in (getattr(m, "tool_calls", None) or []) if isinstance(m, AIMessage) else []:
            if tc.get("name") == "search_internet":
                c[_normalize_query((tc.get("args") or {}).get("query", ""))] += 1
    return c


def _final_text(out):
    if not out or not out.get("messages"):
        return ""
    return str(out["messages"][-1].content or "")


def _run(agent, mock_get, items=_ITEMS):
    _get_queries.clear()
    _post_calls.clear()
    with patch("assist.tools.requests.get", mock_get), \
         patch("assist.tools.requests.post", _mock_post), \
         patch("assist.tools._host_throttle", lambda host: None), \
         patch.dict(os.environ, {"ASSIST_SEARCH_URL": "http://searxng.test"}, clear=False):
        os.environ.pop("TAVILY_API_KEY", None)
        try:
            return invoke_with_rollback(
                agent,
                {"messages": [{"role": "user", "content": _prompt(items)}]},
                {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 70})
        except Exception as e:  # recursion cap / LLM timeout mid-loop = a runaway signal
            print(f"\n[search-runaway] run ended via {type(e).__name__}\n")
            return None


class TestSearchRunawayLoop(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def test_search_runaway_reproduces(self):
        """BASELINE (no breaker): the same dead query is searched more than a few times, or
        the searcher fires a lot of total searches — the incident shape."""
        out = _run(_confined_agent(self.model), _mock_get_empty)
        emitted = _search_emissions(out)
        worst = emitted.most_common(1)[0][1] if emitted else 0
        total = sum(emitted.values())
        # _get_queries (backend GETs) survives even when a hard runaway hits recursion_limit
        # and _run returns None (emissions would then read empty) — so a severe reproduction
        # counts as a reproduction, not a false failure.
        total_backend = sum(_get_queries.values())
        print(f"\n[search-runaway BASELINE] emissions={dict(emitted)} worst={worst} "
              f"total={total} backend_gets={total_backend} out_is_none={out is None}\n")
        self.assertFalse(_post_calls, "Tavily must not be hit")
        self.assertTrue(
            out is None or worst > _SAME_QUERY_CAP or total > 10 or total_backend > 10,
            f"expected a search runaway (incident: ~80x/query); got worst={worst} total={total} "
            f"backend={total_backend}. If low, the model gave up gracefully — tune the "
            f"prompt/item to induce the loop.")

    def test_breaker_bounds_runaway_and_agent_answers(self):
        """WITH the breaker: no single query is *backend-searched* past one sweep (circuit
        break), total stays bounded, and the agent still answers (finalize-not-kill)."""
        out = _run(_confined_agent(self.model, (SearchRunawayBreakerMiddleware(),)), _mock_get_empty)
        emitted = _search_emissions(out)
        total = sum(emitted.values())
        # circuit break: each distinct query hits the backend at most ONE sweep (<=6 engine
        # GETs), no matter how many times the model re-emitted it.
        worst_backend = _get_queries.most_common(1)[0][1] if _get_queries else 0
        print(f"\n[search-runaway GUARDED] emissions={dict(emitted)} total={total} "
              f"backend_gets={dict(_get_queries)} worst_backend={worst_backend}\n")
        self.assertFalse(_post_calls, "Tavily must not be hit")
        self.assertLessEqual(worst_backend, 6,
                             f"circuit-break failed: a query hit the backend {worst_backend} times "
                             f"(>1 sweep of 6 engines); counts={dict(_get_queries)}")
        self.assertLessEqual(total, _TOTAL_CAP + 6,
                             f"total search emissions {total} exceeded cap+batch; {dict(emitted)}")
        self.assertTrue(_final_text(out).strip(),
                        "agent returned no answer — breaker should finalize (nudge), not kill")

    def test_legit_breadth_not_clipped(self):
        """REGRESSION: a run where searches RETURN results (distinct queries) must not trip
        the breaker — the model researches to a real answer, no refusal/short-circuit."""
        out = _run(_confined_agent(self.model, (SearchRunawayBreakerMiddleware(),)),
                   _mock_get_results, items=("Acme Rover 3 educational robot",))
        emitted = _search_emissions(out)
        print(f"\n[search-runaway REGRESSION] emissions={dict(emitted)} "
              f"backend_gets={dict(_get_queries)}\n")
        # every distinct query reached the backend (nothing short-circuited), and the model
        # produced an answer — breadth intact.
        self.assertTrue(_final_text(out).strip(), "agent produced no answer on a healthy run")
        self.assertLessEqual(sum(emitted.values()), _TOTAL_CAP,
                             "a healthy distinct-query run should not approach the total cap")
