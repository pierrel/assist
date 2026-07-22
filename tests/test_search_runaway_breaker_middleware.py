"""Unit tests for SearchRunawayBreakerMiddleware — deterministic, no LLM.

Drives wrap_tool_call with hand-built search_internet histories and asserts each of the four
bounds: total-cap, consecutive-empty, same-query cap, and the empty circuit-break — plus the
finalize/short-circuit return shapes (status, tool_call_id pairing) the design depends on.
"""
from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage

from assist.middleware.search_runaway_breaker import (
    SearchRunawayBreakerMiddleware, _SAME_QUERY_CAP, _TOTAL_CAP, _CONSECUTIVE_EMPTY_CAP,
    _FINALIZE_STEER,
)
from assist.tools import _SEARCH_EMPTY_GUIDANCE

_RESULTS = "[{'title': 't', 'url': 'https://e.test/1', 'content': 'c'}]"  # a non-empty result


def _ai(query, tcid):
    return AIMessage(content="", tool_calls=[{"name": "search_internet",
                                              "args": {"query": query}, "id": tcid}])


def _history(pairs, prefix="h"):
    """pairs = list of (query, result_content); builds AIMessage + its ToolMessage per pair
    with unique tool_call_ids (duplicate ids can't occur in a real history and would mask a
    pairing bug)."""
    msgs = []
    for i, (q, res) in enumerate(pairs):
        tcid = f"{prefix}{i}"
        msgs.append(_ai(q, tcid))
        msgs.append(ToolMessage(content=res, tool_call_id=tcid, name="search_internet"))
    return msgs


def _run(mw, query, history, tcid="new"):
    tool_call = {"name": "search_internet", "args": {"query": query}, "id": tcid}
    req = SimpleNamespace(tool_call=tool_call, state={"messages": history})
    handled = {"ran": False}

    def handler(r):
        handled["ran"] = True
        return ToolMessage(content=_RESULTS, tool_call_id=tcid, name="search_internet")

    return mw.wrap_tool_call(req, handler), handled["ran"]


def test_first_search_passes_through():
    out, ran = _run(SearchRunawayBreakerMiddleware(), "cubetto review", [])
    assert ran and out.content == _RESULTS


def test_non_search_tool_passes_through():
    mw = SearchRunawayBreakerMiddleware()
    tc = {"name": "read_url", "args": {"url": "https://x.test"}, "id": "n"}
    req = SimpleNamespace(tool_call=tc, state={"messages": []})
    ran = {"v": False}

    def handler(r):
        ran["v"] = True
        return ToolMessage(content="page", tool_call_id="n", name="read_url")

    mw.wrap_tool_call(req, handler)
    assert ran["v"]


def test_same_query_cap_refuses_the_fourth_identical():
    # 3 prior identical NON-empty searches (so the empty circuit-break isn't what fires)
    hist = _history([("cubetto specs", _RESULTS)] * _SAME_QUERY_CAP)
    out, ran = _run(SearchRunawayBreakerMiddleware(), "Cubetto  SPECS", hist)  # normalizes equal
    assert not ran
    assert out.status == "error" and "already run this exact search" in out.content
    assert out.tool_call_id == "new"


def test_same_query_under_cap_passes():
    hist = _history([("cubetto specs", _RESULTS)] * (_SAME_QUERY_CAP - 1))
    out, ran = _run(SearchRunawayBreakerMiddleware(), "cubetto specs", hist)
    assert ran


def test_empty_query_repeat_short_circuits_without_backend():
    # one prior EMPTY result for this query → the repeat is served from cache, no handler
    hist = _history([("zorblax qx-9 specs", _SEARCH_EMPTY_GUIDANCE)])
    out, ran = _run(SearchRunawayBreakerMiddleware(), "zorblax qx-9 specs", hist)
    assert not ran                                  # backend NOT hit (circuit broken)
    assert out.content == _SEARCH_EMPTY_GUIDANCE     # byte-identical to a real empty
    assert out.status != "error"                     # a served empty is NOT an error
    assert out.tool_call_id == "new"                 # pairs, so it counts toward next decision


def test_repeated_empty_query_hits_same_query_cap():
    """Empty results ALSO count toward the same-query cap: 3 prior empties of one query → the
    4th is refused-steer (distinct path from the non-empty same-query test above)."""
    q = "nabbo codepal battery"
    hist = _history([(q, _SEARCH_EMPTY_GUIDANCE)] * _SAME_QUERY_CAP)
    out, ran = _run(SearchRunawayBreakerMiddleware(), q, hist)
    assert not ran and out.status == "error" and "already run this exact search" in out.content


def test_malformed_empty_or_null_query_is_counted_not_bypassed():
    """A search_internet(query="") / (query=None) must flow THROUGH counting (not bypass every
    bound keyed on the empty query) and must not crash the normalizer — both normalize to "".
    3 prior malformed empties → the 4th malformed call does not reach the backend."""
    mw = SearchRunawayBreakerMiddleware()
    hist = _history([("", _SEARCH_EMPTY_GUIDANCE), (None, _SEARCH_EMPTY_GUIDANCE),
                     ("", _SEARCH_EMPTY_GUIDANCE)])
    out, ran = _run(mw, "", hist)
    assert not ran and out.tool_call_id == "new"   # bounded, not passed straight to the backend


def test_consecutive_empty_cap_refuses_distinct_reword_hammer():
    # N distinct empty queries in a trailing run → the (N+1)th DISTINCT one is refused-finalize
    hist = _history([(f"zorblax variant {i}", _SEARCH_EMPTY_GUIDANCE)
                     for i in range(_CONSECUTIVE_EMPTY_CAP)])
    out, ran = _run(SearchRunawayBreakerMiddleware(), "zorblax brand new angle", hist)
    assert not ran
    assert out.status == "error" and out.content == _FINALIZE_STEER


def test_consecutive_empty_run_resets_on_a_non_empty_result():
    # 4 empties, then a NON-empty, then 3 more empties → trailing run is 3 (< cap) → passes
    pairs = ([(f"a{i}", _SEARCH_EMPTY_GUIDANCE) for i in range(4)]
             + [("hit", _RESULTS)]
             + [(f"b{i}", _SEARCH_EMPTY_GUIDANCE) for i in range(3)])
    out, ran = _run(SearchRunawayBreakerMiddleware(), "c brand new", _history(pairs))
    assert ran  # consecutive-empty did not fire (run reset); total is well under cap


def test_total_cap_refuses_the_fifty_first_even_when_all_distinct():
    # TOTAL_CAP distinct NON-empty searches → the next distinct search is refused-finalize
    hist = _history([(f"distinct query number {i}", _RESULTS) for i in range(_TOTAL_CAP)])
    out, ran = _run(SearchRunawayBreakerMiddleware(), "one more distinct query", hist)
    assert not ran
    assert out.status == "error" and out.content == _FINALIZE_STEER
    assert out.tool_call_id == "new"


def test_total_cap_under_limit_passes():
    hist = _history([(f"distinct query number {i}", _RESULTS) for i in range(_TOTAL_CAP - 1)])
    out, ran = _run(SearchRunawayBreakerMiddleware(), "still under the cap", hist)
    assert ran
