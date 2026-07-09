"""The search-down circuit breaker stops a turn from re-searching a dead backend —
model-free (the hazard is the grind, not the model).

Mechanism (post-2026-07-08): FINALIZE-not-kill via ``wrap_tool_call``. After
``threshold`` completed ``search_internet`` results have come back as the exact
unavailable constant this turn, the next ``search_internet`` call is REFUSED — the
breaker returns a corrective ``ToolMessage`` (status="error") instead of executing
it, and the handler is NOT called. The turn continues so the model finalizes from
what it gathered (no strip-to-dead-end that discards partial results).

Crucially the failing searches use DISTINCT query args — the case
LoopDetectionMiddleware's same-args pattern does NOT catch, which is why this
middleware earns its place.
"""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain.tools.tool_node import ToolCallRequest

from assist.tools import _SEARCH_UNAVAILABLE_MESSAGE
from assist.middleware.search_unavailable_breaker import (
    SearchUnavailableBreakerMiddleware,
)


def _ai_search(query, call_id):
    return AIMessage(content="", tool_calls=[
        {"name": "search_internet", "args": {"query": query}, "id": call_id}])


def _ai_read(url, call_id):
    return AIMessage(content="", tool_calls=[
        {"name": "read_url", "args": {"url": url}, "id": call_id}])


def _tool(content, call_id):
    return ToolMessage(content=content, tool_call_id=call_id)


class _Handler:
    """Stand-in tool executor: records whether the real search ran."""
    def __init__(self):
        self.called = False

    def __call__(self, request):
        self.called = True
        return ToolMessage(content="EXECUTED",
                           tool_call_id=request.tool_call.get("id", ""),
                           name="search_internet")


def _run(messages, threshold=2, tool_name="search_internet", query="q_next", call_id="cN"):
    """Drive wrap_tool_call with a fresh ``tool_name`` request against ``messages``
    (the completed history). Returns (result, handler). ``result`` is the corrective
    ToolMessage when refused, else the handler's execution result."""
    mw = SearchUnavailableBreakerMiddleware(threshold=threshold)
    handler = _Handler()
    req = ToolCallRequest(
        tool_call={"name": tool_name, "args": {"query": query}, "id": call_id},
        tool=None, state={"messages": messages}, runtime=None)
    return mw.wrap_tool_call(req, handler), handler


def _refused(result, handler):
    return (not handler.called
            and isinstance(result, ToolMessage)
            and result.status == "error"
            and "do NOT search again" in result.content
            and "look this up" in result.content.lower())


def test_refuses_after_threshold_distinct_queries():
    """Two distinct failing searches already completed -> the next search is refused
    with a finalize nudge, and the real search does NOT run."""
    msgs = [
        HumanMessage(content="research persistent emacs over ssh"),
        _ai_search("emacs tramp persistent session", "c1"),
        _tool(_SEARCH_UNAVAILABLE_MESSAGE, "c1"),
        _ai_search("screen vs tmux remote reattach", "c2"),  # DISTINCT args
        _tool(_SEARCH_UNAVAILABLE_MESSAGE, "c2"),
    ]
    result, handler = _run(msgs)
    assert _refused(result, handler), "should refuse the next search and nudge to finalize"


def test_below_threshold_lets_search_run():
    """One failure so far -> the next search is allowed (handler runs)."""
    msgs = [
        HumanMessage(content="research X"),
        _ai_search("q1", "c1"),
        _tool(_SEARCH_UNAVAILABLE_MESSAGE, "c1"),
    ]
    result, handler = _run(msgs)
    assert handler.called and result.content == "EXECUTED"


def test_genuine_empty_results_not_counted():
    """A healthy backend returning ``[]`` (no results) is NOT an outage."""
    msgs = [
        HumanMessage(content="research obscure thing"),
        _ai_search("q1", "c1"), _tool("[]", "c1"),
        _ai_search("q2", "c2"), _tool("[]", "c2"),
    ]
    result, handler = _run(msgs)
    assert handler.called


def test_real_results_not_counted():
    msgs = [
        HumanMessage(content="research X"),
        _ai_search("q1", "c1"),
        _tool("[{'title': 'a hit', 'url': 'https://x', 'content': '...'}]", "c1"),
        _ai_search("q2", "c2"),
        _tool("[{'title': 'another', 'url': 'https://y', 'content': '...'}]", "c2"),
    ]
    result, handler = _run(msgs)
    assert handler.called


def test_partial_results_not_counted():
    """A PARTIAL result (results present + a rate-limit note) is usable, not an
    outage — it must NOT count toward the breaker (the model should keep using it)."""
    partial = ("PARTIAL RESULTS — some search engines were rate-limited (brave). "
               "[{'title': 'a', 'url': 'https://x', 'content': '...'}]")
    msgs = [
        HumanMessage(content="research X"),
        _ai_search("q1", "c1"), _tool(partial, "c1"),
        _ai_search("q2", "c2"), _tool(partial, "c2"),
    ]
    result, handler = _run(msgs)
    assert handler.called, "partial results are usable and must not trip the breaker"


def test_non_search_tool_passes_through():
    """The breaker only governs search_internet — a read_url call is never refused,
    even past the threshold."""
    msgs = [
        HumanMessage(content="research X"),
        _ai_search("q1", "c1"), _tool(_SEARCH_UNAVAILABLE_MESSAGE, "c1"),
        _ai_search("q2", "c2"), _tool(_SEARCH_UNAVAILABLE_MESSAGE, "c2"),
    ]
    result, handler = _run(msgs, tool_name="read_url", call_id="r1")
    assert handler.called


def test_threshold_is_configurable():
    msgs = [
        HumanMessage(content="research X"),
        _ai_search("q1", "c1"), _tool(_SEARCH_UNAVAILABLE_MESSAGE, "c1"),
        _ai_search("q2", "c2"), _tool(_SEARCH_UNAVAILABLE_MESSAGE, "c2"),
    ]
    assert _run(msgs, threshold=3)[1].called            # 2 failures < 3 -> allowed
    assert _refused(*_run(msgs, threshold=2))            # trips at 2


def test_unavailable_results_from_prior_turn_not_counted():
    """A prior turn's outage must not poison a fresh turn."""
    msgs = [
        HumanMessage(content="earlier question"),
        _ai_search("old1", "a1"), _tool(_SEARCH_UNAVAILABLE_MESSAGE, "a1"),
        _ai_search("old2", "a2"), _tool(_SEARCH_UNAVAILABLE_MESSAGE, "a2"),
        AIMessage(content="(relayed unavailable last turn)"),
        HumanMessage(content="new question"),
        _ai_search("new1", "b1"), _tool(_SEARCH_UNAVAILABLE_MESSAGE, "b1"),
    ]
    result, handler = _run(msgs)
    assert handler.called, "only one unavailable this turn -> allowed"


def test_counts_cumulatively_across_heavy_read_interleaving():
    """Cumulative over the whole turn, not a recency window: many read_url calls
    interleaved between failed searches must not push earlier unavailable out of view."""
    msgs = [HumanMessage(content="research X")]
    cid = 0
    for s in range(4):
        msgs.append(_ai_search(f"q{s}", f"s{cid}"))
        msgs.append(_tool(_SEARCH_UNAVAILABLE_MESSAGE, f"s{cid}"))
        cid += 1
        for r in range(5):
            msgs.append(_ai_read(f"https://x/{s}/{r}", f"r{cid}"))
            msgs.append(_tool("Error fetching URL: down", f"r{cid}"))
            cid += 1
    result, handler = _run(msgs, threshold=4)
    assert _refused(result, handler), "cumulative count must trip despite interleaving"


def test_threshold_floored_at_one():
    """A misconfigured 0/negative threshold is clamped to 1 — never refuses a healthy
    (zero-failure) search."""
    assert SearchUnavailableBreakerMiddleware(threshold=0).threshold == 1
    assert SearchUnavailableBreakerMiddleware(threshold=-5).threshold == 1
    healthy = [HumanMessage(content="x")]  # 0 unavailable this turn
    result, handler = _run(healthy, threshold=0)
    assert handler.called


def test_production_default_threshold_is_backstop_value():
    """The shipped default (4) is a HARD backstop above where the prompt should stop
    the model — guards against a careless change back to a trip-happy 1-2."""
    assert SearchUnavailableBreakerMiddleware().threshold == 4


def test_default_threshold_allows_three_failures():
    """At the default (4), three failed searches do NOT yet refuse the next."""
    msgs = [HumanMessage(content="research X")]
    for i in range(3):
        msgs.append(_ai_search(f"q{i}", f"c{i}"))
        msgs.append(_tool(_SEARCH_UNAVAILABLE_MESSAGE, f"c{i}"))
    result, handler = _run(msgs, threshold=4)
    assert handler.called, "only 3 completed unavailable -> the 4th is still allowed"
