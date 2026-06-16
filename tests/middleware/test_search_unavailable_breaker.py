"""The search-down circuit breaker terminates a turn that keeps querying a
dead backend — model-free (the hazard is the grind, not the model).

Asserts the SYMPTOM: when ``search_internet`` has returned the exact
unavailable constant ``threshold`` times and the model asks for ANOTHER search,
``after_model`` emits a terminal AIMessage with NO tool calls (so the agent loop
ends instead of grinding) whose content relays the search-unavailable status.

Crucially the failing searches use DISTINCT query args — that is the case
LoopDetectionMiddleware's same-args pattern does NOT catch, which is exactly why
this middleware earns its place.
"""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from assist.tools import _SEARCH_UNAVAILABLE_MESSAGE
from assist.middleware.search_unavailable_breaker import (
    SearchUnavailableBreakerMiddleware,
)


def _ai_search(query, call_id):
    """An AIMessage requesting a search_internet call with a DISTINCT query."""
    return AIMessage(content="", tool_calls=[
        {"name": "search_internet", "args": {"query": query}, "id": call_id}])


def _ai_read(url, call_id):
    return AIMessage(content="", tool_calls=[
        {"name": "read_url", "args": {"url": url}, "id": call_id}])


def _tool(content, call_id):
    return ToolMessage(content=content, tool_call_id=call_id)


def _run(messages, threshold=2):
    mw = SearchUnavailableBreakerMiddleware(threshold=threshold)
    return mw.after_model({"messages": messages}, None)


def test_terminates_after_threshold_distinct_queries():
    """Two distinct failing searches, then a third requested -> turn ends."""
    msgs = [
        HumanMessage(content="research persistent emacs over ssh"),
        _ai_search("emacs tramp persistent session", "c1"),
        _tool(_SEARCH_UNAVAILABLE_MESSAGE, "c1"),
        _ai_search("screen vs tmux remote reattach", "c2"),  # DISTINCT args
        _tool(_SEARCH_UNAVAILABLE_MESSAGE, "c2"),
        _ai_search("mosh persistent shell", "c3"),           # the (N+1)th
    ]
    result = _run(msgs)
    assert result is not None, "should terminate the turn"
    out = result["messages"][0]
    assert out.tool_calls == [], "tool calls must be stripped so the loop ends"
    assert not out.additional_kwargs.get("tool_calls"), "raw tool_calls stripped too"
    assert "unavailable" in out.content.lower()
    assert "look this up" in out.content.lower()  # trips the orchestrator relay branch


def test_below_threshold_does_not_terminate():
    """One failure + another search requested -> let it try again."""
    msgs = [
        HumanMessage(content="research X"),
        _ai_search("q1", "c1"),
        _tool(_SEARCH_UNAVAILABLE_MESSAGE, "c1"),
        _ai_search("q2", "c2"),
    ]
    assert _run(msgs) is None


def test_genuine_empty_results_not_counted():
    """A healthy backend returning ``[]`` (no results) is NOT an outage — must
    not be counted, even across the threshold."""
    msgs = [
        HumanMessage(content="research obscure thing"),
        _ai_search("q1", "c1"),
        _tool("[]", "c1"),
        _ai_search("q2", "c2"),
        _tool("[]", "c2"),
        _ai_search("q3", "c3"),
    ]
    assert _run(msgs) is None


def test_real_results_not_counted():
    """Successful searches never trigger the breaker."""
    msgs = [
        HumanMessage(content="research X"),
        _ai_search("q1", "c1"),
        _tool("[{'title': 'a hit', 'url': 'https://x', 'content': '...'}]", "c1"),
        _ai_search("q2", "c2"),
        _tool("[{'title': 'another', 'url': 'https://y', 'content': '...'}]", "c2"),
        _ai_search("q3", "c3"),
    ]
    assert _run(msgs) is None


def test_model_moved_on_to_other_tool_not_terminated():
    """Threshold reached, but the latest message calls a DIFFERENT tool — the
    model is already breaking out, so don't cut it off."""
    msgs = [
        HumanMessage(content="research X"),
        _ai_search("q1", "c1"),
        _tool(_SEARCH_UNAVAILABLE_MESSAGE, "c1"),
        _ai_search("q2", "c2"),
        _tool(_SEARCH_UNAVAILABLE_MESSAGE, "c2"),
        _ai_read("https://example.com", "c3"),  # not a search
    ]
    assert _run(msgs) is None


def test_threshold_is_configurable():
    """With threshold=3, two failures + a third request is still allowed."""
    msgs = [
        HumanMessage(content="research X"),
        _ai_search("q1", "c1"),
        _tool(_SEARCH_UNAVAILABLE_MESSAGE, "c1"),
        _ai_search("q2", "c2"),
        _tool(_SEARCH_UNAVAILABLE_MESSAGE, "c2"),
        _ai_search("q3", "c3"),
    ]
    assert _run(msgs, threshold=3) is None
    assert _run(msgs, threshold=2) is not None  # same tail trips at 2


def test_unavailable_results_from_prior_turn_not_counted():
    """A prior turn's outage must not poison a fresh turn — the per-turn slice
    starts at the most recent HumanMessage."""
    msgs = [
        # prior turn: search was down
        HumanMessage(content="earlier question"),
        _ai_search("old1", "a1"),
        _tool(_SEARCH_UNAVAILABLE_MESSAGE, "a1"),
        _ai_search("old2", "a2"),
        _tool(_SEARCH_UNAVAILABLE_MESSAGE, "a2"),
        AIMessage(content="(relayed unavailable last turn)"),
        # new turn: only one failure so far
        HumanMessage(content="new question"),
        _ai_search("new1", "b1"),
        _tool(_SEARCH_UNAVAILABLE_MESSAGE, "b1"),
        _ai_search("new2", "b2"),
    ]
    assert _run(msgs) is None


def test_no_tool_calls_or_empty_returns_none():
    assert _run([]) is None
    assert _run([HumanMessage(content="hi")]) is None
    assert _run([HumanMessage(content="hi"), AIMessage(content="done, no tools")]) is None


def test_production_default_threshold_is_backstop_value():
    """The shipped default (4) is a HARD backstop set ABOVE where the prompt
    should make the model stop on its own — raise/lower deliberately, the live
    eval pins the actual value.  Guards against a careless change back to a
    trip-happy 1-2."""
    mw = SearchUnavailableBreakerMiddleware()
    assert mw.threshold == 4


def test_default_threshold_allows_more_before_breaking():
    """At the default (4), three failed searches do NOT yet trip — the prompt
    gets room to stop the model first."""
    msgs = [HumanMessage(content="research X")]
    for i in range(3):
        msgs.append(_ai_search(f"q{i}", f"c{i}"))
        msgs.append(_tool(_SEARCH_UNAVAILABLE_MESSAGE, f"c{i}"))
    msgs.append(_ai_search("q_next", "c_next"))  # the 4th request, only 3 completed
    mw = SearchUnavailableBreakerMiddleware()  # default threshold 4
    assert mw.after_model({"messages": msgs}, None) is None


def test_terminal_message_replaces_in_place_not_duplicated():
    """The returned message carries the original AIMessage's id, so the messages
    reducer REPLACES it in place — the merged history must not keep both the
    tool-call-bearing message AND the terminal one (a duplicate would leave the
    stripped tool call to re-dispatch)."""
    from langgraph.graph.message import add_messages

    last = AIMessage(content="", id="ai_last", tool_calls=[
        {"name": "search_internet", "args": {"query": "q3"}, "id": "c3"}])
    msgs = [
        HumanMessage(content="research X", id="h1"),
        _ai_search("q1", "c1"), _tool(_SEARCH_UNAVAILABLE_MESSAGE, "c1"),
        _ai_search("q2", "c2"), _tool(_SEARCH_UNAVAILABLE_MESSAGE, "c2"),
        last,
    ]
    result = _run(msgs)
    assert result is not None
    merged = add_messages(msgs, result["messages"])
    # exactly one message with the last AIMessage's id, and it has no tool calls
    same_id = [m for m in merged if getattr(m, "id", None) == "ai_last"]
    assert len(same_id) == 1, "tool-call message was duplicated, not replaced"
    assert same_id[0].tool_calls == [], "the surviving message still has tool calls"
    assert merged[-1].id == "ai_last", "terminal message is not the last message"
