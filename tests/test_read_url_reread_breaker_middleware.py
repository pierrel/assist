"""Unit tests for ReadUrlRereadBreaker — the deterministic same-URL re-read bound.

Drives wrap_tool_call with a hand-built message history of prior read_url calls and asserts:
the (max_reads+1)th fetch of the same URL is refused with a corrective result, while an
under-threshold fetch, a different URL, and a non-read_url tool pass through. No LLM.
"""
from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage

from assist.middleware.read_url_reread_breaker import ReadUrlRereadBreaker


def _ai_read(url, tcid="c"):
    return AIMessage(content="", tool_calls=[{"name": "read_url", "args": {"url": url}, "id": tcid}])


def _req(url, tcid="new"):
    tool_call = {"name": "read_url", "args": {"url": url}, "id": tcid}
    return SimpleNamespace(tool_call=tool_call, state=None), tool_call


def _history(url, n):
    """A message history with n prior read_url calls to url (AIMessage + its ToolMessage)."""
    msgs = []
    for i in range(n):
        msgs.append(_ai_read(url, f"c{i}"))
        msgs.append(ToolMessage(content="page text", tool_call_id=f"c{i}", name="read_url"))
    return msgs


def _run(mw, url, history):
    req, tc = _req(url)
    req.state = {"messages": history}
    handled = {"ran": False}

    def handler(r):
        handled["ran"] = True
        return ToolMessage(content="fresh page", tool_call_id=tc["id"], name="read_url")

    return mw.wrap_tool_call(req, handler), handled["ran"]


_URL = "https://pubmed.ncbi.nlm.nih.gov/36650300/"


def test_refuses_once_over_max_reads():
    mw = ReadUrlRereadBreaker(max_reads=3)
    out, ran = _run(mw, _URL, _history(_URL, 3))   # already read 3x → this is the 4th
    assert not ran                                  # the tool did NOT run
    assert out.status == "error" and "already read" in out.content
    assert "do not keep retrying" in out.content.lower()


def test_allows_under_max_reads():
    mw = ReadUrlRereadBreaker(max_reads=3)
    out, ran = _run(mw, _URL, _history(_URL, 2))   # read 2x → the 3rd is allowed
    assert ran and out.content == "fresh page"


def test_different_url_not_counted():
    mw = ReadUrlRereadBreaker(max_reads=3)
    # 3 prior reads of a DIFFERENT url must not block this url's first read
    out, ran = _run(mw, _URL, _history("https://example.com/other", 3))
    assert ran


def test_normalized_url_matches_across_trivial_variants():
    mw = ReadUrlRereadBreaker(max_reads=3)
    # trailing slash / fragment differences are the SAME url (normalize_url)
    history = _history(_URL, 2) + _history(_URL.rstrip("/") + "#sec", 1)
    out, ran = _run(mw, _URL, history)             # 3 prior (normalized) → 4th refused
    assert not ran and out.status == "error"


def test_non_read_url_tool_passes_through():
    mw = ReadUrlRereadBreaker(max_reads=3)
    req = SimpleNamespace(
        tool_call={"name": "search_internet", "args": {"query": "x"}, "id": "s"},
        state={"messages": _history(_URL, 9)})
    ran = {"v": False}

    def handler(r):
        ran["v"] = True
        return ToolMessage(content="ok", tool_call_id="s", name="search_internet")

    mw.wrap_tool_call(req, handler)
    assert ran["v"]


def test_missing_url_passes_through():
    mw = ReadUrlRereadBreaker(max_reads=3)
    req = SimpleNamespace(tool_call={"name": "read_url", "args": {}, "id": "x"}, state={"messages": []})
    ran = {"v": False}

    def handler(r):
        ran["v"] = True
        return ToolMessage(content="ok", tool_call_id="x", name="read_url")

    mw.wrap_tool_call(req, handler)
    assert ran["v"]


def test_current_inflight_call_not_counted():
    # At wrap_tool_call time the state already contains the AIMessage carrying the call
    # being executed (no ToolMessage yet). It must NOT count — otherwise the threshold
    # shifts by one and prod refuses a read these tests say is allowed.
    mw = ReadUrlRereadBreaker(max_reads=3)
    history = _history(_URL, 2) + [_ai_read(_URL, "current")]   # 2 completed + the in-flight call
    out, ran = _run(mw, _URL, history)
    assert ran and out.content == "fresh page"                  # 3rd fetch allowed


def test_parallel_same_url_siblings_all_allowed_on_first_fetch():
    # One AIMessage with several same-URL calls and NO completed reads: siblings have no
    # ToolMessage yet, so none count — the guard must never refuse a FIRST fetch.
    mw = ReadUrlRereadBreaker(max_reads=3)
    parallel = AIMessage(content="", tool_calls=[
        {"name": "read_url", "args": {"url": _URL}, "id": f"p{i}"} for i in range(3)])
    out, ran = _run(mw, _URL, [parallel])
    assert ran


def test_refusals_keep_the_cap_engaged():
    # A refusal's corrective ToolMessage pairs with its call, so the completed count
    # stays at/over the cap and later retries stay refused.
    mw = ReadUrlRereadBreaker(max_reads=3)
    history = _history(_URL, 3)
    history.append(_ai_read(_URL, "r1"))
    history.append(ToolMessage(content="You have already read ...", tool_call_id="r1",
                               name="read_url", status="error"))
    out, ran = _run(mw, _URL, history)
    assert not ran and out.status == "error"
