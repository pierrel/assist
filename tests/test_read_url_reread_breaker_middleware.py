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
