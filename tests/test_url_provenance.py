"""Unit tests for UrlProvenanceMiddleware — the research read_url guard.

Deterministic (no LLM): drive wrap_tool_call directly with a constructed
ToolCallRequest and assert allow (handler invoked) vs reject (corrective
ToolMessage, handler NOT invoked).
"""
from unittest import TestCase

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain.tools.tool_node import ToolCallRequest

from assist.middleware.url_provenance import UrlProvenanceMiddleware, normalize_url


def _search_result(urls):
    items = [{"title": "t", "url": u, "content": "snippet"} for u in urls]
    return ToolMessage(content=str(items), name="search_internet", tool_call_id="s1")


class _Handler:
    """Stand-in tool executor: records invocation, returns a sentinel."""
    def __init__(self):
        self.called_with = None

    def __call__(self, request):
        self.called_with = request
        return ToolMessage(content="FETCHED", tool_call_id=request.tool_call.get("id", ""),
                           name="read_url")


def _request(url, messages):
    return ToolCallRequest(
        tool_call={"name": "read_url", "args": {"url": url}, "id": "r1"},
        tool=None,
        state={"messages": messages},
        runtime=None,
    )


class TestUrlProvenanceMiddleware(TestCase):
    def setUp(self):
        self.mw = UrlProvenanceMiddleware()

    def _call(self, url, messages):
        handler = _Handler()
        result = self.mw.wrap_tool_call(_request(url, messages), handler)
        return result, handler

    def test_allows_url_from_a_search_result(self):
        msgs = [HumanMessage(content="find casio watches"),
                _search_result(["https://shop.example/f91w", "https://shop.example/la670"])]
        result, handler = self._call("https://shop.example/f91w", msgs)
        self.assertIsNotNone(handler.called_with, "should pass a provenanced URL through")
        self.assertEqual(result.content, "FETCHED")

    def test_rejects_fabricated_url(self):
        msgs = [_search_result(["https://shop.example/f91w"])]
        result, handler = self._call("https://www.casio.com/watch/kids/f-91w/", msgs)
        self.assertIsNone(handler.called_with, "fabricated URL must NOT reach the tool")
        self.assertEqual(result.status, "error")
        self.assertIn("not fetched", result.content)
        self.assertIn("https://shop.example/f91w", result.content)  # lists a real one

    def test_allows_user_provided_url(self):
        # A URL in the question itself is provenanced (copyable, not invented).
        msgs = [HumanMessage(content="summarize https://blog.example/post-1 for me")]
        _result, handler = self._call("https://blog.example/post-1", msgs)
        self.assertIsNotNone(handler.called_with)

    def test_allows_link_followed_from_a_fetched_page(self):
        # A URL found inside a page the agent already read is legitimate.
        msgs = [_search_result(["https://shop.example/index"]),
                ToolMessage(content="See also https://shop.example/deep/page-2 for specs.",
                            name="read_url", tool_call_id="r0")]
        _result, handler = self._call("https://shop.example/deep/page-2", msgs)
        self.assertIsNotNone(handler.called_with)

    def test_normalizes_trailing_slash(self):
        msgs = [_search_result(["https://shop.example/f91w"])]
        _result, handler = self._call("https://shop.example/f91w/", msgs)
        self.assertIsNotNone(handler.called_with, "trailing-slash variant should match")

    def test_passes_non_read_url_tools_through(self):
        handler = _Handler()
        req = ToolCallRequest(
            tool_call={"name": "search_internet", "args": {"query": "x"}, "id": "s2"},
            tool=None, state={"messages": []}, runtime=None)
        self.mw.wrap_tool_call(req, handler)
        self.assertIsNotNone(handler.called_with, "non-read_url tools are untouched")

    def test_no_search_yet_rejects(self):
        # read_url before any provenance source -> reject (must search first).
        result, handler = self._call("https://www.casio.com/x", [HumanMessage(content="hi")])
        self.assertIsNone(handler.called_with)
        self.assertEqual(result.status, "error")

    def test_empty_url_passes_through(self):
        # Don't second-guess a malformed call; let the tool surface its own error.
        _result, handler = self._call("", [_search_result(["https://shop.example/a"])])
        self.assertIsNotNone(handler.called_with)
