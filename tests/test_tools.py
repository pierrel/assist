"""Tests for the web tools in assist.tools.

``search_internet`` goes through a self-hosted SearXNG instance with NO
fallback — if SearXNG is unset/unreachable/erroring, or returns zero results
while reporting any engine failures, it RETURNS ``_SEARCH_UNAVAILABLE_MESSAGE``
(loud + logged, not raised, so the agent can relay the outage).  The HTTP
call is mocked so no network is involved.  ``read_url``'s per-host throttle
is tested by patching the module's ``time``.  Each test resets module state
in a fixture so order-of-execution doesn't matter."""
from __future__ import annotations

import ast

import pytest
from unittest.mock import patch, MagicMock

from assist import tools


@pytest.fixture(autouse=True)
def _reset_tool_state(monkeypatch):
    """Reset per-host throttle state and clear ASSIST_SEARCH_URL so each test
    starts from a known config (SearXNG tests set it explicitly)."""
    tools._host_last_call.clear()
    monkeypatch.delenv("ASSIST_SEARCH_URL", raising=False)
    yield


def _resp(json_payload):
    """A fake requests.Response: .json() returns the payload, and
    .raise_for_status() is a no-op (MagicMock auto-creates it; set it
    explicitly so the success path can't accidentally raise)."""
    r = MagicMock()
    r.json.return_value = json_payload
    r.raise_for_status.return_value = None
    return r


def _html_resp(html):
    """A fake requests.Response carrying HTML in ``.text`` (for read_url)."""
    r = MagicMock()
    r.text = html
    r.raise_for_status.return_value = None
    return r


# -------------------- _host_throttle (read_url) -------------------------

class TestHostThrottle:
    def test_first_call_to_a_host_does_not_sleep(self):
        with patch.object(tools, "time") as t:
            t.time.return_value = 1000.0
            tools._host_throttle("example.com")
            t.sleep.assert_not_called()

    def test_second_call_to_same_host_within_window_sleeps(self):
        with patch.object(tools, "time") as t:
            # call 1 (not slept) = 1000.0; call 2 (slept): one before-sleep
            # check, one after-sleep to record the actual call time.
            t.time.side_effect = [1000.0, 1000.3, 1000.3]
            tools._host_throttle("example.com")
            tools._host_throttle("example.com")
            t.sleep.assert_called_once()
            slept = t.sleep.call_args.args[0]
            assert slept == pytest.approx(tools._HOST_MIN_DELAY - 0.3)

    def test_different_hosts_do_not_block_each_other(self):
        """Per-host point: a burst of fetches to distinct sites isn't
        artificially serialised."""
        with patch.object(tools, "time") as t:
            t.time.side_effect = [1000.0, 1000.0, 1000.1, 1000.1]
            tools._host_throttle("example.com")
            tools._host_throttle("other.com")
            t.sleep.assert_not_called()

    def test_empty_host_is_noop(self):
        with patch.object(tools, "time") as t:
            tools._host_throttle("")
            t.sleep.assert_not_called()
            t.time.assert_not_called()

    def test_host_dict_pruned_when_over_threshold(self):
        """Over-threshold dict size triggers a prune that drops entries older
        than _HOST_DICT_PRUNE_KEEP_S, bounding memory in a long-running
        process that touches many distinct hosts (Copilot PR #118 review #1)."""
        threshold = tools._HOST_DICT_PRUNE_THRESHOLD
        keep_s = tools._HOST_DICT_PRUNE_KEEP_S
        now_anchor = 10_000.0
        for i in range(threshold // 2 + 5):
            tools._host_last_call[f"stale-{i}.example"] = now_anchor - keep_s - 10
        for i in range(threshold // 2 + 5):
            tools._host_last_call[f"fresh-{i}.example"] = now_anchor - 1
        assert len(tools._host_last_call) > threshold
        with patch.object(tools.time, "time", return_value=now_anchor), \
             patch.object(tools.time, "sleep"):
            tools._host_throttle("new-host.example")
        assert not any(h.startswith("stale-") for h in tools._host_last_call)
        assert all(h.startswith("fresh-") or h == "new-host.example"
                   for h in tools._host_last_call)
        assert len(tools._host_last_call) == threshold // 2 + 5 + 1


# -------------------- search_internet (SearXNG, no fallback) ------------

class TestSearchInternet:
    URL = "http://127.0.0.1:8890"

    def test_unset_search_url_returns_unavailable(self):
        """No ASSIST_SEARCH_URL is a misconfiguration — surfaced loudly as the
        unavailable message (logged ERROR), not a silent no-op and not an
        exception that crashes the turn."""
        assert tools.search_internet("anything") == tools._SEARCH_UNAVAILABLE_MESSAGE

    def test_returns_normalized_results(self, monkeypatch):
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        payload = {"results": [
            {"title": "T1", "url": "https://a.com", "content": "c1"},
            {"title": "T2", "url": "https://b.com", "content": "c2"},
        ]}
        with patch.object(tools, "requests") as req:
            req.get.return_value = _resp(payload)
            result = tools.search_internet("q")
        assert "https://a.com" in result and "https://b.com" in result
        assert "c1" in result and "T2" in result

    def test_queries_searxng_json_endpoint(self, monkeypatch):
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        with patch.object(tools, "requests") as req:
            req.get.return_value = _resp({"results": [
                {"title": "x", "url": "https://e.com", "content": "y"}]})
            tools.search_internet("hello world")
            args, kwargs = req.get.call_args
            assert args[0] == self.URL + "/search"
            assert kwargs["params"]["q"] == "hello world"
            assert kwargs["params"]["format"] == "json"

    def test_respects_max_results(self, monkeypatch):
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        payload = {"results": [
            {"title": f"T{i}", "url": f"https://s{i}.com", "content": "c"}
            for i in range(10)
        ]}
        with patch.object(tools, "requests") as req:
            req.get.return_value = _resp(payload)
            result = tools.search_internet("q", max_results=3)
        # Parse rather than string-count so the test is resilient to repr
        # formatting (quotes/spacing/key order).
        assert len(ast.literal_eval(result)) == 3

    def test_genuine_empty_returns_guidance(self, monkeypatch):
        """Zero results with NO engine failures is a real 'no results' answer, not a backend
        failure — return the empty-guidance steer (was a bare '[]', which the small model
        re-hammered) so the agent pivots instead of re-running the dead query."""
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        with patch.object(tools, "requests") as req:
            req.get.return_value = _resp({"results": [], "unresponsive_engines": []})
            assert tools.search_internet("obscure") == tools._SEARCH_EMPTY_GUIDANCE

    # --- Backend-failure modes: return the unavailable MESSAGE (loud, logged),
    # NOT raise.  Raising would crash the research turn; the agent must receive
    # a tool result it can relay ("couldn't search — unavailable").  Each case
    # is a distinct malformed/broken-backend shape that must not be read as a
    # genuine "no results".

    def test_empty_with_failed_engines_returns_unavailable(self, monkeypatch):
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        with patch.object(tools, "requests") as req:
            req.get.return_value = _resp({
                "results": [],
                "unresponsive_engines": [["google", "timeout"], ["brave", "CAPTCHA"]],
            })
            assert tools.search_internet("q") == tools._SEARCH_UNAVAILABLE_MESSAGE

    def test_rotation_returns_first_engine_with_results(self, monkeypatch):
        """The engine rotation returns the FIRST engine (in order) that yields results,
        without querying the rest."""
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        queried = []

        def fake_get(url, params=None, timeout=None, **kw):
            eng = (params or {}).get("engines")
            queried.append(eng)
            return _resp({"results": [{"title": "T", "url": "https://a.com", "content": "c"}],
                          "unresponsive_engines": []})

        with patch.object(tools, "requests") as req:
            req.get.side_effect = fake_get
            result = tools.search_internet("q")
        assert "https://a.com" in result
        assert queried == [tools._ROTATION_ENGINES[0]]  # stopped after the first

    def test_rotation_falls_through_down_engines_to_a_working_one(self, monkeypatch):
        """Down/throttled engines are skipped; the first WORKING engine's results win —
        this is the fix for the fan-out starving a working-but-slower engine (mojeek)."""
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)

        def fake_get(url, params=None, timeout=None, **kw):
            eng = (params or {}).get("engines")
            if eng == "mojeek":
                return _resp({"results": [{"title": "M", "url": "https://m.com", "content": "c"}],
                              "unresponsive_engines": []})
            return _resp({"results": [], "unresponsive_engines": [[eng, "CAPTCHA"]]})

        with patch.object(tools, "requests") as req:
            req.get.side_effect = fake_get
            result = tools.search_internet("q")
        assert "https://m.com" in result

    def test_rotation_all_engines_down_no_tavily_returns_unavailable(self, monkeypatch):
        """Every engine throttled and no Tavily key → the relayable unavailable message."""
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        def fake_get(url, params=None, timeout=None, **kw):
            eng = (params or {}).get("engines")
            return _resp({"results": [], "unresponsive_engines": [[eng, "CAPTCHA"]]})

        with patch.object(tools, "requests") as req:
            req.get.side_effect = fake_get
            assert tools.search_internet("q") == tools._SEARCH_UNAVAILABLE_MESSAGE

    def test_rotation_exhausted_falls_back_to_tavily(self, monkeypatch):
        """When every SearXNG engine is down AND a Tavily key is set, search_internet falls
        back to the Tavily API and returns its results."""
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")

        def fake_get(url, params=None, timeout=None, **kw):  # every SearXNG engine down
            return _resp({"results": [],
                          "unresponsive_engines": [[(params or {}).get("engines"), "CAPTCHA"]]})

        def fake_post(url, json=None, timeout=None, **kw):  # Tavily backup
            assert "tavily" in url and (json or {}).get("api_key") == "test-key"
            return _resp({"results": [{"title": "Tav", "url": "https://t.com", "content": "c"}]})

        with patch.object(tools, "requests") as req:
            req.get.side_effect = fake_get
            req.post.side_effect = fake_post
            result = tools.search_internet("q")
        assert result != tools._SEARCH_UNAVAILABLE_MESSAGE
        assert "https://t.com" in result

    def test_tavily_backup_only_after_rotation_exhausted(self, monkeypatch):
        """Tavily is NOT called when a SearXNG engine returns results (it's the last resort)."""
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        with patch.object(tools, "requests") as req:
            req.get.return_value = _resp({"results": [{"title": "T", "url": "https://a.com",
                                                       "content": "c"}], "unresponsive_engines": []})
            result = tools.search_internet("q")
        assert "https://a.com" in result
        req.post.assert_not_called()

    def test_rotation_logs_each_engine_outcome(self, monkeypatch, caplog):
        """Each engine's outcome is logged (which engine + good/bad) for observability."""
        import logging
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        def fake_get(url, params=None, timeout=None, **kw):
            eng = (params or {}).get("engines")
            if eng == "mojeek":
                return _resp({"results": [{"title": "M", "url": "https://m.com", "content": "c"}],
                              "unresponsive_engines": []})
            return _resp({"results": [], "unresponsive_engines": [[eng, "CAPTCHA"]]})

        with patch.object(tools, "requests") as req, \
             caplog.at_level(logging.INFO, logger="assist.tools"):
            req.get.side_effect = fake_get
            tools.search_internet("q")
        msgs = " ".join(r.message for r in caplog.records)
        assert "searxng: google rate-limited/unavailable" in msgs
        assert "searxng: mojeek returned 1 results" in msgs

    def test_rotation_all_engines_healthy_but_empty_returns_empty(self, monkeypatch):
        """Every engine up but genuinely no results → the empty-guidance steer (a real
        no-results answer), NOT unavailable."""
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        with patch.object(tools, "requests") as req:
            req.get.return_value = _resp({"results": [], "unresponsive_engines": []})
            assert tools.search_internet("q") == tools._SEARCH_EMPTY_GUIDANCE

    def test_results_present_with_some_failed_engines_returns_them_plainly(self, monkeypatch):
        """Results present + some engines throttled is the routine healthy metasearch case
        (one engine timing out while others return a full set) — return the results plainly,
        no per-call 'partial' caveat (that would falsely flag nearly every report)."""
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        with patch.object(tools, "requests") as req:
            req.get.return_value = _resp({
                "results": [{"title": "T1", "url": "https://a.com", "content": "c1"}],
                "unresponsive_engines": [["brave", "too many requests"]],
            })
            result = tools.search_internet("q")
        assert result != tools._SEARCH_UNAVAILABLE_MESSAGE
        assert "https://a.com" in result and "c1" in result
        assert "PARTIAL" not in result  # no per-call throttle caveat when results came back

    def test_non_dict_payload_returns_unavailable(self, monkeypatch):
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        with patch.object(tools, "requests") as req:
            req.get.return_value = _resp(["not", "a", "dict"])
            assert tools.search_internet("q") == tools._SEARCH_UNAVAILABLE_MESSAGE

    def test_non_list_results_returns_unavailable(self, monkeypatch):
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        with patch.object(tools, "requests") as req:
            req.get.return_value = _resp({"results": {"unexpected": "object"}})
            assert tools.search_internet("q") == tools._SEARCH_UNAVAILABLE_MESSAGE

    def test_missing_results_field_returns_unavailable(self, monkeypatch):
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        with patch.object(tools, "requests") as req:
            req.get.return_value = _resp({"query": "q"})  # no 'results' key
            assert tools.search_internet("q") == tools._SEARCH_UNAVAILABLE_MESSAGE

    def test_falsy_non_list_results_returns_unavailable(self, monkeypatch):
        """A FALSY non-list `results` ({} or "") must not be coerced to [] and
        read as a genuine 'no results'."""
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        for bad in ({}, ""):
            with patch.object(tools, "requests") as req:
                req.get.return_value = _resp({"results": bad})
                assert tools.search_internet("q") == tools._SEARCH_UNAVAILABLE_MESSAGE

    def test_malformed_unresponsive_engines_returns_unavailable(self, monkeypatch):
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        for bad in ({}, "", {"google": "timeout"}):
            with patch.object(tools, "requests") as req:
                req.get.return_value = _resp({"results": [], "unresponsive_engines": bad})
                assert tools.search_internet("q") == tools._SEARCH_UNAVAILABLE_MESSAGE

    def test_transport_error_returns_unavailable(self, monkeypatch):
        """SearXNG unreachable → unavailable message (relayed), not an
        exception that crashes the turn."""
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        with patch.object(tools, "requests") as req:
            req.get.side_effect = Exception("connection refused")
            assert tools.search_internet("q") == tools._SEARCH_UNAVAILABLE_MESSAGE

    def test_http_error_returns_unavailable(self, monkeypatch):
        """A non-2xx from SearXNG (raise_for_status) is a loud failure relayed
        as the unavailable message."""
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        bad = MagicMock()
        bad.raise_for_status.side_effect = Exception("503 Service Unavailable")
        with patch.object(tools, "requests") as req:
            req.get.return_value = bad
            assert tools.search_internet("q") == tools._SEARCH_UNAVAILABLE_MESSAGE



# -------------------- read_url main-content extraction -------------------

class TestReadUrlExtraction:
    """read_url returns the marked <article>/<main> body (WITH its own
    heading/byline/footer) and drops outside chrome; a page with no article
    degrades to whole-page text minus scripts (no regression). Parser-based, so
    nested tags / attributes-containing-'>' / comments / entities are handled."""

    ARTICLE_PAGE = (
        "<html><head><title>T</title>"
        "<style>.x{color:red}</style><script>track('noise')</script></head>"
        "<body>"
        "<nav>Home About Contact</nav>"
        "<article><header><h1>Real Title</h1><span>By Sam</span></header>"
        "<p>The actual article body has the facts.</p>"
        "<ul><li>point one</li><li>point two</li></ul>"
        "<footer>Citations example.org</footer></article>"
        "<aside>Related links ad slot</aside>"
        "<footer>Site copyright</footer></body></html>"
    )

    def _read(self, html):
        with patch.object(tools, "requests") as rq, \
                patch.object(tools, "_host_throttle"):
            rq.get.return_value = _html_resp(html)
            return tools.read_url("https://example.com/x")

    def test_returns_article_body(self):
        out = self._read(self.ARTICLE_PAGE)
        assert "The actual article body has the facts." in out
        assert "point one" in out and "point two" in out

    def test_keeps_article_own_header_and_footer(self):
        # The BLOCKER the review caught: title/byline/citations live in the
        # article's OWN <header>/<footer> and must survive extraction.
        out = self._read(self.ARTICLE_PAGE)
        assert "Real Title" in out and "By Sam" in out
        assert "Citations example.org" in out

    def test_drops_outside_chrome(self):
        out = self._read(self.ARTICLE_PAGE)
        for chrome in ("Home About Contact", "Related links", "Site copyright",
                       "track(", "color:red"):
            assert chrome not in out

    def test_concatenates_multiple_articles(self):
        # Multi-article pages: keep all regions, not just the largest.
        html = ("<body><article>" + ("filler " * 50) + "</article>"
                "<article>the real answer is 42</article></body>")
        assert "the real answer is 42" in self._read(html)

    def test_no_article_degrades_to_whole_page_minus_scripts(self):
        # No <article>: same as the old whole-page strip (no regression),
        # chrome text included — but scripts/styles still removed.
        html = ("<body><nav>menu items</nav>"
                "<script>var leak='secret'</script>"
                "<div><p>Body paragraph content.</p></div></body>")
        out = self._read(html)
        assert "Body paragraph content." in out
        assert "menu items" in out                 # no article -> nothing dropped
        assert "leak" not in out and "var leak" not in out  # script gone

    def test_attribute_with_gt_does_not_corrupt(self):
        # The regex tag-stripper leaked `b">` here; the parser handles it.
        html = '<article><p data-x="a>b">hello world facts</p></article>'
        out = self._read(html)
        assert "hello world facts" in out
        assert 'b">' not in out and "a>b" not in out

    def test_html_comment_dropped(self):
        html = "<article><!-- tracker > pixel --><p>visible text</p></article>"
        out = self._read(html)
        assert "visible text" in out
        assert "tracker" not in out and "pixel" not in out

    def test_entities_decoded(self):
        assert self._read("<article><p>a &amp; b &lt; c</p></article>") == "a & b < c"

    def test_noscript_fallback_kept(self):
        # read_url runs no JS, so <noscript> is the page's no-JS fallback —
        # the relevant content for us. A real fetch of a JS-gated forum showed
        # dropping it returned empty; keep it.
        out = self._read("<body><noscript>real fallback content</noscript></body>")
        assert "real fallback content" in out

    def test_script_inside_article_dropped(self):
        html = "<article><script>var s='leak'</script><p>kept</p></article>"
        out = self._read(html)
        assert "kept" in out and "leak" not in out

    def test_adjacent_tags_keep_word_boundary(self):
        # Round-2 BLOCKER: text nodes must not fuse across tags. The old strip
        # put a space at every tag; the parser must too.
        assert "one two" in self._read("<p>one</p><p>two</p>")
        assert "a b c" in self._read("<div>a<b>b</b>c</div>")

    def test_flushes_dangling_trailing_token(self):
        # feed() buffers an incomplete trailing token; close() must flush it,
        # else a doc ending mid-token (here a bare entity) returns empty.
        assert "tail text" in tools._extract_main_content("<article>tail text &amp")

    def test_returns_full_content_uncapped(self):
        # read_url no longer truncates — it returns the full extracted text; the
        # large-result offload to a file is ToolResultToFileMiddleware's job now.
        assert len(self._read("<article><p>" + ("x" * 30000) + "</p></article>")) == 30000
        assert len(self._read("<article><p>" + ("x" * 9000) + "</p></article>")) == 9000

    def test_error_path_unchanged(self):
        with patch.object(tools, "requests") as rq, \
                patch.object(tools, "_host_throttle"):
            rq.get.side_effect = RuntimeError("boom")
            out = tools.read_url("https://example.com/err")
        assert out.startswith("Error fetching URL:")

    def test_extract_main_content_helper(self):
        assert tools._extract_main_content(
            "<article><p>just this</p></article><footer>not this</footer>"
        ) == "just this"

    def test_extract_surfaces_links_absolute_and_deduped(self):
        html = ('<main><p>Manuals</p>'
                '<a href="/pages/manual">Manual</a>'
                '<a href="/pages/manual">dup</a>'          # deduped by href
                '<a href="#top">skip anchor</a>'           # in-page anchor dropped
                '<a href="mailto:x@y.z">skip mail</a>'
                '<script src="/cdn/assets/x.js"></script>' # script href never a link
                '</main>')
        out = tools._extract_main_content(html, base_url="https://s.example/pages/support")
        assert "Links on this page:" in out
        assert "https://s.example/pages/manual  (Manual)" in out
        assert out.count("/pages/manual") == 1             # deduped
        assert "#top" not in out and "mailto" not in out and "/cdn/assets/" not in out

    def test_extract_no_links_section_when_none(self):
        assert "Links on this page" not in tools._extract_main_content(
            "<main><p>text only</p></main>", base_url="https://s.example/")
