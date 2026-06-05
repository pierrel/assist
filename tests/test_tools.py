"""Tests for the web tools in assist.tools.

``search_internet`` goes through a self-hosted SearXNG instance with NO
fallback — if SearXNG is unset/unreachable/erroring, or returns zero results
while reporting any engine failures, it raises (failures must be loud).  The HTTP
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

    def test_raises_when_search_url_unset(self):
        """No ASSIST_SEARCH_URL is a misconfiguration, not a silent no-op —
        search has no fallback, so it must fail loudly."""
        with pytest.raises(RuntimeError, match="ASSIST_SEARCH_URL"):
            tools.search_internet("anything")

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

    def test_genuine_empty_returns_bracket(self, monkeypatch):
        """Zero results with NO engine failures is a real 'no results' answer,
        not a backend failure — return '[]' so the agent can pivot."""
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        with patch.object(tools, "requests") as req:
            req.get.return_value = _resp({"results": [], "unresponsive_engines": []})
            assert tools.search_internet("obscure") == "[]"

    def test_empty_with_failed_engines_raises(self, monkeypatch):
        """Zero results while any engine reported a failure is a loud backend
        failure, not a 'no results' answer."""
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        with patch.object(tools, "requests") as req:
            req.get.return_value = _resp({
                "results": [],
                "unresponsive_engines": [["google", "timeout"], ["brave", "CAPTCHA"]],
            })
            with pytest.raises(RuntimeError, match="unhealthy|engines failed|engines"):
                tools.search_internet("q")

    def test_non_dict_payload_raises(self, monkeypatch):
        """Valid JSON of an unexpected top-level shape (list/string) is still
        a broken backend → clear loud failure, not a bare AttributeError."""
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        with patch.object(tools, "requests") as req:
            req.get.return_value = _resp(["not", "a", "dict"])
            with pytest.raises(RuntimeError, match="unexpected response shape"):
                tools.search_internet("q")

    def test_non_list_results_raises(self, monkeypatch):
        """A dict payload whose `results` is not a list is also a broken
        backend → loud failure rather than a TypeError on the slice."""
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        with patch.object(tools, "requests") as req:
            req.get.return_value = _resp({"results": {"unexpected": "object"}})
            with pytest.raises(RuntimeError, match="unexpected type|results"):
                tools.search_internet("q")

    def test_missing_results_field_raises(self, monkeypatch):
        """A valid SearXNG response always carries a `results` list; a dict
        with no `results` field is malformed → loud failure, not "[]"."""
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        with patch.object(tools, "requests") as req:
            req.get.return_value = _resp({"query": "q"})  # no 'results' key
            with pytest.raises(RuntimeError, match="no 'results'|results"):
                tools.search_internet("q")

    def test_falsy_non_list_results_raises(self, monkeypatch):
        """A FALSY non-list `results` (e.g. {} or "") must also raise — it must
        not be silently coerced to [] and treated as a genuine 'no results'."""
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        for bad in ({}, ""):
            with patch.object(tools, "requests") as req:
                req.get.return_value = _resp({"results": bad})
                with pytest.raises(RuntimeError, match="unexpected type|results"):
                    tools.search_internet("q")

    def test_malformed_unresponsive_engines_raises(self, monkeypatch):
        """Empty results with a malformed (non-list) unresponsive_engines —
        incl. falsy {}/"" — is a broken backend, not 'no results'."""
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        for bad in ({}, "", {"google": "timeout"}):
            with patch.object(tools, "requests") as req:
                req.get.return_value = _resp({"results": [], "unresponsive_engines": bad})
                with pytest.raises(RuntimeError, match="unexpected type|unresponsive"):
                    tools.search_internet("q")

    def test_transport_error_raises(self, monkeypatch):
        """SearXNG unreachable → loud RuntimeError, no silent fallback."""
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        with patch.object(tools, "requests") as req:
            req.get.side_effect = Exception("connection refused")
            with pytest.raises(RuntimeError, match="unavailable"):
                tools.search_internet("q")

    def test_http_error_raises(self, monkeypatch):
        """A non-2xx from SearXNG (raise_for_status) is a loud failure."""
        monkeypatch.setenv("ASSIST_SEARCH_URL", self.URL)
        bad = MagicMock()
        bad.raise_for_status.side_effect = Exception("503 Service Unavailable")
        with patch.object(tools, "requests") as req:
            req.get.return_value = bad
            with pytest.raises(RuntimeError, match="unavailable"):
                tools.search_internet("q")
