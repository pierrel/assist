"""Tests for the throttled / circuit-broken web tools in assist.tools.

The throttle helpers and the search circuit breaker are tested by
patching the module's ``time`` and ``DDGS`` references, so no real
sleeps and no real network are involved.  Each test resets the module's
global state in a fixture so order-of-execution doesn't matter."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from assist import tools


@pytest.fixture(autouse=True)
def _reset_tool_state():
    """Reset module-level throttle + circuit state before each test."""
    tools._search_last_call_time = 0.0
    tools._search_consecutive_failures = 0
    tools._search_circuit_open_until = 0.0
    tools._host_last_call.clear()
    yield


# -------------------- _search_throttle ----------------------------------

class TestSearchThrottle:
    def test_first_call_does_not_sleep(self):
        """No prior call → elapsed is huge → no sleep."""
        with patch.object(tools, "time") as t:
            t.time.return_value = 1000.0
            tools._search_throttle()
            t.sleep.assert_not_called()

    def test_second_call_within_window_sleeps_difference(self):
        """Second call 1s after first should sleep (MIN_DELAY - 1s)."""
        with patch.object(tools, "time") as t:
            t.time.side_effect = [1000.0, 1000.0, 1001.0, 1001.0]
            tools._search_throttle()  # first
            tools._search_throttle()  # second, 1s later
            t.sleep.assert_called_once_with(tools._SEARCH_MIN_DELAY - 1.0)

    def test_second_call_past_window_does_not_sleep(self):
        """Second call after MIN_DELAY elapsed → no sleep."""
        with patch.object(tools, "time") as t:
            t.time.side_effect = [1000.0, 1000.0,
                                  1000.0 + tools._SEARCH_MIN_DELAY + 1, 1000.0 + tools._SEARCH_MIN_DELAY + 1]
            tools._search_throttle()
            tools._search_throttle()
            t.sleep.assert_not_called()


# -------------------- _host_throttle ------------------------------------

class TestHostThrottle:
    def test_first_call_to_a_host_does_not_sleep(self):
        with patch.object(tools, "time") as t:
            t.time.return_value = 1000.0
            tools._host_throttle("example.com")
            t.sleep.assert_not_called()

    def test_second_call_to_same_host_within_window_sleeps(self):
        with patch.object(tools, "time") as t:
            # Three time.time() calls expected: call 1 (not-slept) = 1;
            # call 2 (slept) = 2 (one before-sleep check, one after-sleep
            # to record the actual call time).
            t.time.side_effect = [1000.0, 1000.3, 1000.3]
            tools._host_throttle("example.com")
            tools._host_throttle("example.com")
            t.sleep.assert_called_once()
            # Approx because 1.0 - 0.3 isn't exactly 0.7 in float.
            slept = t.sleep.call_args.args[0]
            assert slept == pytest.approx(tools._HOST_MIN_DELAY - 0.3)

    def test_different_hosts_do_not_block_each_other(self):
        """The whole point of per-host: a burst of fetches to distinct
        sites isn't artificially serialised."""
        with patch.object(tools, "time") as t:
            t.time.side_effect = [1000.0, 1000.0, 1000.1, 1000.1]
            tools._host_throttle("example.com")
            tools._host_throttle("other.com")
            t.sleep.assert_not_called()

    def test_empty_host_is_noop(self):
        """A URL whose hostname parses as empty (rare) shouldn't crash
        — just skip the throttle."""
        with patch.object(tools, "time") as t:
            tools._host_throttle("")
            t.sleep.assert_not_called()
            t.time.assert_not_called()

    def test_host_dict_pruned_when_over_threshold(self):
        """Over-threshold dict size triggers a prune that drops entries
        older than _HOST_DICT_PRUNE_KEEP_S, bounding memory in a
        long-running process that touches many distinct hosts
        (Copilot PR #118 review item #1)."""
        # Seed the dict: half "fresh" (within keep-window), half "stale".
        threshold = tools._HOST_DICT_PRUNE_THRESHOLD
        keep_s = tools._HOST_DICT_PRUNE_KEEP_S
        now_anchor = 10_000.0
        # Stale: their last-call is `keep_s + 10` seconds ago.
        for i in range(threshold // 2 + 5):
            tools._host_last_call[f"stale-{i}.example"] = now_anchor - keep_s - 10
        # Fresh: their last-call is 1 second ago.
        for i in range(threshold // 2 + 5):
            tools._host_last_call[f"fresh-{i}.example"] = now_anchor - 1
        # Sanity: we should be over threshold so the prune triggers.
        assert len(tools._host_last_call) > threshold
        # Drive `_host_throttle` for a new host at `now_anchor`.
        with patch.object(tools.time, "time", return_value=now_anchor), \
             patch.object(tools.time, "sleep"):
            tools._host_throttle("new-host.example")
        # All stale entries should be gone; all fresh + the new one stay.
        assert not any(h.startswith("stale-") for h in tools._host_last_call)
        assert all(h.startswith("fresh-") or h == "new-host.example"
                   for h in tools._host_last_call)
        # And size is now bounded by the fresh-plus-new count.
        assert len(tools._host_last_call) == threshold // 2 + 5 + 1


# -------------------- circuit breaker primitives ------------------------

class TestCircuitBreakerPrimitives:
    def test_failures_below_threshold_keep_circuit_closed(self):
        for _ in range(tools._SEARCH_CIRCUIT_FAILURE_THRESHOLD - 1):
            tools._record_search_failure()
        assert not tools._circuit_is_open()

    def test_threshold_failures_open_circuit(self):
        with patch.object(tools, "time") as t:
            t.time.return_value = 5000.0
            for _ in range(tools._SEARCH_CIRCUIT_FAILURE_THRESHOLD):
                tools._record_search_failure()
            assert tools._circuit_is_open()
            assert tools._search_circuit_open_until == 5000.0 + tools._SEARCH_CIRCUIT_DURATION_S

    def test_success_resets_failure_counter(self):
        tools._record_search_failure()
        tools._record_search_failure()
        tools._record_search_success()
        # Failure count is back to 0, so next THRESHOLD failures are needed
        # to open the circuit.
        for _ in range(tools._SEARCH_CIRCUIT_FAILURE_THRESHOLD - 1):
            tools._record_search_failure()
        assert not tools._circuit_is_open()

    def test_circuit_recloses_after_duration(self):
        with patch.object(tools, "time") as t:
            t.time.return_value = 5000.0
            for _ in range(tools._SEARCH_CIRCUIT_FAILURE_THRESHOLD):
                tools._record_search_failure()
            assert tools._circuit_is_open()
            # Jump past the duration.
            t.time.return_value = 5000.0 + tools._SEARCH_CIRCUIT_DURATION_S + 1
            assert not tools._circuit_is_open()


# -------------------- search_internet end-to-end ------------------------

class TestSearchInternet:
    def test_open_circuit_returns_explicit_message_without_calling_DDG(self):
        """The model needs an explicit 'search is rate-limited' message
        — NOT the silent '[]', which the model would interpret as 'no
        results' and potentially write a confidently-empty report or
        retry immediately."""
        with patch.object(tools, "time") as t, \
             patch.object(tools, "DDGS") as ddgs:
            t.time.return_value = 5000.0
            tools._search_circuit_open_until = 5000.0 + 60  # circuit open
            result = tools.search_internet("anything")
            assert result == tools._CIRCUIT_OPEN_MESSAGE
            ddgs.assert_not_called()

    def test_DDG_exception_does_not_open_circuit_below_threshold(self):
        """One or two exceptions should NOT open the circuit — sporadic
        DDG hiccups are normal.  Below threshold returns the bare '[]'
        so the model treats it as 'no results' and pivots, the same
        behavior pre-patch."""
        with patch.object(tools, "time") as t, \
             patch.object(tools, "DDGS") as ddgs:
            t.time.return_value = 5000.0
            ddgs.return_value.text.side_effect = Exception("boom")
            result = tools.search_internet("query")
            assert result == "[]"
            assert not tools._circuit_is_open()

    def test_threshold_failures_open_circuit_via_real_path(self):
        """Driving search_internet through THRESHOLD consecutive
        exceptions opens the circuit and the next call short-circuits."""
        with patch.object(tools, "time") as t, \
             patch.object(tools, "DDGS") as ddgs:
            t.time.return_value = 5000.0
            ddgs.return_value.text.side_effect = Exception("boom")
            for _ in range(tools._SEARCH_CIRCUIT_FAILURE_THRESHOLD):
                tools.search_internet("query")
            # Circuit should now be open; the next call doesn't reach DDGS.
            ddgs.reset_mock()
            result = tools.search_internet("another query")
            assert result == tools._CIRCUIT_OPEN_MESSAGE
            ddgs.assert_not_called()

    def test_success_after_failures_keeps_circuit_closed(self):
        """A successful call resets the failure counter, so the next
        N-1 failures don't open the circuit on their own."""
        with patch.object(tools, "time") as t, \
             patch.object(tools, "DDGS") as ddgs:
            t.time.return_value = 5000.0
            # First call fails, second succeeds, third fails.
            fake_results = [{"title": "x", "href": "https://e.com", "body": "y"}]
            ddgs.return_value.text.side_effect = [
                Exception("boom"),
                fake_results,
                Exception("boom"),
            ]
            tools.search_internet("q1")
            tools.search_internet("q2")
            tools.search_internet("q3")
            # 1 failure → success (reset) → 1 failure = below threshold.
            assert not tools._circuit_is_open()


# -------------------- rate-limit detection on exception -----------------

class TestRateLimitDetection:
    @pytest.mark.parametrize("exc", [
        TimeoutError("operation timed out"),
        ConnectionError("Connection refused"),
        ConnectionResetError("Connection reset by peer"),
        Exception("HTTP 429 Too Many Requests"),
        Exception("HTTP 403 Forbidden"),
        Exception("DuckDuckGo blocked the request: please solve the CAPTCHA"),
        Exception("Rate limit exceeded"),
    ])
    def test_detector_matches_known_rate_limit_shapes(self, exc):
        assert tools._exception_looks_like_rate_limit(exc), \
            f"detector should have caught {type(exc).__name__}: {exc}"

    @pytest.mark.parametrize("exc", [
        ValueError("invalid query"),
        KeyError("missing field 'href'"),
        Exception("Unable to parse response as JSON"),
        RuntimeError("internal library error"),
    ])
    def test_detector_does_not_match_generic_failures(self, exc):
        assert not tools._exception_looks_like_rate_limit(exc), \
            f"detector falsely flagged {type(exc).__name__}: {exc}"

    def test_rate_limit_exception_opens_circuit_on_first_failure(self):
        """The whole point: a single timeout/429/etc should NOT wait for
        the 3-failure threshold — it should open immediately and return
        the explicit message, so the next search call short-circuits."""
        with patch.object(tools, "time") as t, \
             patch.object(tools, "DDGS") as ddgs:
            t.time.return_value = 5000.0
            ddgs.return_value.text.side_effect = TimeoutError("Read timeout")
            result = tools.search_internet("query")
            assert result == tools._CIRCUIT_OPEN_MESSAGE
            assert tools._circuit_is_open()
            # And the NEXT call doesn't even invoke DDGS — short-circuited.
            ddgs.reset_mock()
            result2 = tools.search_internet("another query")
            assert result2 == tools._CIRCUIT_OPEN_MESSAGE
            ddgs.assert_not_called()

    def test_generic_exception_still_uses_threshold_path(self):
        """Counterpart: a non-rate-limit exception must NOT trip the
        circuit early — sporadic parse errors etc. shouldn't take search
        offline for 10 minutes.  Still returns the bare '[]' so the model
        treats it as 'no results' and pivots."""
        with patch.object(tools, "time") as t, \
             patch.object(tools, "DDGS") as ddgs:
            t.time.return_value = 5000.0
            ddgs.return_value.text.side_effect = ValueError("bad query")
            result = tools.search_internet("query")
            assert result == "[]"
            assert not tools._circuit_is_open()

    def test_ddgs_ratelimit_exception_type_detected_explicitly(self):
        """If `ddgs` itself raises `RatelimitException`, the explicit
        type check should match even though the message text might not
        contain our substring indicators."""
        if not tools._DDGS_RATE_LIMIT_TYPES:
            pytest.skip("ddgs.exceptions module not importable here")
        RL = tools._DDGS_RATE_LIMIT_TYPES[0]  # RatelimitException
        # Message intentionally contains NO substring indicator — the type
        # check is what must catch this.
        exc = RL("blip")
        assert tools._exception_looks_like_rate_limit(exc, elapsed_s=0.0)

    def test_slow_ddgs_no_results_treated_as_rate_limit(self):
        """ddgs's `DDGSException("No results found.")` is its catch-all
        when nothing parses — raised both for *genuine* empty results
        AND for TCP timeouts (the message is misleading).  A call that
        took >=3s is almost certainly a timeout disguised as empty."""
        if not tools._DDGS_RATE_LIMIT_TYPES:
            pytest.skip("ddgs.exceptions module not importable here")
        # Use the base DDGSException so the type check DOESN'T trip; we
        # need the timing heuristic to be what catches it.
        import ddgs.exceptions
        exc = ddgs.exceptions.DDGSException("No results found.")
        assert tools._exception_looks_like_rate_limit(exc, elapsed_s=4.5)
        # And the FAST counterpart is treated as a genuine empty result.
        assert not tools._exception_looks_like_rate_limit(exc, elapsed_s=0.2)

    def test_search_internet_opens_circuit_on_slow_no_results(self):
        """End-to-end through search_internet: a `DDGSException` raised
        after ~5s (TCP-timeout shape) should open the circuit
        immediately on the FIRST failure, returning the explicit
        message rather than the bare '[]'.

        Drives ``time.time`` via a stateful counter (rather than a fixed
        side_effect list) because the precise number of internal
        ``time.time()`` calls per ``search_internet`` invocation is an
        implementation detail (circuit check, throttle x2, t0, elapsed,
        circuit-open, plus any future probes); a list would have to grow
        every refactor."""
        if not tools._DDGS_RATE_LIMIT_TYPES:
            pytest.skip("ddgs.exceptions module not importable here")
        import ddgs.exceptions
        # Stateful clock: the first 4 calls (circuit_check + throttle x2 +
        # t0) return 5000, then the elapsed call and onward return 5005
        # so `elapsed == 5.0` (> 3s threshold = treated as TCP timeout).
        calls = {"n": 0}
        def fake_time():
            calls["n"] += 1
            return 5000.0 if calls["n"] <= 4 else 5005.0
        with patch.object(tools.time, "time", side_effect=fake_time), \
             patch.object(tools.time, "sleep"), \
             patch.object(tools, "DDGS") as ddgs_class:
            ddgs_class.return_value.text.side_effect = (
                ddgs.exceptions.DDGSException("No results found.")
            )
            result = tools.search_internet("query")
            assert result == tools._CIRCUIT_OPEN_MESSAGE
            assert tools._circuit_is_open()

    def test_search_internet_fast_no_results_does_NOT_open_circuit(self):
        """End-to-end: a fast `DDGSException` (genuine empty results, not
        a timeout) must NOT trip the circuit — just return '[]'."""
        if not tools._DDGS_RATE_LIMIT_TYPES:
            pytest.skip("ddgs.exceptions module not importable here")
        import ddgs.exceptions
        # 100ms elapsed (genuine empty parse), well below the 3s threshold.
        calls = {"n": 0}
        def fake_time():
            calls["n"] += 1
            return 5000.0 if calls["n"] <= 4 else 5000.1
        with patch.object(tools.time, "time", side_effect=fake_time), \
             patch.object(tools.time, "sleep"), \
             patch.object(tools, "DDGS") as ddgs_class:
            ddgs_class.return_value.text.side_effect = (
                ddgs.exceptions.DDGSException("No results found.")
            )
            result = tools.search_internet("query")
            assert result == "[]"
            assert not tools._circuit_is_open()
