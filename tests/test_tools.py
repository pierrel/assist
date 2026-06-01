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
            t.time.side_effect = [1000.0, 1000.0, 1000.3, 1000.3]
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
