"""Web tools the agent exposes: a throttled DuckDuckGo search and a
per-host throttled URL fetch.

Throttle/rate-limit shape (added 2026-05-31 after a research-agent ran
~9,000 fetch_url calls in two hours and got the dev box's IP
soft-blocked by DDG):

- ``search_internet`` is throttled to ~one call every ``_SEARCH_MIN_DELAY``
  seconds and circuit-broken after ``_SEARCH_CIRCUIT_FAILURE_THRESHOLD``
  consecutive failures.  DDG has no public rate-limit documentation for
  anonymous HTML/lite endpoints; community-observed safe rates are
  ~1 query / 2-5s, so 4s is the conservative middle.  When the circuit
  is open we return an explicit message ``_CIRCUIT_OPEN_MESSAGE`` rather
  than the silent ``"[]"`` (no results) — the latter would let the model
  retry or write a confidently-empty report; the former tells the model
  *why* it can't search and what to do.
- ``read_url`` is throttled per-host (1s between calls to the same host)
  rather than globally, so a burst of fetches to different sites isn't
  artificially serialised but a tight loop against one bot-protected
  site (the casio-runaway shape) is locally rate-limited before that
  site's edge does the same to us.
"""

import re
import time
import threading
from urllib.parse import urlparse

import requests
from ddgs import DDGS

# --- Search throttle (cross-tool, global) ---
_search_lock = threading.Lock()
_search_last_call_time = 0.0
_SEARCH_MIN_DELAY = 4.0

# --- Search circuit breaker ---
# After N consecutive search failures (DDG timeout / exception), open the
# circuit for `_SEARCH_CIRCUIT_DURATION_S` and return the explicit message
# below instead of attempting the call — saves further hits to a DDG
# endpoint that's already blocking us, and gives the model an
# unambiguous "search is dead, finalize what you have" instruction.
_search_circuit_lock = threading.Lock()
_search_consecutive_failures = 0
_SEARCH_CIRCUIT_FAILURE_THRESHOLD = 3
_search_circuit_open_until = 0.0
_SEARCH_CIRCUIT_DURATION_S = 600  # 10 minutes

_CIRCUIT_OPEN_MESSAGE = (
    "Search is rate-limited (DuckDuckGo). Cannot retry for ~10 minutes. "
    "Finalize your response using what's already gathered; tell the user "
    "search is temporarily rate-limited and to try again in a few minutes."
)

# Substrings in an exception's type-name + str() that suggest the failure
# is an upstream rate-limit / block (as opposed to a transient network
# blip or a parse failure on a one-off bad result).  When detected, we
# open the circuit IMMEDIATELY rather than waiting for
# _SEARCH_CIRCUIT_FAILURE_THRESHOLD failures — the cost of a false
# positive (10 min of no search; the agent finalizes with what it has)
# is much lower than the cost of a false negative (more requests to a
# blocked endpoint, deeper IP burn, longer recovery).  Biased toward
# false positives accordingly: timeouts, connection-resets, and
# explicit 4xx/429 all count.
_RATE_LIMIT_EXC_INDICATORS = (
    "timeout", "timed out", "read timeout",
    "connection refused", "connection reset", "reset by peer",
    "rate limit", "rate-limit", "too many requests",
    "blocked", "challenge", "captcha", "forbidden",
    " 429", " 403",
)

# --- Per-host fetch throttle ---
_host_lock = threading.Lock()
_host_last_call: dict[str, float] = {}
_HOST_MIN_DELAY = 1.0


def _search_throttle() -> None:
    """Block until at least ``_SEARCH_MIN_DELAY`` has passed since the
    last search call."""
    global _search_last_call_time
    with _search_lock:
        now = time.time()
        elapsed = now - _search_last_call_time
        if elapsed < _SEARCH_MIN_DELAY:
            time.sleep(_SEARCH_MIN_DELAY - elapsed)
        _search_last_call_time = time.time()


def _host_throttle(host: str) -> None:
    """Block until at least ``_HOST_MIN_DELAY`` has passed since the last
    fetch to this specific ``host``.  No-op for empty/None host."""
    if not host:
        return
    with _host_lock:
        now = time.time()
        last = _host_last_call.get(host, 0.0)
        elapsed = now - last
        if elapsed < _HOST_MIN_DELAY:
            time.sleep(_HOST_MIN_DELAY - elapsed)
        _host_last_call[host] = time.time()


def _circuit_is_open() -> bool:
    """True if the search circuit is currently open (wall-clock-bounded)."""
    return time.time() < _search_circuit_open_until


def _record_search_failure() -> None:
    """Bump the consecutive-failure counter; open the circuit if at threshold."""
    global _search_consecutive_failures, _search_circuit_open_until
    with _search_circuit_lock:
        _search_consecutive_failures += 1
        if _search_consecutive_failures >= _SEARCH_CIRCUIT_FAILURE_THRESHOLD:
            _search_circuit_open_until = time.time() + _SEARCH_CIRCUIT_DURATION_S


def _record_search_success() -> None:
    """Reset the consecutive-failure counter; a successful call closes
    the circuit's path back to working."""
    global _search_consecutive_failures
    with _search_circuit_lock:
        _search_consecutive_failures = 0


def _open_search_circuit_now() -> None:
    """Open the search circuit immediately (rate-limit DETECTED, not
    just a generic failure).  Distinct from `_record_search_failure`,
    which counts toward the threshold — this jumps straight to open."""
    global _search_consecutive_failures, _search_circuit_open_until
    with _search_circuit_lock:
        _search_consecutive_failures = _SEARCH_CIRCUIT_FAILURE_THRESHOLD
        _search_circuit_open_until = time.time() + _SEARCH_CIRCUIT_DURATION_S


def _exception_looks_like_rate_limit(exc: BaseException) -> bool:
    """Heuristic: does ``exc`` (its type-name + str()) match any
    `_RATE_LIMIT_EXC_INDICATORS` substring?  Used by `search_internet`
    to short-circuit on detected upstream blocks instead of waiting for
    the consecutive-failures threshold.  Case-insensitive."""
    blob = f"{type(exc).__name__}: {exc}".lower()
    return any(ind in blob for ind in _RATE_LIMIT_EXC_INDICATORS)


def read_url(url: str) -> str:
    """Extract the content from the given url.

    Per-host throttled (~1s between calls to the same host) so a burst
    of fetches to different sites isn't artificially serialised, but a
    tight loop against one bot-protected site is rate-limited locally."""
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        host = ""
    _host_throttle(host)
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"},
            timeout=15,
        )
        resp.raise_for_status()
        text = resp.text
        text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:4000]
    except Exception as e:
        return f"Error fetching URL: {e}"


def search_internet(
        query: str,
        max_results: int = 5,
) -> str:
    """Used to search the internet for information on a given topic using a query string.

    Throttled to ~4s between DDG calls and circuit-broken on 3 consecutive
    failures.  When the circuit is open, returns an explicit-message string
    so the model finalizes its response instead of silently retrying or
    writing a confidently-empty report."""
    if _circuit_is_open():
        return _CIRCUIT_OPEN_MESSAGE
    _search_throttle()
    try:
        results = DDGS().text(query,
                              max_results=max_results,
                              backend="duckduckgo")
        _record_search_success()
    except Exception as e:
        # Rate-limit / block DETECTED (timeout, connection reset, 429,
        # 403, captcha challenge, etc.)?  Skip the slow consecutive-failures
        # threshold and open the circuit NOW so we stop hitting an
        # already-blocking endpoint.  The model gets the same explicit
        # "search is rate-limited" instruction it would after threshold.
        if _exception_looks_like_rate_limit(e):
            _open_search_circuit_now()
            return _CIRCUIT_OPEN_MESSAGE
        _record_search_failure()
        # If THIS failure tipped us into circuit-open state, surface the
        # explicit message immediately rather than the bare "[]" — the
        # model gets the same instruction whether the circuit opened
        # before or during this call.
        if _circuit_is_open():
            return _CIRCUIT_OPEN_MESSAGE
        return "[]"
    normalized = [{"title": r["title"], "url": r["href"], "content": r["body"]} for r in results]
    return str(normalized)
