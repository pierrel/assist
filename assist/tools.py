"""Web tools the agent exposes: a self-hosted SearXNG metasearch and a
per-host throttled URL fetch.

Search goes through a self-hosted SearXNG instance (``ASSIST_SEARCH_URL``) —
private, on hardware we control, multi-engine, no API key.  There is NO
fallback: if SearXNG is unset, unreachable, errors, or returns zero results
while reporting any engine failures, ``search_internet`` RETURNS an explicit
``_SEARCH_UNAVAILABLE_MESSAGE`` (logged at ERROR) that the agent relays — it
does NOT raise into the agent loop (a raised exception would crash the research
turn).  A broken search backend still fails LOUDLY (logged + surfaced to the
user), it just doesn't silently degrade to a flaky scraper that hides the
outage behind worse results.

``read_url`` is throttled per-host (1s between calls to the same host) rather
than globally, so a burst of fetches to different sites isn't artificially
serialised, but a tight loop against one bot-protected site is rate-limited
locally before that site's edge does the same to us (the 2026-05-31
casio-runaway shape: ~9,000 fetches across many distinct URLs in two hours).
"""

import logging
import os
import re
import time
import threading
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

# Self-hosted SearXNG metasearch endpoint (see scripts/searxng.sh /
# `make searxng-up`).  search_internet REQUIRES this — there is no fallback.
_SEARXNG_TIMEOUT_S = 10.0

# Returned (not raised) when the search backend is broken/unreachable.  This is
# "fail loud" done right: the agent RECEIVES this as the tool result, the
# research prompts relay it ("couldn't look this up — search is unavailable"),
# and the user is told plainly — instead of an exception crashing the research
# turn.  The operator still gets a logger.error with the specific cause.  No
# wait-time framing: a broken backend is an outage to fix, not a rate-limit to
# wait out.
_SEARCH_UNAVAILABLE_MESSAGE = (
    "Web search is unavailable — the search backend is down. STOP: do not "
    "search again, and do not retry with a different query — it is down and "
    "every retry will fail the same way. Do NOT answer from your own "
    "knowledge. Your final message must tell the user you couldn't look this "
    "up because web search is currently unavailable."
)


def _search_unavailable(reason: str) -> str:
    """Log the specific cause for the operator and return the uniform
    model-facing unavailable message (loud, not raised)."""
    logger.error("Web search unavailable: %s", reason)
    return _SEARCH_UNAVAILABLE_MESSAGE

# --- Per-host fetch throttle ---
_host_lock = threading.Lock()
_host_last_call: dict[str, float] = {}
_HOST_MIN_DELAY = 1.0
# When the per-host dict crosses this size, drop entries we haven't touched in
# `_HOST_DICT_PRUNE_KEEP_S` seconds.  Bounds memory in a long-running process
# that fetches many distinct hosts (PR #118 Copilot review #1).  Cheap when
# small; the prune scan runs only on threshold-cross.
_HOST_DICT_PRUNE_THRESHOLD = 256
_HOST_DICT_PRUNE_KEEP_S = 60.0


def _host_throttle(host: str) -> None:
    """Block until at least ``_HOST_MIN_DELAY`` has passed since the last
    fetch to this specific ``host``.  No-op for empty/None host.

    Opportunistically prunes ``_host_last_call`` when it crosses
    ``_HOST_DICT_PRUNE_THRESHOLD`` entries — see the constant for the
    rationale (long-running process + many distinct hosts)."""
    if not host:
        return
    with _host_lock:
        now = time.time()
        last = _host_last_call.get(host, 0.0)
        elapsed = now - last
        if elapsed < _HOST_MIN_DELAY:
            time.sleep(_HOST_MIN_DELAY - elapsed)
            now = time.time()  # refresh after sleep so the recorded call-time is accurate
        _host_last_call[host] = now
        if len(_host_last_call) > _HOST_DICT_PRUNE_THRESHOLD:
            cutoff = now - _HOST_DICT_PRUNE_KEEP_S
            for h in list(_host_last_call):
                if _host_last_call[h] < cutoff:
                    del _host_last_call[h]


def read_url(url: str) -> str:
    """Extract the content from the given url.

    Per-host throttled (~1s between calls to the same host) so a burst of
    fetches to different sites isn't artificially serialised, but a tight
    loop against one bot-protected site is rate-limited locally."""
    # `urlparse` never raises on str input; for empty/malformed URLs
    # `.hostname` is None and `_host_throttle` no-ops on the empty string.
    host = urlparse(url).hostname or ""
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
    """Search the web via the self-hosted SearXNG metasearch at
    ``ASSIST_SEARCH_URL`` (private, multi-engine, no key).

    There is deliberately NO fallback.  If SearXNG is unset, unreachable,
    errors, returns a malformed response, or returns zero results while
    reporting any engine failures (``unresponsive_engines``), this RETURNS the
    explicit ``_SEARCH_UNAVAILABLE_MESSAGE`` (logged at ERROR) — a broken
    backend fails LOUDLY, but as a tool result the agent relays, not an
    exception that crashes the research turn.  A genuine empty result for a
    healthy query (zero results, no engine errors) returns ``"[]"`` so the
    agent can treat it as "no results"."""
    base_url = os.getenv("ASSIST_SEARCH_URL")
    if not base_url:
        return _search_unavailable(
            "ASSIST_SEARCH_URL is not set — a self-hosted SearXNG instance is "
            "required for web search (run `make searxng-up`)."
        )
    try:
        resp = requests.get(
            base_url.rstrip("/") + "/search",
            params={"q": query, "format": "json"},
            timeout=_SEARXNG_TIMEOUT_S,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        return _search_unavailable(f"SearXNG at {base_url} unreachable/errored: {e}")

    # A valid SearXNG response is a dict carrying a `results` list (possibly
    # empty).  Any deviation — non-dict body, missing key, non-list value (incl.
    # falsy {}/"", so don't coerce with `or []`) — is a malformed/unhealthy
    # backend, not a "no results" answer.
    if not isinstance(payload, dict):
        return _search_unavailable(
            f"SearXNG at {base_url} returned unexpected JSON shape: "
            f"{type(payload).__name__}"
        )
    if "results" not in payload:
        return _search_unavailable(f"SearXNG at {base_url} response missing 'results'")
    results = payload["results"]
    if not isinstance(results, list):
        return _search_unavailable(
            f"SearXNG at {base_url} 'results' not a list: {type(results).__name__}"
        )
    if not results:
        # Distinguish "empty results while at least one engine reported a
        # failure" (a loud backend failure) from a genuine empty result set for
        # this query.  SearXNG always returns `unresponsive_engines` as a list
        # (failing engines, empty when all healthy); a missing key means "none"
        # but a present non-list value is a malformed/unhealthy backend.
        unresponsive = payload.get("unresponsive_engines", [])
        if not isinstance(unresponsive, list):
            return _search_unavailable(
                f"SearXNG at {base_url} 'unresponsive_engines' not a list: "
                f"{type(unresponsive).__name__}"
            )
        if unresponsive:
            return _search_unavailable(
                f"SearXNG at {base_url} returned no results and engines failed: "
                f"{unresponsive}"
            )
        return "[]"

    normalized = [
        {"title": r.get("title", ""), "url": r.get("url", ""),
         "content": r.get("content", "")}
        for r in results[:max_results]
    ]
    return str(normalized)
