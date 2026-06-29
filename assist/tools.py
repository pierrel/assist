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
from html.parser import HTMLParser
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
    "Web search is unavailable — the search backend could not be reached or "
    "returned an unusable response. STOP: do not search again, and do not "
    "retry with a different query — it will keep failing the same way. Do NOT "
    "answer from your own knowledge. Your final message must tell the user you "
    "couldn't look this up because web search is currently unavailable."
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


# Tags whose TEXT is never readable content. Kept minimal and matched to the
# prior whole-page strip (script/style only) so the no-article path cannot
# regress. We deliberately KEEP <noscript>: read_url runs no JS, so a page's
# no-JS fallback content is exactly what we want (a real fetch of a JS-gated
# forum returned only its <noscript> text — dropping it returned nothing).
_NOISE_TAGS = {"script", "style"}
# Page regions that mark the main article body when present.
_MAIN_TAGS = {"article", "main"}


class _MainContentExtractor(HTMLParser):
    """One-pass HTML → text that prefers the marked main article.

    Collects two text streams while parsing: ``main`` (text inside
    ``<article>``/``<main>``, kept WITH that region's own heading/byline/footer)
    and ``body`` (all page text minus ``_NOISE_TAGS``). ``text()`` returns the
    main stream when the page marked one, else the body stream — so a page with
    no ``<article>`` degrades to the same whole-page strip as before, never less.

    Using the stdlib parser (not regex) handles nesting, quoted attributes
    containing ``>``, comments, and entities by construction — the failure
    classes a regex tag-stripper gets wrong."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._noise = 0
        self._main = 0
        self._main_text: list[str] = []
        self._body_text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _NOISE_TAGS:
            self._noise += 1
        elif tag in _MAIN_TAGS:
            self._main += 1
        self._boundary()

    def handle_endtag(self, tag):
        if tag in _NOISE_TAGS and self._noise:
            self._noise -= 1
        elif tag in _MAIN_TAGS and self._main:
            self._main -= 1
        self._boundary()

    def _boundary(self):
        # A tag transition is a word boundary: adjacent elements' text must not
        # fuse ("<p>one</p><p>two</p>" -> "one two", not "onetwo"). Mirrors the
        # old strip's "every tag -> space"; the final whitespace-collapse in
        # text() absorbs the extra spaces.
        if self._noise:
            return
        self._body_text.append(" ")
        if self._main:
            self._main_text.append(" ")

    def handle_data(self, data):
        if self._noise:
            return
        self._body_text.append(data)
        if self._main:
            self._main_text.append(data)

    def text(self) -> str:
        chosen = (self._main_text if any(s.strip() for s in self._main_text)
                  else self._body_text)
        return re.sub(r"\s+", " ", "".join(chosen)).strip()


def _extract_main_content(html: str) -> str:
    """Article text where the page marks one (``<article>``/``<main>``), else the
    whole-page text with scripts/styles removed. See ``_MainContentExtractor``."""
    parser = _MainContentExtractor()
    parser.feed(html)
    parser.close()  # flush any token left dangling at end-of-document
    return parser.text()


def read_url(url: str) -> str:
    """Extract the readable content from the given url.

    Returns the page's main article text where the page marks one
    (``<article>``/``<main>``) so the char budget holds signal, not nav/footer
    chrome; degrades to a whole-page text strip (scripts/styles removed) when
    the page marks no article. Capped at 4000 chars.

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
        return _extract_main_content(resp.text)[:4000]
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


# --- Local travel: real-world time/distance via the self-hosted MOTIS engine ---
#
# ONE service (ASSIST_ROUTING_URL) does geocoding + multimodal routing.  Same
# shape as search_internet: read the URL from env, call it, normalize to a
# model-facing string, FAIL LOUD-BUT-RETURNED (never raise into the agent loop).
# Fully private (self-hosted); coverage = the regions loaded into MOTIS, so an
# out-of-region place comes back as no route, never a guess.  See
# docs/2026-06-28-local-travel-skill.org.
# Per-call timeout to the (local) MOTIS service.  Kept tight: travel() makes up
# to 6 sequential calls, but if MOTIS is down the FIRST geocode call raises here
# and travel() aborts to "unavailable" (~one timeout), so the realistic worst
# case is bounded; a full slow-but-up MOTIS is the only path to several timeouts.
_TRAVEL_TIMEOUT_S = 8.0
# MOTIS direct (street) modes -> user label.  Transit is queried separately.
_TRAVEL_DIRECT_MODES = (("Car", "CAR"), ("Bike", "BIKE"), ("Walk", "WALK"))
# Generous cap on a single direct leg so a long intra-metro walk/bike still
# returns (MOTIS's default maxDirectTime drops them).
_TRAVEL_MAX_DIRECT_S = 4 * 3600

_TRAVEL_UNAVAILABLE_MESSAGE = (
    "Travel routing is unavailable -- the routing service could not be reached. "
    "Tell the user you couldn't look up travel times right now. Do NOT estimate "
    "the distance or time from your own knowledge -- give no numbers."
)


def _travel_unavailable(reason: str) -> str:
    logger.error("Travel unavailable: %s", reason)
    return _TRAVEL_UNAVAILABLE_MESSAGE


def _fmt_duration(seconds: float) -> str:
    m = int(round(seconds / 60.0))
    if m < 60:
        return f"{m} min"
    return f"{m // 60} h {m % 60:02d} min"


def _fmt_distance_m(meters: float) -> str:
    km = meters / 1000.0
    return f"{km:.1f} km" if km >= 0.1 else f"{int(round(meters))} m"


class _TravelBackendError(Exception):
    """The routing service is unset / unreachable / errored — distinct from a
    successful response that simply has no match or no route, so travel() can say
    "unavailable" rather than misleadingly "couldn't find that place"."""


def _motis_get(path: str, params: dict) -> dict | list:
    """GET a MOTIS API path -> parsed JSON (a list for /geocode, a dict for
    /plan).  Raises _TravelBackendError when the
    service is unset/unreachable/errors (callers turn that into the unavailable
    message); a successful-but-empty response is the caller's "no result"."""
    base = os.getenv("ASSIST_ROUTING_URL")
    if not base:
        raise _TravelBackendError("ASSIST_ROUTING_URL is not set")
    try:
        resp = requests.get(base.rstrip("/") + path, params=params,
                            timeout=_TRAVEL_TIMEOUT_S)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise _TravelBackendError(f"MOTIS {path} failed: {e}") from e


def _geocode(place: str) -> dict | None:
    """Resolve a place NAME to coords via MOTIS geocoding (self-hosted).  Returns
    {lat, lon, name} (top match, with the resolved name so the agent can say what
    it routed to), or None if nothing matched.  Raises _TravelBackendError if the
    service is down (so the caller distinguishes that from a genuine no-match)."""
    hits = _motis_get("/api/v1/geocode", {"text": place})
    if not isinstance(hits, list):  # wrong shape = a backend problem, not "no match"
        raise _TravelBackendError(f"unexpected geocode response: {type(hits).__name__}")
    for h in hits:  # take the first USABLE hit; skip a malformed one (no false not-found)
        try:
            return {"lat": float(h["lat"]), "lon": float(h["lon"]),
                    "name": h.get("name") or place}  # never propagate a blank name
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return None


def _plan_direct(o: dict, d: dict, mode: str) -> dict | None:
    """A car/bike/walk (street) route via MOTIS /plan.  Returns
    {duration_s, distance_m} or None (service down / no route)."""
    try:
        data = _motis_get("/api/v1/plan", {
            "fromPlace": f"{o['lat']},{o['lon']}", "toPlace": f"{d['lat']},{d['lon']}",
            "directModes": mode, "maxDirectTime": _TRAVEL_MAX_DIRECT_S})
    except _TravelBackendError:
        return None  # a transient per-mode failure -> that mode shows "unavailable"
    # A non-dict 200 body (proxy/error page) would make .get raise -> contract says
    # never raise into the agent loop, so treat any non-dict as "no route".
    direct = data.get("direct") or [] if isinstance(data, dict) else []
    if not direct:
        return None
    try:  # never raise into the agent loop on a malformed itinerary (module contract)
        it = direct[0]
        dist = sum(leg.get("distance", 0) or 0 for leg in it.get("legs", []))
        return {"duration_s": float(it["duration"]), "distance_m": float(dist)}
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def _plan_transit(o: dict, d: dict) -> dict | None:
    """The fastest public-transit journey via MOTIS /plan.  Returns {duration_s}
    or None (no transit coverage / no journey).  Distance is omitted -- multimodal
    legs don't give a clean single road-distance (see the design doc)."""
    try:
        data = _motis_get("/api/v1/plan", {
            "fromPlace": f"{o['lat']},{o['lon']}", "toPlace": f"{d['lat']},{d['lon']}",
            "transitModes": "TRANSIT"})
    except _TravelBackendError:
        return None
    its = data.get("itineraries") or [] if isinstance(data, dict) else []
    if not its:
        return None
    try:  # never raise into the agent loop on a malformed itinerary (module contract)
        return {"duration_s": float(min(it["duration"] for it in its))}
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def travel(origin: str, destination: str) -> str:
    """Real-world travel time and distance between two places, by car, bike,
    walking, and public transit.

    Use this for any "how long / how far from A to B" or "is it faster to bike or
    take the train" question.  It gives times + distances, NOT turn-by-turn
    directions.  Pass plain place
    NAMES/addresses as the user said them (e.g. "the Ferry Building", "123 Main
    St") -- do NOT pass coordinates.  Returns a short per-mode summary computed by
    the routing service -- time and distance for car/bike/walk, time for transit
    (transit distance isn't reported); relay those numbers, never invent your own.
    Covers the loaded metro area(s); a place outside them comes back as no route,
    not a guess.
    """
    try:
        o = _geocode(origin)
        d = _geocode(destination)
    except _TravelBackendError as e:
        return _travel_unavailable(str(e))  # service down -> "unavailable", not "not found"
    if o is None:
        return (f"I couldn't find a place matching '{origin}'. Ask the user to "
                "clarify or give a more specific location.")
    if d is None:
        return (f"I couldn't find a place matching '{destination}'. Ask the user "
                "to clarify or give a more specific location.")

    lines = [f'Travel from "{o["name"]}" to "{d["name"]}":']
    for label, mode in _TRAVEL_DIRECT_MODES:
        r = _plan_direct(o, d, mode)
        lines.append(
            f"- {label}: {_fmt_duration(r['duration_s'])}, {_fmt_distance_m(r['distance_m'])}"
            if r else f"- {label}: unavailable")
    t = _plan_transit(o, d)
    lines.append(f"- Transit: {_fmt_duration(t['duration_s'])}" if t
                 else "- Transit: unavailable")
    return "\n".join(lines)
