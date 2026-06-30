"""Tools the agent exposes: a self-hosted SearXNG metasearch, a per-host
throttled URL fetch, and a self-hosted MOTIS-backed ``travel`` tool (real-world
time/distance by car/bike/walk/transit — see the travel() section below;
geocoding goes through an optional self-hosted Nominatim via ``ASSIST_GEOCODER_URL``).

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
import math
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
    miles = meters / 1609.344  # US units throughout (travel + directions)
    return f"{miles:.1f} mi" if miles >= 0.1 else f"{int(round(meters * 3.28084))} ft"


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


def _first_usable_hit(hits, place: str) -> dict | None:
    """First geocoder hit with parseable coords -> {lat, lon, name}; skip a
    malformed one (no false not-found); None if none usable.  A non-list `hits`
    is a wrong-shape 200 = backend problem, so raise _TravelBackendError (the
    caller turns that into "unavailable", not "couldn't find").  Works for both
    backends: MOTIS hits carry `name`; Nominatim carries `name`/`display_name`."""
    if not isinstance(hits, list):
        raise _TravelBackendError(f"unexpected geocoder response: {type(hits).__name__}")
    for h in hits:
        try:
            lat, lon = float(h["lat"]), float(h["lon"])
            name = h.get("name") or h.get("display_name") or place
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:  # also rejects NaN/inf
            return {"lat": lat, "lon": lon, "name": name}
        # out-of-range/NaN/inf coords would poison /plan as "nan,nan" -> skip the hit
    return None


def _nominatim_geocode(place: str) -> dict | None:
    """Geocode a place NAME via the self-hosted Nominatim search API.  The loaded
    OSM extract IS the region, so no country/viewbox filter is needed.  Raises
    _TravelBackendError if Nominatim is unreachable/errors (callers -> unavailable)."""
    base = os.getenv("ASSIST_GEOCODER_URL")
    try:
        resp = requests.get(base.rstrip("/") + "/search",
                            params={"q": place, "format": "jsonv2", "limit": 5},
                            timeout=_TRAVEL_TIMEOUT_S)
        resp.raise_for_status()
        hits = resp.json()
    except Exception as e:
        raise _TravelBackendError(f"Nominatim /search failed: {e}") from e
    return _first_usable_hit(hits, place)


def _parse_coord_string(place: str) -> dict | None:
    """A bare ``"lat,lon"`` (exactly two in-range numbers) → a geocode hit
    {lat, lon, name}, so "from here" routes from the user's coordinates (from the
    message context) without a forward-geocode. A real place name (anything that
    isn't two numbers) → None, falling through to geocoding."""
    parts = str(place).split(",")
    if len(parts) != 2:
        return None
    try:
        lat, lon = float(parts[0].strip()), float(parts[1].strip())
    except (ValueError, TypeError):
        return None
    if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
        return {"lat": lat, "lon": lon, "name": "your location"}
    return None


def _geocode(place: str) -> dict | None:
    """Resolve a place NAME to {lat, lon, name} (top match, with the resolved name
    so the agent can say what it routed to), or None if nothing matched.  A bare
    ``"lat,lon"`` is passed through directly (no geocode) so "from here" routes from
    the user's coordinates.  Uses the self-hosted Nominatim geocoder when
    ASSIST_GEOCODER_URL is set (far better at real addresses/POIs than MOTIS's
    built-in geocoder), else falls back to MOTIS /api/v1/geocode.  Raises
    _TravelBackendError if the chosen backend is down."""
    coords = _parse_coord_string(place)
    if coords:
        return coords
    if os.getenv("ASSIST_GEOCODER_URL"):
        return _nominatim_geocode(place)
    return _first_usable_hit(_motis_get("/api/v1/geocode", {"text": place}), place)


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


def _resolve_od(origin: str, destination: str):
    """Geocode origin + destination -> (o, d) dicts, or an error STRING the caller
    returns to the agent: service down -> the "unavailable" message; a name that
    doesn't resolve -> "couldn't find …".  Shared by travel() and directions()."""
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
    return o, d


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
    # Routing is required for the whole tool; without it, geocoding (now possibly
    # via Nominatim) would succeed and every mode would come back "unavailable" --
    # so fail fast with the standard message instead.
    if not os.getenv("ASSIST_ROUTING_URL"):
        return _travel_unavailable("ASSIST_ROUTING_URL is not set")
    od = _resolve_od(origin, destination)
    if isinstance(od, str):
        return od
    o, d = od

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


# --- directions (turn-by-turn) -------------------------------------------------
# `directions` is the step-by-step sibling of `travel`: same _geocode + MOTIS /plan
# + fail-loud-but-RETURNED contract, but it walks the per-leg/step detail travel
# discards.  MOTIS's own turn field (relativeDirection) is uniformly "CONTINUE", so
# street turns are derived from polyline geometry, CONFIDENCE-GATED: a left/right is
# emitted only when the heading change is unambiguous, else the neutral "Continue
# onto" -- so a confident WRONG turn is unreachable by construction.

_DIRECTIONS_STREET = {  # mode word -> (MOTIS directModes, label)
    "car": ("CAR", "Driving"), "drive": ("CAR", "Driving"), "driving": ("CAR", "Driving"),
    "bike": ("BIKE", "Biking"), "bicycle": ("BIKE", "Biking"), "cycling": ("BIKE", "Biking"),
    "cycle": ("BIKE", "Biking"),
    "walk": ("WALK", "Walking"), "walking": ("WALK", "Walking"), "foot": ("WALK", "Walking"),
    "on foot": ("WALK", "Walking"),
}
_DIRECTIONS_TRANSIT = {"transit", "bus", "train", "subway", "metro", "rail", "tram",
                       "light rail", "public transit"}
_TRANSIT_NOUN = {"BUS": "bus", "TRAM": "tram", "SUBWAY": "train", "RAIL": "train",
                 "FERRY": "ferry", "FUNICULAR": "funicular", "CABLE_CAR": "cable car",
                 "COACH": "coach"}
_TURN_STRAIGHT_DEG = 30.0   # |heading change| below this -> "Continue" (no turn)
_TURN_UTURN_DEG = 160.0     # above this -> U-turn
_MIN_RUN_M = 15             # drop negligible connector runs from the step list


def _normalize_mode(mode: str):
    """('street', MOTIS_directModes, label) | ('transit', None, 'Transit') | None."""
    m = str(mode or "").strip().lower()  # str() -> never raise on a non-string arg
    if m in _DIRECTIONS_STREET:
        directmode, label = _DIRECTIONS_STREET[m]
        return ("street", directmode, label)
    if m in _DIRECTIONS_TRANSIT:
        return ("transit", None, "Transit")
    return None


def _decode_polyline(points, precision) -> list:
    """Decode a Google-encoded polyline -> [(lat, lon), ...]; [] on any bad input
    (never raises -- module contract)."""
    if not isinstance(points, str):
        return []
    try:
        # Clamp the (untrusted backend) precision so factor is never 0 or absurd:
        # a very negative precision would underflow 10**p to 0.0 -> ZeroDivisionError.
        factor = 10 ** max(0, min(int(precision), 15))
        out = []
        lat = lon = 0
        i = 0
        n = len(points)
        while i < n:
            for axis in range(2):
                shift = result = 0
                while True:
                    b = ord(points[i]) - 63
                    i += 1
                    result |= (b & 0x1f) << shift
                    shift += 5
                    if b < 0x20:
                        break
                delta = ~(result >> 1) if (result & 1) else (result >> 1)
                if axis == 0:
                    lat += delta
                else:
                    lon += delta
            out.append((lat / factor, lon / factor))
        return out
    except (IndexError, TypeError, ValueError, OverflowError):
        return []  # OverflowError: int(Infinity) from a malformed precision field


def _bearing(a, b) -> float:
    """Initial compass bearing (0-360deg) from point a=(lat,lon) to b."""
    lat1, lat2 = math.radians(a[0]), math.radians(b[0])
    dlon = math.radians(b[1] - a[1])
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def _turn_phrase(prev_exit, this_entry) -> str:
    """Confidence-gated turn: commit to left/right ONLY when the heading change is
    unambiguous; otherwise 'Continue onto' (a confident wrong turn is unreachable)."""
    if prev_exit is None or this_entry is None:
        return "Continue onto"
    delta = ((this_entry - prev_exit + 180) % 360) - 180  # signed -180..180
    a = abs(delta)
    if a < _TURN_STRAIGHT_DEG:
        return "Continue onto"
    if a > _TURN_UTURN_DEG:
        return "Make a U-turn onto"
    return "Turn right onto" if delta > 0 else "Turn left onto"


def _consolidate_street_steps(steps) -> list:
    """Collapse consecutive same-streetName steps into runs with entry/exit bearings
    (from each run's decoded geometry).  -> [{name, distance_m, entry_bearing,
    exit_bearing}].  Never raises."""
    runs = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        name = (step.get("streetName") or "").strip()
        dist = step.get("distance")
        dist = dist if isinstance(dist, (int, float)) else 0  # never raise on a bad type
        pl = step.get("polyline") if isinstance(step.get("polyline"), dict) else {}
        pts = _decode_polyline(pl.get("points"), pl.get("precision", 7))
        if runs and runs[-1]["name"] == name:
            runs[-1]["distance_m"] += dist
            runs[-1]["_pts"].extend(pts)
        else:
            runs.append({"name": name, "distance_m": dist, "_pts": list(pts)})
    out = []
    for r in runs:
        if r["distance_m"] < _MIN_RUN_M:
            continue
        pts = r["_pts"]
        entry = _bearing(pts[0], pts[1]) if len(pts) >= 2 else None
        exit_ = _bearing(pts[-2], pts[-1]) if len(pts) >= 2 else None
        out.append({"name": r["name"], "distance_m": r["distance_m"],
                    "entry_bearing": entry, "exit_bearing": exit_})
    return out


def _street_narrative(o: dict, d: dict, directmode: str, label: str):
    """A car/bike/walk turn-by-turn list, or None (no route).  Raises
    _TravelBackendError only if MOTIS is down (caller -> 'unavailable')."""
    data = _motis_get("/api/v1/plan", {
        "fromPlace": f"{o['lat']},{o['lon']}", "toPlace": f"{d['lat']},{d['lon']}",
        "directModes": directmode, "maxDirectTime": _TRAVEL_MAX_DIRECT_S})
    try:
        direct = (data.get("direct") if isinstance(data, dict) else None) or []
        if not direct:
            return None
        legs = direct[0]["legs"]  # aggregate across legs, like travel._plan_direct sums them
        steps = [s for leg in legs for s in (leg.get("steps") or [])]
        total_dist = sum((leg.get("distance") or 0) for leg in legs)
        runs = _consolidate_street_steps(steps)
        if not runs:
            return None
        lines = [f'{label} directions from "{o["name"]}" to "{d["name"]}" '
                 f'(~{_fmt_duration(direct[0]["duration"])}, {_fmt_distance_m(total_dist)}):']
        prev_exit = None
        for n, r in enumerate(runs, start=1):
            phrase = "Head onto" if n == 1 else _turn_phrase(prev_exit, r["entry_bearing"])
            lines.append(f"{n}. {phrase} {r['name'] or 'an unnamed road'} "
                         f"({_fmt_distance_m(r['distance_m'])})")
            prev_exit = r["exit_bearing"]
        lines.append(f'{len(runs) + 1}. Arrive at "{d["name"]}"')
        return "\n".join(lines)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError):
        return None  # malformed itinerary -> "no route", never raise into the agent loop


def _transit_narrative(o: dict, d: dict):
    """A walk/board/transfer/alight transit list, or None (no journey).  Raises
    _TravelBackendError only if MOTIS is down."""
    data = _motis_get("/api/v1/plan", {
        "fromPlace": f"{o['lat']},{o['lon']}", "toPlace": f"{d['lat']},{d['lon']}",
        "transitModes": "TRANSIT"})
    try:
        its = (data.get("itineraries") if isinstance(data, dict) else None) or []
        if not its:
            return None
        it = min(its, key=lambda x: x["duration"])
        legs = [l for l in (it.get("legs") or [])  # drop degenerate zero-length legs
                if not (not l.get("duration") and
                        (l.get("from") or {}).get("name") == (l.get("to") or {}).get("name"))]
        if not legs:
            return None
        lines = [f'Transit directions from "{o["name"]}" to "{d["name"]}" '
                 f'(~{_fmt_duration(it["duration"])}):']
        for n, l in enumerate(legs, start=1):
            to_name = (l.get("to") or {}).get("name") or "your destination"
            if l.get("mode") == "WALK":
                dest = f'"{d["name"]}"' if n == len(legs) else to_name
                lines.append(f"{n}. Walk to {dest} ({_fmt_duration(l.get('duration', 0))})")
            else:
                noun = _TRANSIT_NOUN.get(l.get("mode"), "line")
                route = l.get("routeShortName") or l.get("tripShortName") or noun
                toward = f" toward {l['headsign']}" if l.get("headsign") else ""
                stops = len(l.get("intermediateStops") or []) + 1  # stops ridden to the alighting stop
                lines.append(f"{n}. Take the {route} {noun}{toward} to {to_name} "
                             f"({stops} stop{'s' if stops != 1 else ''})")
        return "\n".join(lines)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError):
        return None


def directions(origin: str, destination: str, mode: str) -> str:
    """Step-by-step route directions between two places for a SINGLE travel mode.

    Use this when the user wants DIRECTIONS / how to actually get somewhere -- "how
    do I get to X", "directions to Y", "which bus do I take", "walk me through it" --
    as opposed to "how long / how far" (use `travel` for that).  Pass plain place
    NAMES (the geocoder resolves them) and a `mode`: "car", "bike", "walk", or
    "transit".  Returns a numbered list -- turns + streets for car/bike/walk
    (street turns are approximate), or walk/board/transfer/alight for transit -- in
    US units (miles).  Relay it; never invent streets or lines.  A place outside the
    covered area or with no route comes back as "couldn't find a route", not a guess.
    """
    if not os.getenv("ASSIST_ROUTING_URL"):
        return _travel_unavailable("ASSIST_ROUTING_URL is not set")
    m = _normalize_mode(mode)
    if m is None:
        return ("Tell me which travel mode you want directions for: car, bike, "
                "walk, or transit.")
    kind, directmode, label = m
    od = _resolve_od(origin, destination)
    if isinstance(od, str):
        return od
    o, d = od
    try:
        out = _transit_narrative(o, d) if kind == "transit" else _street_narrative(o, d, directmode, label)
    except _TravelBackendError as e:
        return _travel_unavailable(str(e))  # plan service down -> "unavailable"
    if out is None:
        route_kind = "transit" if kind == "transit" else label.lower()
        return (f'I couldn\'t find a {route_kind} route from "{o["name"]}" to '
                f'"{d["name"]}".')
    return out
