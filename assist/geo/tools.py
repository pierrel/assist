"""The geo agent tools — ``list_regions`` (what's covered), ``find_regions`` (resolve a
place the user named → the exact downloadable-region id), and
``propose_region_download`` (record a HITL download proposal; never downloads).

Intended for the web ``AgentSpec`` NORMAL tool set (like ``notify``) — NOT the untrusted
SMS-triage set (an inbound text must not drive region logic) and not a core built-in
(the registry/catalog live in the web deployment's travel-infra dir; emacsos/CLI have
neither). That wiring lands with the web increment. Built by ``geo_tools(registry,
catalog, proposals)`` closing over the injected stores so this module imports no web
state.

``find_regions`` is the B1 fix: the small model can't reliably reproduce Geofabrik's id
scheme (``us/washington`` vs the bare ``socal``), so it never types an id — it passes a
place/region NAME as the user said it (or the state it inferred a city is in, e.g.
"Seattle" → "Washington"), and code returns the exact canonical id(s).
``propose_region_download`` is the matching by-construction sink: the id must be a
catalog id (``is_allowed``), and its body only RECORDS a proposal — the import fires
only from the user's approval on the recorded slug, so a hallucinated id or an injected
"download X" can never start a download. None of these raise into the loop.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
from langgraph.config import get_config

from assist.geo.catalog import Catalog
from assist.geo.model import STATE_IMPORTING, STATE_READY
from assist.geo.proposals import Proposal, ProposalStore
from assist.geo.registry import RegionRegistry

logger = logging.getLogger(__name__)

_MAX_CANDIDATES = 8
_HEAD_TIMEOUT_S = 10
_ALLOWED_PBF_HOST = "download.geofabrik.de"   # matches the builder's T2 gate

# U2: the cost the user consents to isn't bytes — it's the geocoder rebuild. A fixed,
# code-supplied line (never agent prose) shown with every proposal; the web approval
# UI reuses it.
DEGRADATION_WARNING = ("Adding a region downloads map data and rebuilds the geocoder "
                       "(can take up to a few hours); address lookups for all regions "
                       "are degraded until it finishes.")


def _thread_id() -> str | None:
    try:
        tid = ((get_config() or {}).get("configurable") or {}).get("thread_id")
    except RuntimeError:   # outside a langgraph runtime (direct call in a test)
        return None
    # coerce to str: the config may carry a UUID object (AgentHarness), and origin_tid
    # is persisted as JSON + passed to the dispatch callback — both want a string.
    return str(tid) if tid is not None else None


def _fmt_size(n: int | None) -> str:
    if n is None:
        return "size unknown"
    mb = n / 1e6
    return f"~{mb / 1000:.1f} GB" if mb >= 1000 else f"~{mb:.0f} MB"


def _head_size(url: str) -> int | None:
    """The one deterministic download-size figure (T4): a single bounded HEAD on the
    catalog's PBF URL. The URL is validated https-on-Geofabrik BEFORE the request, and
    redirects are NOT followed (T2 — a redirect would fire a request at an unvalidated
    host; the Geofabrik PBFs serve directly, so a redirect just yields no size). Any
    failure/redirect → None (size unknown), never a raise."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != _ALLOWED_PBF_HOST:
        logger.warning("geo: refusing size HEAD to off-host url %s", url)
        return None
    try:
        resp = requests.head(url, timeout=_HEAD_TIMEOUT_S, allow_redirects=False)
        n = int(resp.headers.get("Content-Length", ""))
        return n if n > 0 else None
    except (requests.RequestException, ValueError):
        return None


def geo_tools(registry: RegionRegistry, catalog: Catalog,
              proposals: ProposalStore) -> list:
    """Return [list_regions, find_regions, propose_region_download], closing over the
    loaded-region registry, the downloadable-region catalog, and the proposal store."""

    def list_regions() -> str:
        """List the geographic regions currently loaded — the areas you can do travel /
        directions / address lookups in — and whether each includes public-transit data.

        Call this when the user asks what areas or places you cover, and when a place
        can't be found (to tell them what IS covered before offering to download more),
        and before telling the user transit is unavailable somewhere (so you can name
        the region). Takes no arguments.
        """
        ready = [r for r in registry.all() if r.state == STATE_READY]
        if not ready:
            return "No regions are loaded yet."
        lines = [f"- {r.display_name} ({'with transit' if r.has_transit else 'no transit'})"
                 for r in sorted(ready, key=lambda r: r.display_name)]
        return "Loaded regions (travel/directions/geocoding work here):\n" + "\n".join(lines)

    def find_regions(query: str) -> str:
        """Find DOWNLOADABLE geographic regions matching a place, so you can offer to add
        one. Pass a region/state/area NAME as the user said it, or — for a city or
        address you can't locate — the state, province, or region it is in that you
        infer (e.g. for "Seattle" pass "Washington"; for "a cafe in San Diego" pass
        "Southern California" or "California"). NEVER type a region id yourself — pass a
        name and use the exact id from the result.

        Returns candidate regions, each with its exact id (and whether it's already
        loaded), smallest area first (prefer the tightest region covering the need). If
        nothing matches, the place may be a typo or you named the wrong area — ask the
        user to clarify which state/region it's in.
        """
        hits = catalog.search(query)
        if not hits:
            return (f"No downloadable region matches \"{query}\". Ask the user which US "
                    "state or larger region the place is in, then search that.")
        # READY only — matches list_regions/is_loaded; a failed/importing region must
        # NOT read as "already loaded" (else the agent tells the user they're covered
        # when they're not, and won't re-offer the download).
        loaded = {r.slug for r in registry.all() if r.state == STATE_READY}
        lines = [f"- {e.display_name} [id: {e.slug}]"
                 f"{' — already loaded' if e.slug in loaded else ''}"
                 for e in hits[:_MAX_CANDIDATES]]
        more = "" if len(hits) <= _MAX_CANDIDATES else f"\n(+{len(hits) - _MAX_CANDIDATES} more; refine the name)"
        return ("Matching downloadable regions (use the exact id when proposing a "
                "download):\n" + "\n".join(lines) + more)

    def propose_region_download(region_id: str, user_request: str) -> str:
        """Record a proposal to download a new geographic region, for the user to
        approve. This does NOT download anything — the download and import only start
        after the user approves the proposal themselves.

        Call this only after the user confirms they want the region added.
        ``region_id`` must be the exact id from a ``find_regions`` result (e.g.
        "us/washington") — never one you typed yourself. ``user_request`` is the user's
        original ask, verbatim (e.g. "directions in Portland next week"), so it can be
        picked back up once the region is ready.
        """
        slug = str(region_id or "").strip()
        # The by-construction sink (B1): only a catalog id is proposable; a typed /
        # hallucinated / injected id dies here, before any record exists.
        entry = catalog.get(slug)
        if entry is None:
            return (f"\"{slug}\" is not a downloadable region id. Call find_regions "
                    "with the place's name and use the exact id it returns.")
        existing = registry.get(slug)
        if existing is not None and existing.state == STATE_READY:
            return f"{entry.display_name} is already loaded — no download needed."
        if existing is not None and existing.state == STATE_IMPORTING:
            # already approved + downloading — re-proposing would overwrite the pending
            # record's thread/request (a TOCTOU on the approval), and the "awaiting
            # approval" reply would be false.
            return f"{entry.display_name} is already downloading — no new proposal needed."
        tid = _thread_id()
        if not tid:
            return "Couldn't record the proposal: no active thread."
        size = _head_size(entry.url)
        proposals.put(Proposal(
            slug=slug, display_name=entry.display_name, bbox=entry.bbox,
            size_bytes=size, origin_tid=tid,
            user_request=str(user_request or "")[:500],
            created_at=datetime.now(timezone.utc).isoformat()))
        return (f"Proposed downloading {entry.display_name} ({_fmt_size(size)}). "
                f"{DEGRADATION_WARNING} Nothing downloads until the user approves the "
                "proposal; tell them it's awaiting their approval.")

    return [list_regions, find_regions, propose_region_download]
