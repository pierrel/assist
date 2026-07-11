"""The read-only geo agent tools — ``list_regions`` (what's covered) and
``find_regions`` (resolve a place the user named → the exact downloadable-region id).

Wired into the web ``AgentSpec`` NORMAL tool set (like ``notify``) — NOT the untrusted
SMS-triage set (an inbound text must not drive region logic) and not a core built-in
(the registry/catalog live in the web deployment's travel-infra dir; emacsos/CLI have
neither). Built by ``geo_tools(registry, catalog)`` closing over the injected stores so
this module imports no web state.

``find_regions`` is the B1 fix: the small model can't reliably reproduce Geofabrik's id
scheme (``us/washington`` vs the bare ``socal``), so it never types an id — it passes a
place/region NAME as the user said it (or the US state it inferred a city is in, e.g.
"Seattle" → "Washington"), and code returns the exact canonical id(s) to use when
proposing a download. Both tools are pure queries (CQS) that never raise into the loop.
"""
from __future__ import annotations

from assist.geo.catalog import Catalog
from assist.geo.model import STATE_READY
from assist.geo.registry import RegionRegistry

_MAX_CANDIDATES = 8


def geo_tools(registry: RegionRegistry, catalog: Catalog) -> list:
    """Return [list_regions, find_regions], closing over the loaded-region registry and
    the downloadable-region catalog."""

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
        address you can't locate — the US state or region it is in that you infer (e.g.
        for "Seattle" pass "Washington"; for "a cafe in San Diego" pass "Southern
        California" or "California"). NEVER type a region id yourself — pass a name and
        use the exact id from the result.

        Returns candidate regions, each with its exact id (and whether it's already
        loaded), smallest area first (prefer the tightest region covering the need). If
        nothing matches, the place may be a typo or you named the wrong area — ask the
        user to clarify which state/region it's in.
        """
        hits = catalog.search(query)
        if not hits:
            return (f"No downloadable region matches \"{query}\". Ask the user which US "
                    "state or larger region the place is in, then search that.")
        loaded = {r.slug for r in registry.all()}
        lines = [f"- {e.display_name} [id: {e.slug}]"
                 f"{' — already loaded' if e.slug in loaded else ''}"
                 for e in hits[:_MAX_CANDIDATES]]
        more = "" if len(hits) <= _MAX_CANDIDATES else f"\n(+{len(hits) - _MAX_CANDIDATES} more; refine the name)"
        return ("Matching downloadable regions (use the exact id when proposing a "
                "download):\n" + "\n".join(lines) + more)

    return [list_regions, find_regions]
