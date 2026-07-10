"""The two geo records.

``Region`` is a LOADED region's operational state — the registry's unique payload,
the part the catalog can't know: which slugs are loaded, their import state, whether
transit is on, when last used, whether the "it's ready" notice was delivered.
``display_name``/``bbox`` are copied in at add-time (from the catalog) so the ``/geo``
page can render the loaded list without a catalog join and stays correct even if a
later catalog snapshot drops the id.

``CatalogEntry`` is a DOWNLOADABLE region from the pinned Geofabrik-index snapshot —
the download allowlist + the picker/resolve source. ``url`` (the PBF) always comes
from the index, never a caller; ``size_bytes`` is cached at snapshot-build time (no
live request-path fetch); ``transit_feed`` is the hand-curated overlay's feed id for
this region, or None where transit isn't configured.

Both parse leniently: ``from_dict`` returns ``None`` on a malformed/short record so a
single bad entry can't poison the whole file (the stores skip Nones), mirroring the
shape-validation discipline in ``manage/web/state.py``'s timings reader.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace

logger = logging.getLogger(__name__)

# A region's import lifecycle. importing → ready (validated import) | failed.
STATE_IMPORTING = "importing"
STATE_READY = "ready"
STATE_FAILED = "failed"
_STATES = (STATE_IMPORTING, STATE_READY, STATE_FAILED)

BBox = tuple[float, float, float, float]   # (min_lon, min_lat, max_lon, max_lat)


def _parse_bbox(v) -> BBox | None:
    """A bbox is exactly four finite numbers [min_lon, min_lat, max_lon, max_lat]."""
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    try:
        lon0, lat0, lon1, lat1 = (float(x) for x in v)
    except (TypeError, ValueError):
        return None
    if lon0 > lon1 or lat0 > lat1:
        return None
    return (lon0, lat0, lon1, lat1)


@dataclass(frozen=True)
class Region:
    """One loaded region's operational state (a ``regions.json`` record)."""
    slug: str
    display_name: str
    bbox: BBox
    has_transit: bool = False
    state: str = STATE_IMPORTING
    last_used_at: str | None = None        # ISO-8601 UTC; None = never used
    completion_delivered: bool = True       # False only while a ready import owes a notice
    added_at: str | None = None            # ISO-8601 UTC

    def with_state(self, state: str) -> "Region":
        return replace(self, state=state)

    def to_dict(self) -> dict:
        return {"slug": self.slug, "display_name": self.display_name,
                "bbox": list(self.bbox), "has_transit": self.has_transit,
                "state": self.state, "last_used_at": self.last_used_at,
                "completion_delivered": self.completion_delivered,
                "added_at": self.added_at}

    @classmethod
    def from_dict(cls, d: dict) -> "Region | None":
        slug = d.get("slug")
        bbox = _parse_bbox(d.get("bbox"))
        if not isinstance(slug, str) or not slug or bbox is None:
            logger.warning("geo: skipping malformed region record: %r", d)
            return None
        state = d.get("state")
        return cls(
            slug=slug,
            display_name=d.get("display_name") or slug,
            bbox=bbox,
            has_transit=bool(d.get("has_transit", False)),
            state=state if state in _STATES else STATE_READY,
            last_used_at=d.get("last_used_at"),
            completion_delivered=bool(d.get("completion_delivered", True)),
            added_at=d.get("added_at"))


@dataclass(frozen=True)
class CatalogEntry:
    """One downloadable region from the pinned Geofabrik-index snapshot."""
    slug: str
    display_name: str
    bbox: BBox
    url: str                                # the PBF download URL (from the index only)
    size_bytes: int | None = None          # cached at snapshot-build time; None = unknown
    transit_feed: str | None = None        # hand-curated overlay feed id, or None

    def to_dict(self) -> dict:
        return {"slug": self.slug, "display_name": self.display_name,
                "bbox": list(self.bbox), "url": self.url,
                "size_bytes": self.size_bytes, "transit_feed": self.transit_feed}

    @classmethod
    def from_dict(cls, d: dict) -> "CatalogEntry | None":
        slug = d.get("slug")
        url = d.get("url")
        bbox = _parse_bbox(d.get("bbox"))
        if not isinstance(slug, str) or not slug or not isinstance(url, str) or bbox is None:
            logger.warning("geo: skipping malformed catalog entry: %r", d)
            return None
        size = d.get("size_bytes")
        return cls(
            slug=slug,
            display_name=d.get("display_name") or slug,
            bbox=bbox,
            url=url,
            size_bytes=size if isinstance(size, int) and size >= 0 else None,
            transit_feed=d.get("transit_feed") or None)
