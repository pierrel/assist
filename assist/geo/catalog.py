"""``Catalog`` — the side-effect-free READER over the pinned downloadable-region
snapshot (``<geo_dir>/catalog.json``).

Two jobs: the download **allowlist** (``is_allowed`` — a slug the agent/route asks to
download MUST be a snapshot id; the PBF URL then comes from the entry, never a caller)
and the **resolve/picker source** (``get`` / ``search`` — how ``find_regions`` maps a
place the user named to a canonical id, so the small model never synthesizes a
Geofabrik id).

No network here: the snapshot is BUILT by ``build_catalog`` from the assist venv (fetch
the Geofabrik index, derive bboxes, merge the transit overlay) and only READ on the
request path.
A small mtime cache avoids re-parsing ~555 entries per call; the builder writes
atomically (tmp+rename), so a reader sees the old or the new file, never a torn one.
"""
from __future__ import annotations

import json
import logging
import os
import threading

from assist.geo.model import BBox, CatalogEntry

logger = logging.getLogger(__name__)

CATALOG_FILE = "catalog.json"


def _area(bbox: BBox) -> float:
    lon0, lat0, lon1, lat1 = bbox
    return (lon1 - lon0) * (lat1 - lat0)


class Catalog:
    """Read-only view of the pinned catalog snapshot. ``geo_dir`` is the travel-infra
    dir shared with the registry."""

    def __init__(self, geo_dir: str):
        self._path = os.path.join(geo_dir, CATALOG_FILE)
        self._lock = threading.Lock()
        self._mtime: float | None = None
        self._by_slug: dict[str, CatalogEntry] = {}

    def _entries(self) -> dict[str, CatalogEntry]:
        """Slug→entry, reloaded only when the snapshot file changes (mtime cache)."""
        with self._lock:
            try:
                mtime = os.path.getmtime(self._path)
            except FileNotFoundError:
                self._mtime, self._by_slug = None, {}
                return self._by_slug
            if mtime != self._mtime:
                self._by_slug = self._load()
                self._mtime = mtime
            return self._by_slug

    def _load(self) -> dict[str, CatalogEntry]:
        try:
            with open(self._path) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        if not isinstance(data, list):
            logger.warning("geo: catalog.json is not a list; ignoring")
            return {}
        out: dict[str, CatalogEntry] = {}
        for d in data:
            entry = CatalogEntry.from_dict(d) if isinstance(d, dict) else None
            if entry is not None:
                out[entry.slug] = entry
        return out

    def all(self) -> list[CatalogEntry]:
        return list(self._entries().values())

    def get(self, slug: str) -> CatalogEntry | None:
        return self._entries().get(slug)

    def is_allowed(self, slug: str) -> bool:
        """The download allowlist: a slug is fetchable only if it's a snapshot id."""
        return slug in self._entries()

    def search(self, query: str) -> list[CatalogEntry]:
        """Match a place query to catalog regions, smallest-extent first (prefer the
        tightest region covering the need). Matches either way: the query is part of a
        region's name/slug (a partial query like "washing"), OR a region's name is named
        INSIDE the query (the common "<place>, <country/state>" shape the model produces,
        e.g. "Berlin, Germany" → the "Berlin" region, "Portland, Oregon" → "Oregon")."""
        q = query.strip().lower()
        if not q:
            return []
        hits = []
        for e in self._entries().values():
            name, slug = e.display_name.lower(), e.slug.lower()
            if q in name or q in slug or (len(name) >= 4 and name in q):
                hits.append(e)
        return sorted(hits, key=lambda e: _area(e.bbox))
