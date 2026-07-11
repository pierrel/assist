"""Build the pinned catalog snapshot from the Geofabrik index — the write-time half
of the catalog (the read-time half is ``catalog.py``; see the C3 builder/reader split
in docs/2026-07-10-geo-download-on-demand.org).

Run by the weekly travel-data refresh (never on the request path)::

    python -m assist.geo.build_catalog "$TRAVEL_INFRA_DIR"

Fetches ``index-v1.json`` (a GeoJSON FeatureCollection of ~555 extracts), and for each
feature: validates the PBF URL is https on the Geofabrik host (T2 — a poisoned entry
must not reach curl later), derives the bbox from the (Multi)Polygon geometry, derives a
human display name (52/555 entries carry the raw id as their name — "us/new-york"
becomes "New York", else ``find_regions("Washington")`` can't match), and merges the
hand-curated transit overlay (``transit-overlay.json``, a separate file precisely so a
weekly rebuild can't clobber the curation). Writes ``catalog.json`` atomically
(tmp+rename), so a concurrent reader sees old-or-new, never torn.

No sizes are cached here: the one deterministic size figure the HITL approval needs is
a single constrained HEAD in the propose tool body (T4) — pre-sizing 555 regions
weekly was reviewed out as gold-plating.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
from urllib.parse import urlparse

import requests

from assist.geo.catalog import CATALOG_FILE

logger = logging.getLogger(__name__)

INDEX_URL = "https://download.geofabrik.de/index-v1.json"
# The ONLY host a catalog PBF URL may point at (T2). The index is third-party input;
# an entry failing this is dropped, never carried into the allowlist.
ALLOWED_PBF_HOST = "download.geofabrik.de"
TRANSIT_OVERLAY_FILE = "transit-overlay.json"
_FETCH_TIMEOUT_S = 60


# Display names are third-party index text but land in agent-facing prompts (find_regions
# / list_regions output, the propose reply, and — highest trust — the completion turn
# dispatched as a fresh message). Sanitize at this one build-time chokepoint (T6) so every
# downstream interpolation is over a bounded value: a benign charset, single line, capped.
_NAME_OK = re.compile(r"[^A-Za-z0-9 ,.()'\-/&]")
_NAME_MAX = 64
# A catalog slug becomes the download allowlist + a script argv + an on-disk filename, so
# constrain it here (the same rule as travel-lib.sh:valid_slug): lowercase a-z0-9, '-',
# nested '/', no leading/trailing '/', no '..' or '//', ≤64 chars.
_SLUG_OK = re.compile(r"^[a-z0-9]([a-z0-9/-]*[a-z0-9])?$")


def _valid_slug(s: str) -> bool:
    return (isinstance(s, str) and 0 < len(s) <= 64
            and bool(_SLUG_OK.match(s)) and ".." not in s and "//" not in s)


def _slug_derived_name(entry_id: str) -> str:
    return entry_id.rsplit("/", 1)[-1].replace("-", " ").title()


def _display_name(entry_id: str, name) -> str:
    """A safe human name: the index ``name`` when it's a clean string different from the
    id, else derived from the id's last segment (the id-as-name case is 52/555 — all US
    states). Untrusted input is stripped to a benign charset + capped (T6)."""
    if isinstance(name, str) and name and name != entry_id:
        cleaned = _NAME_OK.sub("", name).strip()[:_NAME_MAX].strip()
        if cleaned:
            return cleaned
    return _slug_derived_name(entry_id)


def _bbox_of(geometry: dict) -> list[float] | None:
    """(min_lon, min_lat, max_lon, max_lat) over a (Multi)Polygon's coordinates.
    Geometry is untrusted third-party input: a non-numeric/non-finite coordinate or a
    malformed ring drops the feature (return None), never aborts the whole build."""
    try:
        coords = geometry.get("coordinates")
        gtype = geometry.get("type")
        if gtype == "Polygon":
            rings = coords or []
        elif gtype == "MultiPolygon":
            rings = [ring for poly in (coords or []) for ring in poly]
        else:
            return None
        lons, lats = [], []
        for ring in rings:
            for pt in ring:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    lon, lat = float(pt[0]), float(pt[1])
                    if math.isfinite(lon) and math.isfinite(lat):
                        lons.append(lon)
                        lats.append(lat)
    except (TypeError, ValueError):
        return None
    if not lons:
        return None
    return [min(lons), min(lats), max(lons), max(lats)]


def _entry_from_feature(feature: dict, transit: dict[str, str]) -> dict | None:
    """One index feature → one catalog entry dict, or None (dropped, logged)."""
    props = feature.get("properties") or {}
    entry_id = props.get("id")
    pbf = (props.get("urls") or {}).get("pbf")
    if not isinstance(entry_id, str) or not entry_id or not isinstance(pbf, str):
        logger.warning("catalog build: feature without id/pbf skipped: %r", props)
        return None
    if not _valid_slug(entry_id):
        logger.warning("catalog build: id %r fails the slug charset, DROPPED", entry_id)
        return None
    parsed = urlparse(pbf)
    if parsed.scheme != "https" or parsed.netloc != ALLOWED_PBF_HOST:
        logger.warning("catalog build: %s PBF URL off-host/non-https, DROPPED: %s",
                       entry_id, pbf)
        return None
    bbox = _bbox_of(feature.get("geometry") or {})
    if bbox is None:
        logger.warning("catalog build: %s has no usable geometry, skipped", entry_id)
        return None
    return {"slug": entry_id,
            "display_name": _display_name(entry_id, props.get("name")),
            "bbox": bbox,
            "url": pbf,
            "size_bytes": None,
            "transit_feed": transit.get(entry_id)}


def _load_transit_overlay(geo_dir: str) -> dict[str, str]:
    """The hand-curated slug→feed map (e.g. ``{"norcal": "511:RG"}``). Its own file so
    a regenerate can't clobber it; missing/corrupt → no transit anywhere."""
    path = os.path.join(geo_dir, TRANSIT_OVERLAY_FILE)
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        logger.warning("catalog build: %s is not an object; ignoring", path)
        return {}
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}


def build(geo_dir: str, index: dict) -> int:
    """Transform a fetched index into ``<geo_dir>/catalog.json``. Returns the entry
    count. Refuses (no write) an implausibly small result so a truncated fetch can't
    replace a good snapshot with a near-empty allowlist — validate-before-swap."""
    transit = _load_transit_overlay(geo_dir)
    entries = [e for f in (index.get("features") or [])
               if (e := _entry_from_feature(f, transit)) is not None]
    if len(entries) < 100:   # the real index is ~555; a tiny result is a bad fetch
        raise ValueError(f"catalog build: only {len(entries)} usable entries — "
                         "refusing to replace the snapshot with a near-empty allowlist")
    path = os.path.join(geo_dir, CATALOG_FILE)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(entries, f)
    os.replace(tmp, path)
    return len(entries)


def fetch_index(url: str = INDEX_URL) -> dict:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != ALLOWED_PBF_HOST:
        raise ValueError(f"index URL must be https on {ALLOWED_PBF_HOST}: {url}")
    resp = requests.get(url, timeout=_FETCH_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if len(argv) != 2:
        print("usage: python -m assist.geo.build_catalog <geo_dir>", file=sys.stderr)
        return 2
    geo_dir = argv[1]
    if not os.path.isdir(geo_dir):
        print(f"not a directory: {geo_dir}", file=sys.stderr)
        return 2
    count = build(geo_dir, fetch_index())
    logger.info("catalog build: wrote %d entries to %s",
                count, os.path.join(geo_dir, CATALOG_FILE))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
