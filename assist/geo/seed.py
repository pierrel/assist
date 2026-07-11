"""Reconcile the registry with what's actually loaded on disk.

The provisioning scripts own the DATA (input/<slug>.osm.pbf per loaded region); the
registry owns the operational STATE. They can drift — a region added by running the
script directly (bootstrap/ops), or a registry file lost — so on web startup we
reconcile: every region-source extract present on disk becomes a ``ready`` registry
entry (name/bbox from the catalog), and the base region keeps its transit flag. This is
idempotent and only ADDS/normalizes ready entries for on-disk regions; it never removes
(a failed/importing entry with no file yet is the Provisioner's to manage).

Slug ↔ filename: the on-disk name flattens '/'→'-' (``us/washington`` →
``us-washington.osm.pbf``), so we recover the slug by matching the catalog (whose slugs
flatten to the same filename) — the catalog is the source of truth for the real id.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from assist.geo.catalog import Catalog
from assist.geo.model import Region, STATE_READY
from assist.geo.registry import RegionRegistry

logger = logging.getLogger(__name__)

MERGED_OSM = "combined.osm.pbf"   # the derived merge, not a region source


def _loaded_slugs(catalog: Catalog, input_dir: str) -> dict[str, str]:
    """Map each region-source file on disk to its catalog slug (by flattened name)."""
    try:
        files = {f for f in os.listdir(input_dir)
                 if f.endswith(".osm.pbf") and f != MERGED_OSM}
    except FileNotFoundError:
        return {}
    by_flat = {e.slug.replace("/", "-"): e.slug for e in catalog.all()}
    out: dict[str, str] = {}
    for f in files:
        slug = by_flat.get(f[: -len(".osm.pbf")])
        if slug:
            out[f] = slug
        else:
            logger.warning("geo seed: on-disk %s matches no catalog slug; ignoring", f)
    return out


def seed_registry(registry: RegionRegistry, catalog: Catalog, input_dir: str,
                  base_slug: str, base_transit_source: str | None) -> None:
    """Ensure every on-disk region is a ready registry entry (idempotent)."""
    now = datetime.now(timezone.utc).isoformat()
    for slug in _loaded_slugs(catalog, input_dir).values():
        existing = registry.get(slug)
        if existing is not None and existing.state == STATE_READY:
            continue   # already recorded ready — leave it (keeps last_used_at etc.)
        entry = catalog.get(slug)
        if entry is None:
            continue
        is_base = slug == base_slug
        registry.put(Region(
            slug=slug, display_name=entry.display_name, bbox=entry.bbox,
            has_transit=is_base and base_transit_source is not None,
            state=STATE_READY, added_at=now))
        logger.info("geo seed: recorded loaded region %s", slug)
