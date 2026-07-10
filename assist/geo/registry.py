"""``RegionRegistry`` — the SOLE reader+writer of ``regions.json``, the one
unambiguous answer to "which regions are loaded, and in what state".

Unlike schedules (``PerThreadJsonStore``, one file per thread), the region registry
is GLOBAL: a single ``<geo_dir>/regions.json`` under the travel-infra dir. So it gets
its own tiny store rather than the per-thread base — same disciplines (one lock
serializing every read-modify-write, atomic tmp+rename, missing/corrupt → empty), a
dict keyed by slug (slugs are unique; a nested id like ``us/washington`` is a plain
JSON key, never a filesystem path here).

The provisioning scripts write NONE of this file — they do the heavy data work only;
the Provisioner writes ``importing`` then ``ready``/``failed`` through THIS class, so
the record format lives in exactly one place (see the design doc's B1/registry
resolution).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import replace
from datetime import datetime, timezone
from typing import Callable

from assist.geo.model import Region, STATE_FAILED, STATE_IMPORTING, STATE_READY

logger = logging.getLogger(__name__)

REGISTRY_FILE = "regions.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RegionRegistry:
    """Disk-backed loaded-region store. ``geo_dir`` is the travel-infra dir that also
    holds the catalog snapshot and the OSM/GTFS inputs."""

    def __init__(self, geo_dir: str):
        self._path = os.path.join(geo_dir, REGISTRY_FILE)
        self._lock = threading.Lock()

    def _read(self) -> dict[str, Region]:
        """Read the registry (callers hold the lock). Missing/corrupt → empty; a single
        malformed record is skipped, never fatal."""
        try:
            with open(self._path) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            logger.warning("geo: regions.json is not an object; ignoring")
            return {}
        out: dict[str, Region] = {}
        for slug, d in data.items():
            region = Region.from_dict(d) if isinstance(d, dict) else None
            if region is not None:
                out[region.slug] = region
        return out

    def _write(self, regions: dict[str, Region]) -> None:
        """Atomically persist the registry (callers hold the lock)."""
        tmp = f"{self._path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump({slug: r.to_dict() for slug, r in regions.items()}, f)
        os.replace(tmp, self._path)

    # --- reads ------------------------------------------------------------------
    def all(self) -> list[Region]:
        with self._lock:
            return list(self._read().values())

    def get(self, slug: str) -> Region | None:
        with self._lock:
            return self._read().get(slug)

    def is_loaded(self, slug: str) -> bool:
        """A region counts as loaded once its import is ready (not merely enqueued)."""
        r = self.get(slug)
        return r is not None and r.state == STATE_READY

    # --- writes (the provisioner + the query path use these) --------------------
    def put(self, region: Region) -> Region:
        """Insert or replace a region wholesale (e.g. the provisioner writing the
        initial ``importing`` record with catalog-sourced name/bbox)."""
        with self._lock:
            regions = self._read()
            regions[region.slug] = region
            self._write(regions)
            return region

    def update(self, slug: str, fn: Callable[[Region], Region]) -> Region | None:
        """Read-modify-write one record under the lock. ``fn`` maps the found region to
        its replacement. Returns None (no write) if the slug isn't present."""
        with self._lock:
            regions = self._read()
            r = regions.get(slug)
            if r is None:
                return None
            regions[slug] = fn(r)
            self._write(regions)
            return regions[slug]

    def set_state(self, slug: str, state: str) -> Region | None:
        return self.update(slug, lambda r: r.with_state(state))

    def touch(self, slug: str) -> None:
        """Stamp ``last_used_at`` when a query resolves inside this region (S1 — powers
        the ``/geo`` page's 'used N days ago' so 'delete unused' is informed). Best
        effort: a slug not in the registry is a no-op."""
        self.update(slug, lambda r: replace(r, last_used_at=_now_iso()))

    def remove(self, slug: str) -> None:
        with self._lock:
            regions = self._read()
            if regions.pop(slug, None) is not None:
                self._write(regions)

    def reconcile(self) -> list[str]:
        """Startup sweep: a record stuck in ``importing`` has no live job (the web
        restart killed the blocking subprocess), so flip it to ``failed`` — nothing
        half-built is served because the scripts validate-before-swap. Returns the
        slugs reconciled."""
        with self._lock:
            regions = self._read()
            stuck = [s for s, r in regions.items() if r.state == STATE_IMPORTING]
            for s in stuck:
                regions[s] = regions[s].with_state(STATE_FAILED)
            if stuck:
                self._write(regions)
            return stuck
