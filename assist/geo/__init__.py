"""Geo regions: the loaded-region registry, the downloadable-region catalog, and
(later) the provisioner that imports them.

Split by volatility (see docs/2026-07-10-geo-download-on-demand.org):
- ``model``    — the ``Region`` (a loaded region's operational state) and
  ``CatalogEntry`` (a downloadable region from the pinned Geofabrik snapshot) records.
- ``registry`` — ``RegionRegistry``, the SOLE reader+writer of ``regions.json``
  (what IS loaded + its state). A global single-file atomic-RMW store.
- ``catalog``  — a side-effect-free reader over the pinned catalog snapshot (the
  download allowlist + the picker/resolve source), read-only on the request path.
- ``build_catalog`` — the write-time half of the catalog: fetches the Geofabrik
  index, derives bboxes, merges the transit overlay, and writes the pinned snapshot
  atomically. Run by the weekly refresh, never on the request path.
"""
