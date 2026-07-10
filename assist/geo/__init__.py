"""Geo regions: the loaded-region registry, the downloadable-region catalog, and
(later) the provisioner that imports them.

Split by volatility (see docs/2026-07-10-geo-download-on-demand.org):
- ``model``    — the ``Region`` (a loaded region's operational state) and
  ``CatalogEntry`` (a downloadable region from the pinned Geofabrik snapshot) records.
- ``registry`` — ``RegionRegistry``, the SOLE reader+writer of ``regions.json``
  (what IS loaded + its state). A global single-file atomic-RMW store.
- ``catalog``  — a side-effect-free reader over the pinned catalog snapshot (the
  download allowlist + the picker/resolve source). The snapshot is BUILT elsewhere
  (the weekly refresh), never on the request path.
"""
