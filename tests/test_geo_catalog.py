"""Unit tests for Catalog — the read-only view of the pinned downloadable-region snapshot.

Covers the allowlist (is_allowed), resolve helpers (get/search/containing with
smallest-first ordering), malformed-entry tolerance, missing-file empty, and the mtime
cache picking up a rebuilt snapshot.
"""
import json
import os

from assist.geo.catalog import CATALOG_FILE, Catalog

# socal (small) nested inside california (large); washington elsewhere.
_SOCAL = {"slug": "socal", "display_name": "Southern California",
          "bbox": [-121.0, 32.0, -114.0, 35.5], "url": "https://download.geofabrik.de/socal.osm.pbf",
          "size_bytes": 698351616, "transit_feed": None}
_CALIF = {"slug": "us/california", "display_name": "California",
          "bbox": [-124.5, 32.0, -114.0, 42.1], "url": "https://download.geofabrik.de/california.osm.pbf",
          "size_bytes": 1200000000}
_WASH = {"slug": "us/washington", "display_name": "Washington",
         "bbox": [-124.8, 45.5, -116.9, 49.1], "url": "https://download.geofabrik.de/washington.osm.pbf"}


def _write(tmp_path, entries):
    (tmp_path / CATALOG_FILE).write_text(json.dumps(entries))


def test_get_and_is_allowed(tmp_path):
    _write(tmp_path, [_SOCAL, _WASH])
    cat = Catalog(str(tmp_path))
    assert cat.get("socal").url == "https://download.geofabrik.de/socal.osm.pbf"
    assert cat.is_allowed("socal") is True
    assert cat.is_allowed("us/washington") is True
    assert cat.is_allowed("evil/../slug") is False   # not a snapshot id → not fetchable
    assert cat.get("nope") is None


def test_missing_file_is_empty(tmp_path):
    cat = Catalog(str(tmp_path))
    assert cat.all() == [] and cat.is_allowed("socal") is False


def test_search_substring_case_insensitive_smallest_first(tmp_path):
    _write(tmp_path, [_CALIF, _SOCAL])
    cat = Catalog(str(tmp_path))
    # "california" is a substring of BOTH "Southern California" and "California" →
    # both match, smallest extent first (socal before the whole state)
    hits = cat.search("CALIFORNIA")
    assert [h.slug for h in hits] == ["socal", "us/california"]
    # "southern" hits only socal
    assert [h.slug for h in cat.search("southern")] == ["socal"]
    assert cat.search("  ") == []


def test_containing_point_smallest_first(tmp_path):
    _write(tmp_path, [_CALIF, _SOCAL, _WASH])
    cat = Catalog(str(tmp_path))
    # a point in LA is inside both socal and california → smallest (socal) first
    hits = cat.containing(-118.24, 34.05)
    assert [h.slug for h in hits] == ["socal", "us/california"]
    # a point in the ocean is in no bbox → empty (⇒ no download proposed)
    assert cat.containing(0.0, 0.0) == []


def test_malformed_entry_skipped(tmp_path):
    _write(tmp_path, [_SOCAL, {"slug": "bad"}, "junk", {"slug": "x", "url": "u", "bbox": [1, 2, 3]}])
    cat = Catalog(str(tmp_path))
    assert {e.slug for e in cat.all()} == {"socal"}


def test_mtime_cache_reloads_on_rebuild(tmp_path):
    _write(tmp_path, [_SOCAL])
    cat = Catalog(str(tmp_path))
    assert {e.slug for e in cat.all()} == {"socal"}
    # rebuild the snapshot with a new region; bump mtime so the cache reloads
    _write(tmp_path, [_SOCAL, _WASH])
    p = str(tmp_path / CATALOG_FILE)
    os.utime(p, (os.path.getmtime(p) + 10, os.path.getmtime(p) + 10))
    assert {e.slug for e in cat.all()} == {"socal", "us/washington"}
