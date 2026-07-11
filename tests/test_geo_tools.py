"""Unit tests for the read-only geo tools (list_regions / find_regions).

Real registry + catalog over a tmp dir (no LLM, no network): asserts the model-facing
strings — coverage listing (ready only, transit flag), name→exact-id resolution
(smallest-first, already-loaded marker, cap+more), and the no-match clarify path.
"""
import json

from assist.geo.catalog import CATALOG_FILE, Catalog
from assist.geo.model import Region, STATE_IMPORTING, STATE_READY
from assist.geo.registry import RegionRegistry
from assist.geo.tools import geo_tools


def _region(slug, name, transit=False, state=STATE_READY):
    return Region(slug=slug, display_name=name, bbox=(-124.0, 32.0, -114.0, 42.0),
                  has_transit=transit, state=state)


def _catalog_entry(slug, name, bbox):
    return {"slug": slug, "display_name": name, "bbox": bbox,
            "url": f"https://download.geofabrik.de/{slug.replace('/', '-')}.osm.pbf"}


def _tools(tmp_path, regions=(), catalog_entries=()):
    reg = RegionRegistry(str(tmp_path))
    for r in regions:
        reg.put(r)
    (tmp_path / CATALOG_FILE).write_text(json.dumps(list(catalog_entries)))
    cat = Catalog(str(tmp_path))
    list_regions, find_regions = geo_tools(reg, cat)
    return list_regions, find_regions


def test_list_regions_empty(tmp_path):
    list_regions, _ = _tools(tmp_path)
    assert "No regions are loaded" in list_regions()


def test_list_regions_shows_ready_with_transit_flag(tmp_path):
    regions = [_region("norcal", "Northern California", transit=True),
               _region("socal", "Southern California", transit=False),
               _region("pnw", "Pacific NW", state=STATE_IMPORTING)]   # not ready → hidden
    list_regions, _ = _tools(tmp_path, regions=regions)
    out = list_regions()
    assert "Northern California (with transit)" in out
    assert "Southern California (no transit)" in out
    assert "Pacific NW" not in out            # importing is not yet loaded


def test_find_regions_resolves_name_to_exact_id_smallest_first(tmp_path):
    entries = [_catalog_entry("us/california", "California", [-124.5, 32.0, -114.0, 42.1]),
               _catalog_entry("socal", "Southern California", [-121.0, 32.0, -114.0, 35.5])]
    _, find_regions = _tools(tmp_path, catalog_entries=entries)
    out = find_regions("california")
    # both match "california"; the smaller (socal) is listed first, each with its id
    assert out.index("[id: socal]") < out.index("[id: us/california]")


def test_find_regions_marks_already_loaded(tmp_path):
    entries = [_catalog_entry("us/washington", "Washington", [-124.8, 45.5, -116.9, 49.1])]
    regions = [_region("us/washington", "Washington")]
    _, find_regions = _tools(tmp_path, regions=regions, catalog_entries=entries)
    out = find_regions("washington")
    assert "[id: us/washington] — already loaded" in out


def test_find_regions_failed_import_is_not_already_loaded(tmp_path):
    # a region whose import FAILED (or is still importing) must NOT read as "already
    # loaded" — else the agent tells the user they're covered when they're not
    from assist.geo.model import STATE_FAILED
    entries = [_catalog_entry("us/washington", "Washington", [-124.8, 45.5, -116.9, 49.1])]
    regions = [_region("us/washington", "Washington", state=STATE_FAILED)]
    _, find_regions = _tools(tmp_path, regions=regions, catalog_entries=entries)
    out = find_regions("washington")
    assert "[id: us/washington]" in out and "already loaded" not in out


def test_find_regions_no_match_asks_to_clarify(tmp_path):
    entries = [_catalog_entry("us/washington", "Washington", [-124.8, 45.5, -116.9, 49.1])]
    _, find_regions = _tools(tmp_path, catalog_entries=entries)
    out = find_regions("Atlantis")
    assert "No downloadable region matches" in out and "which US state" in out


def test_find_regions_caps_candidates(tmp_path):
    entries = [_catalog_entry(f"region-{i}", f"Test Region {i}",
                              [-124.0, 32.0, -114.0 + i * 0.01, 42.0]) for i in range(12)]
    _, find_regions = _tools(tmp_path, catalog_entries=entries)
    out = find_regions("test region")
    assert out.count("[id: region-") == 8            # capped at _MAX_CANDIDATES
    assert "+4 more" in out
