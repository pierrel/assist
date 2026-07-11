"""Unit tests for seed_registry — reconciling the registry with on-disk regions."""
import json

from assist.geo.catalog import CATALOG_FILE, Catalog
from assist.geo.model import STATE_FAILED, STATE_READY
from assist.geo.registry import RegionRegistry
from assist.geo.seed import MERGED_OSM, seed_registry


def _catalog(tmp_path, entries):
    (tmp_path / CATALOG_FILE).write_text(json.dumps(entries))
    return Catalog(str(tmp_path))


def _entry(slug, name, bbox):
    return {"slug": slug, "display_name": name, "bbox": bbox,
            "url": f"https://download.geofabrik.de/{slug.replace('/', '-')}.osm.pbf"}


def _touch_inputs(tmp_path, *filenames):
    inp = tmp_path / "input"
    inp.mkdir(exist_ok=True)
    for f in filenames:
        (inp / f).write_bytes(b"pbf")
    return str(inp)


def test_seed_records_on_disk_regions_ready(tmp_path):
    cat = _catalog(tmp_path, [
        _entry("norcal", "Northern California", [-124.5, 36.0, -119.0, 42.1]),
        _entry("us/rhode-island", "Rhode Island", [-71.9, 41.0, -71.0, 42.0])])
    # nested slug flattens to us-rhode-island.osm.pbf; the merged file is ignored
    inp = _touch_inputs(tmp_path, "norcal.osm.pbf", "us-rhode-island.osm.pbf", MERGED_OSM)
    reg = RegionRegistry(str(tmp_path))
    seed_registry(reg, cat, inp, base_slug="norcal", base_transit_source="511:RG")
    got = {r.slug: r for r in reg.all()}
    assert set(got) == {"norcal", "us/rhode-island"}     # merged file NOT a region
    assert got["norcal"].state == STATE_READY and got["norcal"].has_transit is True
    assert got["us/rhode-island"].has_transit is False   # only the base gets transit
    assert got["us/rhode-island"].display_name == "Rhode Island"


def test_seed_is_idempotent_and_preserves_ready(tmp_path):
    cat = _catalog(tmp_path, [_entry("norcal", "Northern California", [-124.5, 36.0, -119.0, 42.1])])
    inp = _touch_inputs(tmp_path, "norcal.osm.pbf")
    reg = RegionRegistry(str(tmp_path))
    seed_registry(reg, cat, inp, "norcal", "511:RG")
    reg.touch("norcal")                                  # a usage stamp
    seed_registry(reg, cat, inp, "norcal", "511:RG")     # re-run
    assert reg.get("norcal").last_used_at is not None    # not clobbered


def test_seed_normalizes_a_failed_ondisk_region_to_ready(tmp_path):
    # a region whose file IS on disk but whose registry entry is failed → the data is
    # really there, so normalize to ready
    cat = _catalog(tmp_path, [_entry("norcal", "Northern California", [-124.5, 36.0, -119.0, 42.1])])
    inp = _touch_inputs(tmp_path, "norcal.osm.pbf")
    reg = RegionRegistry(str(tmp_path))
    from assist.geo.model import Region
    reg.put(Region(slug="norcal", display_name="Northern California",
                   bbox=(-124.5, 36.0, -119.0, 42.1), state=STATE_FAILED))
    seed_registry(reg, cat, inp, "norcal", "511:RG")
    assert reg.get("norcal").state == STATE_READY


def test_seed_ignores_ondisk_file_with_no_catalog_slug(tmp_path):
    cat = _catalog(tmp_path, [_entry("norcal", "Northern California", [-124.5, 36.0, -119.0, 42.1])])
    inp = _touch_inputs(tmp_path, "norcal.osm.pbf", "mystery-region.osm.pbf")
    reg = RegionRegistry(str(tmp_path))
    seed_registry(reg, cat, inp, "norcal", "511:RG")
    assert {r.slug for r in reg.all()} == {"norcal"}     # mystery ignored (not in catalog)
