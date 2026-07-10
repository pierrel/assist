"""Unit tests for build_catalog — index → pinned snapshot (no network; fixture index).

Covers: entry transformation (bbox from MultiPolygon, display-name derivation for the
name==id entries), the T2 URL gate (off-host/non-https entries DROPPED), the transit
overlay merge (separate file, missing feed → None), the near-empty refuse
(validate-before-swap), atomic write readable by Catalog, and the index-URL pin.
"""
import json

import pytest

from assist.geo.build_catalog import (
    ALLOWED_PBF_HOST, build, fetch_index, TRANSIT_OVERLAY_FILE)
from assist.geo.catalog import Catalog


def _feature(fid, name=None, pbf=None, coords=None):
    return {"type": "Feature",
            "properties": {"id": fid, "name": name or fid,
                           "urls": {"pbf": pbf or f"https://{ALLOWED_PBF_HOST}/{fid}.osm.pbf"}},
            "geometry": {"type": "MultiPolygon",
                         "coordinates": coords or [[[[-121.0, 32.0], [-114.0, 32.0],
                                                      [-114.0, 35.5], [-121.0, 35.5],
                                                      [-121.0, 32.0]]]]}}


def _index(features):
    return {"type": "FeatureCollection", "features": features}


def _bulk(n, start=0):
    """Filler features so builds clear the near-empty floor."""
    return [_feature(f"region-{i}") for i in range(start, start + n)]


def test_build_transforms_and_is_readable_by_catalog(tmp_path):
    feats = [_feature("socal", name="Southern California"),
             _feature("us/washington", name="us/washington",   # name==id → derived
                      coords=[[[[-124.8, 45.5], [-116.9, 45.5], [-116.9, 49.1],
                                [-124.8, 49.1], [-124.8, 45.5]]]])] + _bulk(120)
    count = build(str(tmp_path), _index(feats))
    assert count == 122
    cat = Catalog(str(tmp_path))
    socal = cat.get("socal")
    assert socal.display_name == "Southern California"
    assert socal.bbox == (-121.0, 32.0, -114.0, 35.5)
    assert socal.url == f"https://{ALLOWED_PBF_HOST}/socal.osm.pbf"
    assert socal.size_bytes is None                      # sizes are NOT pre-cached (T4)
    # name==id → prettified from the id's last segment
    assert cat.get("us/washington").display_name == "Washington"


def test_off_host_and_non_https_urls_dropped(tmp_path):
    feats = [_feature("evil", pbf="https://evil.example.com/x.osm.pbf"),
             _feature("plain", pbf=f"http://{ALLOWED_PBF_HOST}/plain.osm.pbf"),
             _feature("good")] + _bulk(120)
    build(str(tmp_path), _index(feats))
    cat = Catalog(str(tmp_path))
    assert cat.is_allowed("good") is True
    assert cat.is_allowed("evil") is False               # T2: never enters the allowlist
    assert cat.is_allowed("plain") is False


def test_transit_overlay_merged_from_separate_file(tmp_path):
    (tmp_path / TRANSIT_OVERLAY_FILE).write_text(json.dumps({"norcal": "511:RG"}))
    build(str(tmp_path), _index([_feature("norcal", name="Northern California"),
                                 _feature("socal")] + _bulk(120)))
    cat = Catalog(str(tmp_path))
    assert cat.get("norcal").transit_feed == "511:RG"
    assert cat.get("socal").transit_feed is None


def test_near_empty_result_refuses_to_overwrite(tmp_path):
    # seed a good snapshot, then try to build from a truncated index
    build(str(tmp_path), _index(_bulk(120)))
    before = Catalog(str(tmp_path)).all()
    with pytest.raises(ValueError, match="refusing"):
        build(str(tmp_path), _index([_feature("only-one")]))
    assert len(Catalog(str(tmp_path)).all()) == len(before)   # old snapshot intact


def test_malformed_features_skipped(tmp_path):
    feats = [_feature("good"),
             {"type": "Feature", "properties": {}, "geometry": {}},         # no id/pbf
             {"type": "Feature",                                            # no geometry
              "properties": {"id": "nogeo",
                             "urls": {"pbf": f"https://{ALLOWED_PBF_HOST}/n.osm.pbf"}},
              "geometry": {"type": "Point", "coordinates": [1, 2]}}] + _bulk(120)
    build(str(tmp_path), _index(feats))
    cat = Catalog(str(tmp_path))
    assert cat.is_allowed("good") and not cat.is_allowed("nogeo")


def test_fetch_index_pins_url():
    with pytest.raises(ValueError, match="must be https"):
        fetch_index("http://download.geofabrik.de/index-v1.json")
    with pytest.raises(ValueError, match="must be https"):
        fetch_index("https://example.com/index-v1.json")
