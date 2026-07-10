"""Unit tests for RegionRegistry — the sole reader+writer of regions.json.

Drives the global single-file store against a tmp dir (no infra, no network): roundtrip,
state transitions, touch, reconcile, malformed-record tolerance, and the missing-file
empty case.
"""
import json

from assist.geo.model import (
    Region, STATE_FAILED, STATE_IMPORTING, STATE_READY)
from assist.geo.registry import REGISTRY_FILE, RegionRegistry


def _region(slug="socal", state=STATE_READY, **kw):
    return Region(slug=slug, display_name=kw.get("display_name", "Southern California"),
                  bbox=kw.get("bbox", (-121.0, 32.0, -114.0, 35.5)),
                  has_transit=kw.get("has_transit", False), state=state,
                  added_at=kw.get("added_at", "2026-07-10T00:00:00+00:00"))


def test_put_get_roundtrip(tmp_path):
    reg = RegionRegistry(str(tmp_path))
    reg.put(_region())
    got = reg.get("socal")
    assert got is not None and got.slug == "socal"
    assert got.bbox == (-121.0, 32.0, -114.0, 35.5) and got.display_name == "Southern California"
    # persisted to the expected file
    assert (tmp_path / REGISTRY_FILE).exists()


def test_missing_file_reads_empty(tmp_path):
    reg = RegionRegistry(str(tmp_path))
    assert reg.all() == [] and reg.get("socal") is None
    assert reg.is_loaded("socal") is False


def test_nested_slug_is_a_plain_key(tmp_path):
    reg = RegionRegistry(str(tmp_path))
    reg.put(_region(slug="us/washington", display_name="Washington"))
    assert reg.get("us/washington").display_name == "Washington"
    # a slash in the slug never creates a path — it's a JSON key
    assert (tmp_path / REGISTRY_FILE).exists()
    assert not (tmp_path / "us").exists()


def test_set_state_and_is_loaded(tmp_path):
    reg = RegionRegistry(str(tmp_path))
    reg.put(_region(state=STATE_IMPORTING))
    assert reg.is_loaded("socal") is False       # importing is not loaded
    reg.set_state("socal", STATE_READY)
    assert reg.is_loaded("socal") is True
    assert reg.get("socal").state == STATE_READY


def test_update_missing_slug_is_none_no_write(tmp_path):
    reg = RegionRegistry(str(tmp_path))
    assert reg.update("nope", lambda r: r) is None


def test_touch_stamps_last_used(tmp_path):
    reg = RegionRegistry(str(tmp_path))
    reg.put(_region())
    assert reg.get("socal").last_used_at is None
    reg.touch("socal")
    assert reg.get("socal").last_used_at is not None
    # a slug not present is a silent no-op
    reg.touch("absent")


def test_remove(tmp_path):
    reg = RegionRegistry(str(tmp_path))
    reg.put(_region())
    reg.remove("socal")
    assert reg.get("socal") is None
    reg.remove("socal")   # idempotent


def test_reconcile_flips_stuck_importing_to_failed(tmp_path):
    reg = RegionRegistry(str(tmp_path))
    reg.put(_region(slug="socal", state=STATE_READY))
    reg.put(_region(slug="pnw", state=STATE_IMPORTING, display_name="Pacific NW"))
    reconciled = reg.reconcile()
    assert reconciled == ["pnw"]
    assert reg.get("pnw").state == STATE_FAILED
    assert reg.get("socal").state == STATE_READY    # ready untouched


def test_malformed_record_skipped_not_fatal(tmp_path):
    # one good, one missing bbox, one non-dict — only the good one survives the read
    (tmp_path / REGISTRY_FILE).write_text(json.dumps({
        "socal": _region().to_dict(),
        "broken": {"slug": "broken"},          # no bbox
        "junk": "not-a-record",
    }))
    reg = RegionRegistry(str(tmp_path))
    slugs = {r.slug for r in reg.all()}
    assert slugs == {"socal"}


def test_corrupt_json_reads_empty(tmp_path):
    (tmp_path / REGISTRY_FILE).write_text("{not valid json")
    reg = RegionRegistry(str(tmp_path))
    assert reg.all() == []
