"""Unit tests for the geo tools (list_regions / find_regions / propose_region_download).

Real registry + catalog + proposal store over a tmp dir (no LLM; the size HEAD is
mocked): asserts the model-facing strings — coverage listing (ready only, transit
flag), name→exact-id resolution (smallest-first, already-loaded marker, cap+more), the
no-match clarify path, and the propose sink (is_allowed gate, size + degradation
warning, proposal recorded with the thread + request).
"""
import json
from unittest.mock import patch

from assist.geo.catalog import CATALOG_FILE, Catalog
from assist.geo.model import Region, STATE_IMPORTING, STATE_READY
from assist.geo.proposals import ProposalStore
from assist.geo.registry import RegionRegistry
from assist.geo.tools import geo_tools


def _region(slug, name, transit=False, state=STATE_READY):
    return Region(slug=slug, display_name=name, bbox=(-124.0, 32.0, -114.0, 42.0),
                  has_transit=transit, state=state)


def _catalog_entry(slug, name, bbox):
    return {"slug": slug, "display_name": name, "bbox": bbox,
            "url": f"https://download.geofabrik.de/{slug.replace('/', '-')}.osm.pbf"}


def _all_tools(tmp_path, regions=(), catalog_entries=()):
    reg = RegionRegistry(str(tmp_path))
    for r in regions:
        reg.put(r)
    (tmp_path / CATALOG_FILE).write_text(json.dumps(list(catalog_entries)))
    props = ProposalStore(str(tmp_path))
    return geo_tools(reg, Catalog(str(tmp_path)), props), reg, props


def _tools(tmp_path, regions=(), catalog_entries=()):
    (list_regions, find_regions, _), _, _ = _all_tools(tmp_path, regions, catalog_entries)
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


# --- propose_region_download (the HITL sink) -------------------------------------

_WA = [-124.8, 45.5, -116.9, 49.1]


class _Head:
    """A fake requests.head response."""
    def __init__(self, url, length="700000000"):
        self.url = url
        self.headers = {"Content-Length": length} if length else {}


def _propose(tmp_path, regions=(), tid="t1"):
    entries = [_catalog_entry("us/washington", "Washington", _WA)]
    (tools, reg, props) = _all_tools(tmp_path, regions=regions, catalog_entries=entries)
    _, _, propose = tools
    return propose, props


def test_propose_rejects_non_catalog_id(tmp_path):
    propose, props = _propose(tmp_path)
    out = propose("us/wa-typo", "directions in Tacoma")
    assert "not a downloadable region id" in out and "find_regions" in out
    assert props.all() == []                         # nothing recorded — the sink held


def test_propose_records_with_size_and_warning(tmp_path):
    propose, props = _propose(tmp_path)
    with patch("assist.geo.tools.requests.head",
               lambda url, **kw: _Head(url)), \
         patch("assist.geo.tools._thread_id", lambda: "t1"):
        out = propose("us/washington", "directions in Tacoma")
    assert "~700 MB" in out and "rebuilds the geocoder" in out
    assert "until the user approves" in out           # proposes, never downloads
    p = props.get("us/washington")
    assert p.origin_tid == "t1" and p.user_request == "directions in Tacoma"
    assert p.size_bytes == 700_000_000 and p.bbox == tuple(_WA)


def test_propose_already_loaded_short_circuits(tmp_path):
    loaded = [_region("us/washington", "Washington")]
    propose, props = _propose(tmp_path, regions=loaded)
    with patch("assist.geo.tools._thread_id", lambda: "t1"):
        out = propose("us/washington", "x")
    assert "already loaded" in out and props.all() == []


def test_propose_size_unknown_on_redirect(tmp_path):
    propose, props = _propose(tmp_path)
    # a redirect (302, no Content-Length since allow_redirects=False) → size unknown
    class _Redirect:
        headers = {}                      # no Content-Length on a redirect response
    with patch("assist.geo.tools.requests.head", lambda url, **kw: _Redirect()), \
         patch("assist.geo.tools._thread_id", lambda: "t1"):
        out = propose("us/washington", "x")
    assert "size unknown" in out
    assert props.get("us/washington").size_bytes is None


def test_propose_refused_while_importing(tmp_path):
    from assist.geo.model import STATE_IMPORTING
    importing = [_region("us/washington", "Washington", state=STATE_IMPORTING)]
    propose, props = _propose(tmp_path, regions=importing)
    with patch("assist.geo.tools._thread_id", lambda: "t1"):
        out = propose("us/washington", "x")
    assert "already downloading" in out and props.all() == []   # no overwrite of the pending record


def test_propose_without_thread_refuses(tmp_path):
    propose, props = _propose(tmp_path)
    with patch("assist.geo.tools._thread_id", lambda: None):
        out = propose("us/washington", "x")
    assert "no active thread" in out and props.all() == []


def test_propose_coerces_uuid_thread_id(tmp_path):
    # the run config can carry a UUID object (AgentHarness) — origin_tid must land as a
    # JSON-serializable string, or the proposal write crashes the tool
    import uuid
    propose, props = _propose(tmp_path)
    tid = uuid.uuid4()
    with patch("assist.geo.tools.requests.head", lambda url, **kw: _Head(url)), \
         patch("assist.geo.tools.get_config",
               lambda: {"configurable": {"thread_id": tid}}):
        out = propose("us/washington", "directions in Tacoma")
    p = props.get("us/washington")
    assert p is not None and p.origin_tid == str(tid) and isinstance(p.origin_tid, str)
    assert "~700 MB" in out
