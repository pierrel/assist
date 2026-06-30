"""Tests for the travel tool (assist/tools.py) — MOTIS geocode + multimodal plan.

CPU/no-model and no live service: HTTP is mocked against MOTIS's real shapes
(/api/v1/geocode -> [{name,lat,lon}], /api/v1/plan -> {direct:[...]} for
car/bike/walk, {itineraries:[...]} for transit). The agent-driven behavior is an
eval, not here.
"""
from unittest.mock import patch

import pytest

from assist import tools


class _Resp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


_GEO = {"civic": {"name": "Civic Center", "lat": "37.779", "lon": "-122.414"},
        "ferry": {"name": "Ferry Building", "lat": "37.795", "lon": "-122.392"}}
# MOTIS /plan direct: per direct mode, duration(s) + a leg distance(m).
_DIRECT = {"CAR": (1320, 14000.0), "BIKE": (2880, 13000.0), "WALK": (9600, 13000.0)}


def _fake_get(url, params=None, timeout=None, **kw):
    params = params or {}
    if "/search" in url:  # Nominatim geocoder (the default path)
        q = params["q"].lower()
        key = "civic" if "civic" in q else ("ferry" if "ferry" in q else None)
        if not key:
            return _Resp([])
        g = _GEO[key]  # Nominatim shape: display_name, no `name` -> exercises the fallback
        return _Resp([{"display_name": g["name"], "lat": g["lat"], "lon": g["lon"]}])
    if "/api/v1/geocode" in url:  # MOTIS built-in geocoder (fallback path)
        t = params["text"].lower()
        key = "civic" if "civic" in t else ("ferry" if "ferry" in t else None)
        return _Resp([_GEO[key]] if key else [])
    if "/api/v1/plan" in url:
        if "directModes" in params:
            secs, dist = _DIRECT[params["directModes"]]
            return _Resp({"direct": [{"duration": secs, "legs": [{"distance": dist}]}]})
        if "transitModes" in params:
            return _Resp({"itineraries": [{"duration": 2100}, {"duration": 2600}]})
    raise AssertionError(f"unexpected GET {url} {params}")


@pytest.fixture
def routing_env(monkeypatch):
    # Production default: Nominatim geocodes, MOTIS routes.
    monkeypatch.setenv("ASSIST_ROUTING_URL", "http://motis")
    monkeypatch.setenv("ASSIST_GEOCODER_URL", "http://nominatim")


class TestFormat:
    def test_duration(self):
        assert tools._fmt_duration(1320) == "22 min"
        assert tools._fmt_duration(9600) == "2 h 40 min"

    def test_distance(self):
        assert tools._fmt_distance_m(14000) == "8.7 mi"
        assert tools._fmt_distance_m(80) == "262 ft"


class TestTravel:
    def test_full_summary(self, routing_env):
        with patch.object(tools.requests, "get", _fake_get):
            out = tools.travel("civic center", "ferry building")
        assert 'from "Civic Center" to "Ferry Building"' in out
        assert "- Car: 22 min, 8.7 mi" in out
        assert "- Bike: 48 min, 8.1 mi" in out
        assert "- Walk: 2 h 40 min, 8.1 mi" in out
        assert "- Transit: 35 min" in out  # fastest of the two itineraries

    def test_passes_coords_not_names_to_plan(self, routing_env):
        seen = []
        def cap(url, params=None, **kw):
            if "/api/v1/plan" in url:
                seen.append(params)
            return _fake_get(url, params=params, **kw)
        with patch.object(tools.requests, "get", cap):
            tools.travel("civic center", "ferry building")
        # every plan call addresses coords, never the place name
        assert all("," in p["fromPlace"] and "civic" not in str(p).lower() for p in seen)

    def test_routing_unset_is_unavailable(self, monkeypatch):
        monkeypatch.delenv("ASSIST_ROUTING_URL", raising=False)
        monkeypatch.delenv("ASSIST_GEOCODER_URL", raising=False)
        assert "unavailable" in tools.travel("a", "b").lower()

    def test_geocoder_set_but_routing_unset_is_unavailable(self, monkeypatch):
        # Routing is required; with Nominatim, geocode would otherwise succeed and
        # every mode show "unavailable". Fail fast with the standard message, no HTTP.
        monkeypatch.setenv("ASSIST_GEOCODER_URL", "http://nominatim")
        monkeypatch.delenv("ASSIST_ROUTING_URL", raising=False)
        called = []
        with patch.object(tools.requests, "get",
                          lambda *a, **k: called.append(1) or _Resp([])):
            out = tools.travel("civic center", "ferry building")
        assert "unavailable" in out.lower()
        assert not called  # bailed before any HTTP

    def test_geocode_uses_nominatim_when_set(self, routing_env):
        seen = []
        def cap(url, params=None, **kw):
            seen.append(url)
            return _fake_get(url, params=params, **kw)
        with patch.object(tools.requests, "get", cap):
            tools.travel("civic center", "ferry building")
        assert any("/search" in u for u in seen)          # geocoded via Nominatim
        assert not any("/api/v1/geocode" in u for u in seen)  # not MOTIS's geocoder

    def test_geocode_falls_back_to_motis_when_unset(self, monkeypatch):
        monkeypatch.setenv("ASSIST_ROUTING_URL", "http://motis")
        monkeypatch.delenv("ASSIST_GEOCODER_URL", raising=False)
        seen = []
        def cap(url, params=None, **kw):
            seen.append(url)
            return _fake_get(url, params=params, **kw)
        with patch.object(tools.requests, "get", cap):
            out = tools.travel("civic center", "ferry building")
        assert any("/api/v1/geocode" in u for u in seen)   # MOTIS geocoder fallback
        assert not any("/search" in u for u in seen)
        assert "- Car: 22 min, 8.7 mi" in out            # full summary still works

    def test_service_down_is_unavailable_not_not_found(self, routing_env):
        # A geocoder/service outage must yield the "unavailable" message, NOT a
        # misleading "couldn't find that place" (service-down != no-match).
        def boom(url, params=None, **kw):
            raise ConnectionError("MOTIS down")
        with patch.object(tools.requests, "get", boom):
            out = tools.travel("civic center", "ferry building")
        assert "unavailable" in out.lower() and "couldn't find" not in out.lower()

    def test_place_not_found_asks_to_clarify(self, routing_env):
        with patch.object(tools.requests, "get", _fake_get):
            out = tools.travel("nowhere-xyz", "ferry building")
        assert "couldn't find" in out.lower() and "nowhere-xyz" in out

    def test_mode_with_no_route_is_unavailable(self, routing_env):
        # A direct mode that returns no `direct` itinerary shows as unavailable.
        def no_car(url, params=None, **kw):
            if "/api/v1/plan" in url and params.get("directModes") == "CAR":
                return _Resp({"direct": []})
            return _fake_get(url, params=params, **kw)
        with patch.object(tools.requests, "get", no_car):
            out = tools.travel("civic center", "ferry building")
        assert "- Car: unavailable" in out and "- Bike: 48 min" in out

    def test_wrong_shape_response_does_not_raise(self, routing_env):
        # A 200 with an unexpected JSON shape must not crash: a non-list geocode
        # -> "unavailable" (backend problem, not "couldn't find"); a non-dict plan
        # -> that mode "unavailable". Never an AttributeError into the agent loop.
        with patch.object(tools.requests, "get",
                          lambda *a, **k: _Resp({"error": "boom"})):  # geocode not a list
            assert "unavailable" in tools.travel("civic", "ferry").lower()

        def wrong_plan(url, params=None, **kw):
            if "/api/v1/plan" in url:
                return _Resp(["not", "a", "dict"])
            return _fake_get(url, params=params, **kw)
        with patch.object(tools.requests, "get", wrong_plan):
            out = tools.travel("civic center", "ferry building")  # must not raise
        assert "- Car: unavailable" in out

    def test_malformed_itinerary_does_not_raise(self, routing_env):
        # Module contract: never raise into the agent loop. A direct itinerary
        # missing `duration` / a transit itinerary missing it -> "unavailable".
        def malformed(url, params=None, **kw):
            if "/api/v1/plan" in url:
                if "directModes" in params:
                    return _Resp({"direct": [{"legs": [{"distance": 100.0}]}]})  # no duration
                if "transitModes" in params:
                    return _Resp({"itineraries": [{}]})  # no duration
            return _fake_get(url, params=params, **kw)
        with patch.object(tools.requests, "get", malformed):
            out = tools.travel("civic center", "ferry building")  # must not raise
        assert "- Car: unavailable" in out and "- Transit: unavailable" in out

    def test_transit_no_coverage_is_unavailable(self, routing_env):
        def no_transit(url, params=None, **kw):
            if "/api/v1/plan" in url and "transitModes" in params:
                return _Resp({"itineraries": []})
            return _fake_get(url, params=params, **kw)
        with patch.object(tools.requests, "get", no_transit):
            out = tools.travel("civic center", "ferry building")
        assert "- Transit: unavailable" in out and "- Car: 22 min" in out


def test_geocode_passes_through_coord_string():
    """A bare "lat,lon" (the "from here" origin) resolves to those coords WITHOUT a
    geocode call; a real place name does not match and falls through to geocoding."""
    # passthrough: exact coords + the synthetic "your location" name (so a real
    # geocode — which would return a different name — provably didn't run)
    assert tools._parse_coord_string("37.7749,-122.4194") == {
        "lat": 37.7749, "lon": -122.4194, "name": "your location"}
    assert tools._geocode("37.7749, -122.4194") == {
        "lat": 37.7749, "lon": -122.4194, "name": "your location"}
    # real names / non-coords → None (fall through to the geocoder)
    for name in ["Ferry Building", "San Francisco, CA", "Building 7, suite 3", "", "1,2,3"]:
        assert tools._parse_coord_string(name) is None
    # out-of-range numbers are not coords
    assert tools._parse_coord_string("200,200") is None
