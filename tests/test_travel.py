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
    if "/api/v1/geocode" in url:
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
    monkeypatch.setenv("ASSIST_ROUTING_URL", "http://motis")


class TestFormat:
    def test_duration(self):
        assert tools._fmt_duration(1320) == "22 min"
        assert tools._fmt_duration(9600) == "2 h 40 min"

    def test_distance(self):
        assert tools._fmt_distance_m(14000) == "14.0 km"
        assert tools._fmt_distance_m(80) == "80 m"


class TestTravel:
    def test_full_summary(self, routing_env):
        with patch.object(tools.requests, "get", _fake_get):
            out = tools.travel("civic center", "ferry building")
        assert 'from "Civic Center" to "Ferry Building"' in out
        assert "- Car: 22 min, 14.0 km" in out
        assert "- Bike: 48 min, 13.0 km" in out
        assert "- Walk: 2 h 40 min, 13.0 km" in out
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
        assert "unavailable" in tools.travel("a", "b").lower()

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
