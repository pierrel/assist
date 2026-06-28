"""Tests for the travel tool (assist/tools.py) — geocode + route + transit.

CPU/no-model and no live services: HTTP is mocked. The agent-driven behavior
(does the model call travel for "how long to X") is an eval, not here.
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


# Canned backend responses keyed by endpoint.
_GEO = {"home": [{"lat": "37.80", "lon": "-122.41", "display_name": "Home, SF"}],
        "ferry": [{"lat": "37.79", "lon": "-122.39", "display_name": "Ferry Building"}]}


def _fake_get(url, **kw):
    if "/search" in url:               # Nominatim geocode
        q = kw["params"]["q"].lower()
        key = "home" if "home" in q else ("ferry" if "ferry" in q else None)
        return _Resp(_GEO.get(key, []))
    if "/journeys" in url:             # Navitia transit
        return _Resp({"journeys": [{"duration": 2100}]})
    raise AssertionError(f"unexpected GET {url}")


def _fake_post(url, **kw):
    if "/route" in url:                # Valhalla
        costing = kw["json"]["costing"]
        secs = {"auto": 1320, "bicycle": 2880, "pedestrian": 9600}[costing]
        km = {"auto": 14.0, "bicycle": 13.0, "pedestrian": 13.0}[costing]
        return _Resp({"trip": {"summary": {"time": secs, "length": km}}})
    raise AssertionError(f"unexpected POST {url}")


@pytest.fixture
def travel_env(monkeypatch):
    monkeypatch.setenv("ASSIST_GEOCODER_URL", "http://geo")
    monkeypatch.setenv("ASSIST_ROUTING_URL", "http://valhalla")
    monkeypatch.setenv("ASSIST_TRANSIT_URL", "http://navitia")
    monkeypatch.setenv("ASSIST_TRANSIT_TOKEN", "tok")


class TestFormat:
    def test_duration(self):
        assert tools._fmt_duration(1320) == "22 min"
        assert tools._fmt_duration(9600) == "2 h 40 min"

    def test_distance(self):
        assert tools._fmt_distance_m(14000) == "14.0 km"
        assert tools._fmt_distance_m(80) == "80 m"


class TestTravel:
    def test_full_summary(self, travel_env):
        with patch.object(tools.requests, "get", _fake_get), \
             patch.object(tools.requests, "post", _fake_post):
            out = tools.travel("home", "ferry building")
        assert 'from "Home, SF" to "Ferry Building"' in out
        assert "- Car: 22 min, 14.0 km" in out
        assert "- Bike: 48 min, 13.0 km" in out
        assert "- Walk: 2 h 40 min, 13.0 km" in out
        assert "- Transit: 35 min" in out

    def test_transit_coords_only(self, travel_env):
        # The hosted transit call must send COORDS, never the place name.
        seen = {}
        def cap_get(url, **kw):
            if "/journeys" in url:
                seen["params"] = kw["params"]
            return _fake_get(url, **kw)
        with patch.object(tools.requests, "get", cap_get), \
             patch.object(tools.requests, "post", _fake_post):
            tools.travel("home", "ferry building")
        assert seen["params"]["from"] == "-122.41;37.8"
        assert "home" not in str(seen["params"]).lower()

    def test_geocoder_unset_is_unavailable(self, monkeypatch):
        monkeypatch.delenv("ASSIST_GEOCODER_URL", raising=False)
        assert "unavailable" in tools.travel("a", "b").lower()

    def test_place_not_found_asks_to_clarify(self, travel_env):
        with patch.object(tools.requests, "get", _fake_get):
            out = tools.travel("nowhere-xyz", "ferry building")
        assert "couldn't find" in out.lower() and "nowhere-xyz" in out

    def test_transit_unavailable_without_token(self, travel_env, monkeypatch):
        monkeypatch.delenv("ASSIST_TRANSIT_TOKEN", raising=False)
        with patch.object(tools.requests, "get", _fake_get), \
             patch.object(tools.requests, "post", _fake_post):
            out = tools.travel("home", "ferry building")
        assert "- Transit: unavailable" in out
        assert "- Car: 22 min" in out  # static modes still work

    def test_routing_down_mode_unavailable(self, travel_env):
        def boom_post(url, **kw):
            raise RuntimeError("valhalla down")
        with patch.object(tools.requests, "get", _fake_get), \
             patch.object(tools.requests, "post", boom_post):
            out = tools.travel("home", "ferry building")
        assert "- Car: unavailable" in out
        assert "- Transit: 35 min" in out  # transit independent of valhalla
