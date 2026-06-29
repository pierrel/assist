"""Tests for the directions tool (assist/tools.py) — turn-by-turn over MOTIS /plan.

CPU/no-model, no live service: HTTP is mocked against MOTIS's real shapes. Street
geometry is built with a local polyline ENCODER (the inverse of the tool's decoder)
so turns are deterministic — "go east then go north" must read as a left turn.
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


def _encode(coords, precision=7):
    """Google-encode [(lat, lon), ...] — inverse of tools._decode_polyline."""
    factor = 10 ** precision
    out = []
    plat = plon = 0
    for lat, lon in coords:
        ilat, ilon = round(lat * factor), round(lon * factor)
        for d in (ilat - plat, ilon - plon):
            d = ~(d << 1) if d < 0 else (d << 1)
            while d >= 0x20:
                out.append(chr((0x20 | (d & 0x1f)) + 63))
                d >>= 5
            out.append(chr(d + 63))
        plat, plon = ilat, ilon
    return "".join(out)


def _step(name, coords, meters):
    return {"streetName": name, "distance": meters,
            "polyline": {"points": _encode(coords), "precision": 7}}


# "A Street" runs EAST (lon increasing), "B Street" runs NORTH (lat increasing):
# the boundary is a ~90° left turn.
_EAST = [(37.78, -122.42), (37.78, -122.41)]
_NORTH = [(37.78, -122.41), (37.79, -122.41)]
_STREET_PLAN = {"direct": [{"duration": 600, "legs": [{"distance": 1200.0, "steps": [
    _step("A Street", _EAST, 600.0),
    _step("B Street", _NORTH, 600.0),
]}]}]}
_TRANSIT_PLAN = {"itineraries": [{"duration": 1260, "legs": [
    {"mode": "WALK", "from": {"name": "Origin"}, "to": {"name": "Stop A"}, "duration": 180},
    {"mode": "BUS", "routeShortName": "9R", "headsign": "Downtown",
     "from": {"name": "Stop A"}, "to": {"name": "Stop B"},
     "intermediateStops": [{}, {}], "duration": 720},
    {"mode": "WALK", "from": {"name": "Stop B"}, "to": {"name": "Dest"}, "duration": 360},
]}]}


def _fake_get(url, params=None, timeout=None, **kw):
    params = params or {}
    if "/search" in url:  # Nominatim geocode: echo the query as the resolved name
        return _Resp([{"lat": "37.78", "lon": "-122.42", "display_name": params["q"]}])
    if "/api/v1/plan" in url:
        return _Resp(_TRANSIT_PLAN if "transitModes" in params else _STREET_PLAN)
    raise AssertionError(f"unexpected GET {url} {params}")


@pytest.fixture
def routing_env(monkeypatch):
    monkeypatch.setenv("ASSIST_ROUTING_URL", "http://motis")
    monkeypatch.setenv("ASSIST_GEOCODER_URL", "http://nominatim")


class TestHelpers:
    def test_decode_polyline_known_vector(self):
        # The canonical Google example (precision 5).
        pts = tools._decode_polyline("_p~iF~ps|U_ulLnnqC_mqNvxq`@", 5)
        assert len(pts) == 3
        assert round(pts[0][0], 5) == 38.5 and round(pts[0][1], 5) == -120.2
        assert round(pts[2][0], 3) == 43.252

    def test_decode_polyline_bad_input_returns_empty(self):
        assert tools._decode_polyline(None, 7) == []
        assert tools._decode_polyline(123, 7) == []

    def test_turn_phrase_confidence_gated(self):
        assert tools._turn_phrase(0, 90) == "Turn right onto"
        assert tools._turn_phrase(90, 0) == "Turn left onto"
        assert tools._turn_phrase(0, 10) == "Continue onto"   # below threshold
        assert tools._turn_phrase(0, 178) == "Make a U-turn onto"
        assert tools._turn_phrase(None, 90) == "Continue onto"  # unknown -> neutral
        assert tools._turn_phrase(90, None) == "Continue onto"


class TestDirections:
    def test_street_directions_has_turns_and_miles(self, routing_env):
        with patch.object(tools.requests, "get", _fake_get):
            out = tools.directions("home", "the office", "car")
        assert out.startswith('Driving directions from "home" to "the office"')
        assert "1. Head onto A Street (0.4 mi)" in out
        assert "2. Turn left onto B Street (0.4 mi)" in out   # east -> north = left
        assert out.rstrip().endswith('Arrive at "the office"')

    def test_bike_uses_street_path(self, routing_env):
        with patch.object(tools.requests, "get", _fake_get):
            out = tools.directions("home", "the office", "bike")
        assert out.startswith("Biking directions")

    def test_transit_directions_narrative(self, routing_env):
        with patch.object(tools.requests, "get", _fake_get):
            out = tools.directions("home", "the office", "transit")
        assert out.startswith("Transit directions")
        assert "1. Walk to Stop A (3 min)" in out
        assert "2. Take the 9R bus toward Downtown to Stop B (3 stops)" in out
        assert '3. Walk to "the office" (6 min)' in out      # last walk -> dest name

    def test_mode_synonyms_normalize(self):
        assert tools._normalize_mode("Driving")[1] == "CAR"
        assert tools._normalize_mode("cycling")[1] == "BIKE"
        assert tools._normalize_mode("on foot")[1] == "WALK"
        assert tools._normalize_mode("subway")[0] == "transit"

    def test_unknown_mode_asks(self, routing_env):
        with patch.object(tools.requests, "get", _fake_get):
            out = tools.directions("a", "b", "teleport")
        assert "car, bike, walk, or transit" in out

    def test_routing_unset_is_unavailable(self, monkeypatch):
        monkeypatch.delenv("ASSIST_ROUTING_URL", raising=False)
        monkeypatch.delenv("ASSIST_GEOCODER_URL", raising=False)
        assert "unavailable" in tools.directions("a", "b", "car").lower()

    def test_no_route_is_reported(self, routing_env):
        def empty(url, params=None, **kw):
            if "/api/v1/plan" in url:
                return _Resp({"direct": []})
            return _fake_get(url, params=params, **kw)
        with patch.object(tools.requests, "get", empty):
            out = tools.directions("home", "office", "car")
        assert "couldn't find a driving route" in out.lower()

    def test_service_down_is_unavailable(self, routing_env):
        def boom(url, params=None, **kw):
            raise ConnectionError("MOTIS down")
        with patch.object(tools.requests, "get", boom):
            assert "unavailable" in tools.directions("home", "office", "transit").lower()

    def test_malformed_plan_does_not_raise(self, routing_env):
        def wrong(url, params=None, **kw):
            if "/api/v1/plan" in url:
                return _Resp(["not", "a", "dict"])
            return _fake_get(url, params=params, **kw)
        with patch.object(tools.requests, "get", wrong):
            out = tools.directions("home", "office", "car")  # must not raise
        assert "couldn't find" in out.lower()
