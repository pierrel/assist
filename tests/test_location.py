"""Private current-location handle and durable last-known browser record."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from assist.location import (CURRENT_LOCATION, LOCATION_CONTEXT_KEY, LocationSnapshot,
                             LocationStore, configured_location, get_location,
                             resolve_location_handle)


def test_store_keeps_latest_valid_record_and_enforces_freshness(tmp_path):
    store = LocationStore(str(tmp_path))
    now = datetime(2026, 8, 15, 12, tzinfo=timezone.utc)
    first = store.record(37.7, -122.4, observed_at=now - timedelta(minutes=1))
    later = store.record(37.8, -122.5, observed_at=now)
    assert first.lat == 37.7 and later.lat == 37.8
    assert store.recent(now=now) == later
    assert store.recent(now=now + timedelta(hours=24, seconds=1)) is None
    assert Path(store.path).stat().st_mode & 0o777 == 0o600


def test_older_record_cannot_overwrite_newer_one(tmp_path):
    store = LocationStore(str(tmp_path))
    now = datetime(2026, 8, 15, 12, tzinfo=timezone.utc)
    store.record(37.8, -122.5, observed_at=now)
    returned = store.record(37.7, -122.4, observed_at=now - timedelta(seconds=1))
    assert returned.lat == 37.8
    assert store.recent(now=now).lat == 37.8


def test_opaque_tool_reveals_no_coordinates_or_address(monkeypatch):
    snapshot = LocationSnapshot(37.7749, -122.4194, datetime.now(timezone.utc))
    monkeypatch.setattr("assist.location.get_config",
                        lambda: {"configurable": {LOCATION_CONTEXT_KEY: snapshot}})
    result = get_location()
    assert CURRENT_LOCATION in result
    assert "37.7749" not in result and "-122.4194" not in result
    assert resolve_location_handle(CURRENT_LOCATION) == {
        "lat": 37.7749, "lon": -122.4194, "name": "your location"}


def test_opaque_handle_is_unavailable_without_turn_snapshot(monkeypatch):
    monkeypatch.setattr("assist.location.get_config", lambda: {"configurable": {}})
    assert "No current" in get_location()
    with pytest.raises(ValueError):
        resolve_location_handle(CURRENT_LOCATION)
    assert configured_location() is None


def test_opaque_handle_rejects_a_stale_turn_snapshot(monkeypatch):
    stale = LocationSnapshot(37.7749, -122.4194,
                             datetime.now(timezone.utc) - timedelta(hours=24, seconds=1))
    monkeypatch.setattr("assist.location.get_config",
                        lambda: {"configurable": {LOCATION_CONTEXT_KEY: stale}})
    assert "No current" in get_location()
    with pytest.raises(ValueError):
        resolve_location_handle(CURRENT_LOCATION)
