"""Private last-known location state and the model-facing opaque handle.

The web process records a browser fix before admitting a visible turn. A turn
receives the resulting snapshot through ``configurable``; tools can route from
``CURRENT_LOCATION`` without putting coordinates or an address in model context.
The store is deliberately outside per-thread workspaces and is never mounted in
a sandbox.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from langgraph.config import get_config

LOCATION_CONTEXT_KEY = "last_known_location"
CURRENT_LOCATION = "CURRENT_LOCATION"
_MAX_AGE = timedelta(hours=24)


@dataclass(frozen=True, slots=True)
class LocationSnapshot:
    """An exact browser fix that tools, but not the model prompt, may use."""

    lat: float
    lon: float
    observed_at: datetime

    def __post_init__(self) -> None:
        if not -90.0 <= self.lat <= 90.0:
            raise ValueError("latitude out of range")
        if not -180.0 <= self.lon <= 180.0:
            raise ValueError("longitude out of range")
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")


class LocationStore:
    """One atomically replaced global browser-location record for a web process."""

    def __init__(self, root: str) -> None:
        self.path = os.path.join(root, "last_location.json")
        self._lock = threading.Lock()

    def record(self, lat: float, lon: float, *, observed_at: datetime | None = None) -> LocationSnapshot:
        """Persist a valid fix; an older concurrent observation cannot overwrite newer."""
        snapshot = LocationSnapshot(float(lat), float(lon),
                                    observed_at or datetime.now(timezone.utc))
        with self._lock:
            existing = self._read()
            if existing is not None and existing.observed_at > snapshot.observed_at:
                return existing
            self._write(snapshot)
        return snapshot

    def recent(self, *, now: datetime | None = None) -> LocationSnapshot | None:
        """Return the record only while it is within the fixed freshness window."""
        now = now or datetime.now(timezone.utc)
        with self._lock:
            snapshot = self._read()
        if snapshot is None:
            return None
        age = now - snapshot.observed_at
        return snapshot if timedelta(0) <= age <= _MAX_AGE else None

    def _read(self) -> LocationSnapshot | None:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            return LocationSnapshot(float(data["lat"]), float(data["lon"]),
                                    datetime.fromisoformat(str(data["observed_at"])))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    def _write(self, snapshot: LocationSnapshot) -> None:
        directory = os.path.dirname(self.path)
        os.makedirs(directory, mode=0o700, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".last_location-", dir=directory, text=True)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"lat": snapshot.lat, "lon": snapshot.lon,
                           "observed_at": snapshot.observed_at.isoformat()}, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, self.path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise


def configured_location() -> LocationSnapshot | None:
    """Return this invocation's private location snapshot, never doing I/O."""
    value = ((get_config() or {}).get("configurable") or {}).get(LOCATION_CONTEXT_KEY)
    if not isinstance(value, LocationSnapshot):
        return None
    age = datetime.now(timezone.utc) - value.observed_at
    return value if timedelta(0) <= age <= _MAX_AGE else None


def get_location() -> str:
    """Check whether a recent browser location is available for travel or directions.

    When it is, pass CURRENT_LOCATION exactly as an origin/destination to
    travel or directions. This tool intentionally does not reveal
    coordinates, an address, or a neighborhood.
    """
    snapshot = configured_location()
    if snapshot is None:
        return ("No current or recent browser location is available for this turn. "
                "Ask the user to send a new message from the web app with location enabled.")
    age = max(0, int((datetime.now(timezone.utc) - snapshot.observed_at).total_seconds()))
    if age < 60:
        age_text = "just now"
    elif age < 3600:
        age_text = f"{age // 60} minutes ago"
    else:
        age_text = f"{age // 3600} hours ago"
    return (f"A recent browser location is available (observed {age_text}). Use "
            f"{CURRENT_LOCATION} exactly as the origin or destination in travel or directions; "
            "it is not an address.")


class LocationUnavailable(ValueError):
    """The opaque handle was requested without a fresh per-turn snapshot."""


def resolve_location_handle(place: str) -> dict | None:
    """Resolve the opaque handle to exact routing coordinates inside a tool."""
    if str(place).strip() != CURRENT_LOCATION:
        return None
    snapshot = configured_location()
    if snapshot is None:
        raise LocationUnavailable
    return {"lat": snapshot.lat, "lon": snapshot.lon, "name": "your location"}
