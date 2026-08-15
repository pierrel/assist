"""The context rider — per-MESSAGE time context a client attaches to one turn.

Distinct from ``AgentSpec`` (per-agent, static): the rider changes every message, so
it rides the per-INVOCATION ``configurable`` run config (the same mechanism emacsos
uses for ``PhoneContext``), NOT the spec.  The web path passes it per turn via
``ThreadManager.get(..., configurable={CONTEXT_RIDER_KEY: rider})``, which builds a
fresh ``Thread`` each turn — so the rider is always current.  A *reused*, long-lived
``Thread`` merges ``configurable`` into its persistent ``runconfig`` at construction,
so such a client must refresh the rider every turn (not set it once) or later turns
see a stale one.  Two consumers, one source:

- the model gets a rendered time line (``ContextRiderMiddleware``, injected
  ephemerally per turn — never checkpointed) so it can reason "you asked this
  morning…";
- deterministic consumers read the value (e.g. the sandbox ``TZ`` for ``date``).

Every field is OPTIONAL — no rider, or an empty one, reproduces prior behavior.
Browser coordinates remain available to the web-admission boundary for private
last-known-location storage, but never render into model context; see
docs/2026-08-15-location-context-tool.org.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

CONTEXT_RIDER_KEY = "context_rider"


@dataclass(frozen=True, slots=True)
class ContextRider:
    sent_at: datetime | None = None   # tz-aware instant the client stamped at send
    tz: str | None = None             # IANA zone, e.g. "America/Los_Angeles"
    lat: float | None = None
    lon: float | None = None
    place_label: str | None = None    # legacy client field; never model-visible

    def __post_init__(self):
        # Validate at the boundary (no network/subprocess/lock — ZoneInfo may do a
        # one-time cached tzdata read) — a bad value should fail here, not silently
        # mislead a consumer deep in a turn.
        if self.tz is not None:
            ZoneInfo(self.tz)  # raises on an unknown zone
        if self.sent_at is not None and self.sent_at.tzinfo is None:
            raise ValueError("ContextRider.sent_at must be timezone-aware")
        if self.lat is not None and not (-90.0 <= self.lat <= 90.0):
            raise ValueError(f"latitude out of range: {self.lat}")
        if self.lon is not None and not (-180.0 <= self.lon <= 180.0):
            raise ValueError(f"longitude out of range: {self.lon}")

    def prose_line(self) -> str | None:
        """A single human-readable time line for the model, or None."""
        parts = []
        when = self._when()
        if when:
            parts.append(f"sent {when}")
        if not parts:
            return None
        return "[Message context: " + "; ".join(parts) + ".]"

    def _when(self) -> str | None:
        if self.sent_at is None:
            return None
        dt = self.sent_at.astimezone(ZoneInfo(self.tz)) if self.tz else self.sent_at
        hour12 = dt.hour % 12 or 12   # avoid %-d/%-I (glibc-only strftime flags)
        stamp = f"{dt:%A, %B} {dt.day}, {dt.year} at {hour12}:{dt.minute:02d} {dt:%p}"
        return f"{stamp} ({self.tz})" if self.tz else stamp
