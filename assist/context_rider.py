"""The context rider — per-MESSAGE context a client attaches to one turn: WHEN and
WHERE the user's message was sent.

Distinct from ``AgentSpec`` (per-agent, static): the rider changes every message, so
it rides the per-INVOCATION ``configurable`` run config (the same mechanism emacsos
uses for ``PhoneContext``), NOT the spec.  The web path passes it per turn via
``ThreadManager.get(..., configurable={CONTEXT_RIDER_KEY: rider})``, which builds a
fresh ``Thread`` each turn — so the rider is always current.  A *reused*, long-lived
``Thread`` merges ``configurable`` into its persistent ``runconfig`` at construction,
so such a client must refresh the rider every turn (not set it once) or later turns
see a stale one.  Two consumers, one source:

- the model gets a rendered prose line (``ContextRiderMiddleware``, injected
  ephemerally per turn — never checkpointed) so it can reason "you asked this
  morning…";
- deterministic consumers read the value (e.g. the sandbox ``TZ`` for ``date``).

Every field is OPTIONAL — no rider, or an empty one, reproduces prior behavior
(server timezone, no location).  v1 wires TIME (sent_at + tz); the geo fields are
defined now (so the contract is stable) but only become a model/tool input as later
iterations land — see docs/2026-06-29-context-rider.org.
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
    place_label: str | None = None    # optional coarse human label ("downtown SF")

    def __post_init__(self):
        # Validate at the boundary (pure CPU, no I/O) — a bad value should fail
        # here, not silently mislead a consumer deep in a turn.
        if self.tz is not None:
            ZoneInfo(self.tz)  # raises on an unknown zone
        if self.sent_at is not None and self.sent_at.tzinfo is None:
            raise ValueError("ContextRider.sent_at must be timezone-aware")
        if self.lat is not None and not (-90.0 <= self.lat <= 90.0):
            raise ValueError(f"latitude out of range: {self.lat}")
        if self.lon is not None and not (-180.0 <= self.lon <= 180.0):
            raise ValueError(f"longitude out of range: {self.lon}")

    def prose_line(self) -> str | None:
        """A single human-readable context line for the model, or None if the rider
        carries nothing. Location is coarse (≈city-block) — this text is what the
        model sees; precise coords stay on the structured object for tools."""
        parts = []
        when = self._when()
        if when:
            parts.append(f"sent {when}")
        where = self._where()
        if where:
            parts.append(f"from {where}")
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

    def _where(self) -> str | None:
        if self.place_label:
            # place_label is the one free-text field and gets folded into the
            # SYSTEM message — collapse whitespace/newlines and cap the length so a
            # client can't smuggle instruction-like text or bloat the prompt.
            return " ".join(self.place_label.split())[:60]
        if self.lat is not None and self.lon is not None:
            return f"~{self.lat:.2f}, {self.lon:.2f}"  # ≈city-block; precise coords stay off the prose
        return None
