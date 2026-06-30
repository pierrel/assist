"""The schedule record — a structured, diffable cadence (NOT an opaque cron string).

Structured fields are what make relative edits patch cleanly ("fire at 5a" sets only
``hour``) and what let the small model fill semantically-named, range-checkable args.
The human-readable label and ``next_fire_at`` are derived by ``cadence.py`` from these
fields — never supplied by the model — so a mislabel can't diverge from what fires.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Cadence:
    """When a schedule fires, as orthogonal fields. Two mutually-exclusive modes
    (validated in cadence.py):

    - clock mode: ``minute`` (+ optional ``hour``; ``hour=None`` = every hour) on the
      days in ``weekdays`` (``None`` = every day). Covers daily / weekly /
      specific-weekdays / hourly.
    - interval mode: ``every_n_minutes`` — fires at clock-aligned multiples
      (minutes-since-midnight % N == 0), optionally filtered by ``weekdays``.
    """
    minute: int = 0
    hour: int | None = None
    weekdays: tuple[int, ...] | None = None   # 0=Mon … 6=Sun; None = every day
    every_n_minutes: int | None = None

    def to_dict(self) -> dict:
        return {"minute": self.minute, "hour": self.hour,
                "weekdays": list(self.weekdays) if self.weekdays is not None else None,
                "every_n_minutes": self.every_n_minutes}

    @classmethod
    def from_dict(cls, d: dict) -> "Cadence":
        wd = d.get("weekdays")
        return cls(minute=d.get("minute", 0), hour=d.get("hour"),
                   weekdays=tuple(wd) if wd is not None else None,
                   every_n_minutes=d.get("every_n_minutes"))


@dataclass(frozen=True)
class Schedule:
    """One schedule bound to one thread. ``next_fire_at`` is an absolute UTC ISO
    instant computed by the cadence engine; ``tz`` is the zone the cadence is read in
    (so "7am" means 7am local)."""
    id: str
    thread_id: str
    prompt: str
    cadence: Cadence
    tz: str
    enabled: bool = True
    next_fire_at: str | None = None   # ISO-8601 UTC
    created_at: str | None = None     # ISO-8601 UTC

    def with_next_fire(self, iso_utc: str | None) -> "Schedule":
        return replace(self, next_fire_at=iso_utc)

    def to_dict(self) -> dict:
        return {"id": self.id, "thread_id": self.thread_id, "prompt": self.prompt,
                "cadence": self.cadence.to_dict(), "tz": self.tz,
                "enabled": self.enabled, "next_fire_at": self.next_fire_at,
                "created_at": self.created_at}

    @classmethod
    def from_dict(cls, d: dict) -> "Schedule":
        return cls(id=d["id"], thread_id=d["thread_id"], prompt=d["prompt"],
                   cadence=Cadence.from_dict(d["cadence"]), tz=d["tz"],
                   enabled=d.get("enabled", True), next_fire_at=d.get("next_fire_at"),
                   created_at=d.get("created_at"))
