"""Pending download proposals — the HITL gate's durable record.

A proposal is what ``propose_region_download`` writes and what the user approves:
the agent can only ever *propose*; nothing downloads until a human approves the
recorded slug (the approve action fires the Provisioner with THIS record's slug).
The *region* is swap-proof — approve fires the recorded key, and ``display_name`` /
``bbox`` / the PBF URL re-derive from the catalog for that slug — so the agent can't
substitute region B for the region A the user saw. (The record's ``user_request`` is
replaceable until approval; propose refuses to overwrite once a region is importing,
and increment 5's approve can pin ``created_at`` if needed.) Dedicated record, NOT the
send_reply interrupt: that interrupt is only armed on triage turns and can't fire on a
normal chat turn (design doc B2).

One ``proposals.json`` per deploy in the geo dir (same single-file atomic-RMW shape
as ``regions.json``), keyed by slug. A proposal also carries what the completion
message needs later (origin thread + the user's request), so delivery survives a
web restart (C1); the Provisioner deletes it once the completion is delivered.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from assist.geo.model import BBox, _parse_bbox

logger = logging.getLogger(__name__)

PROPOSALS_FILE = "proposals.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Proposal:
    """One pending (or approved-and-awaiting-completion-delivery) download proposal."""
    slug: str
    display_name: str
    bbox: BBox
    size_bytes: int | None          # from the propose-time HEAD; None = unknown
    origin_tid: str                 # thread to notify when the import completes
    user_request: str               # the ask that triggered the proposal (A1: code carries intent)
    created_at: str

    def to_dict(self) -> dict:
        return {"slug": self.slug, "display_name": self.display_name,
                "bbox": list(self.bbox), "size_bytes": self.size_bytes,
                "origin_tid": self.origin_tid, "user_request": self.user_request,
                "created_at": self.created_at}

    @classmethod
    def from_dict(cls, d: dict) -> "Proposal | None":
        slug = d.get("slug")
        bbox = _parse_bbox(d.get("bbox"))
        tid = d.get("origin_tid")
        if not isinstance(slug, str) or not slug or bbox is None or not isinstance(tid, str):
            logger.warning("geo: skipping malformed proposal record: %r", d)
            return None
        size = d.get("size_bytes")
        return cls(slug=slug,
                   display_name=d.get("display_name") if isinstance(d.get("display_name"), str)
                   and d.get("display_name") else slug,
                   bbox=bbox,
                   size_bytes=size if isinstance(size, int) and size >= 0 else None,
                   origin_tid=tid,
                   user_request=d.get("user_request") if isinstance(d.get("user_request"), str)
                   else "",
                   created_at=d.get("created_at") or _now_iso())


class ProposalStore:
    """Disk-backed proposal store; sole reader+writer of ``proposals.json``."""

    def __init__(self, geo_dir: str):
        self._path = os.path.join(geo_dir, PROPOSALS_FILE)
        self._lock = threading.Lock()

    def _read(self) -> dict[str, Proposal]:
        try:
            with open(self._path) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            logger.warning("geo: proposals.json is not an object; ignoring")
            return {}
        out: dict[str, Proposal] = {}
        for slug, d in data.items():
            p = Proposal.from_dict(d) if isinstance(d, dict) else None
            if p is not None:
                out[p.slug] = p
        return out

    def _write(self, props: dict[str, Proposal]) -> None:
        tmp = f"{self._path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump({s: p.to_dict() for s, p in props.items()}, f)
        os.replace(tmp, self._path)

    def all(self) -> list[Proposal]:
        with self._lock:
            return list(self._read().values())

    def get(self, slug: str) -> Proposal | None:
        with self._lock:
            return self._read().get(slug)

    def put(self, proposal: Proposal) -> Proposal:
        """Insert or replace (re-proposing the same region overwrites — one pending
        proposal per slug)."""
        with self._lock:
            props = self._read()
            props[proposal.slug] = proposal
            self._write(props)
            return proposal

    def remove(self, slug: str) -> None:
        with self._lock:
            props = self._read()
            if props.pop(slug, None) is not None:
                self._write(props)
