"""Egress requests + grants — one store, one projection, one client map.

Design: docs/2026-07-21-egress-approval-hitl.org. A record is keyed
``tid:host:port`` (grants are THREAD-scoped — Pierre's review; two threads
hold independent grants for the same host) and walks a small state machine:
``pending`` → ``approved`` (with a duration) or ``declined``. The
``declined`` state is load-bearing: a re-request of a declined key gets a
corrective tool return instead of a fresh card.

Two files live in the ``approvals/`` subdir — the ONLY thing the proxy
container sees (read-only mount; proposal state, including agent free text,
never reaches the proxy's parser):

- ``approved-hosts.json`` — a dumb projection regenerated WHOLE on every
  mutation (one writer: this store, under its lock; one reader: the proxy).
  Entry duration is EITHER an aware-UTC ISO ``expires_at`` (the 1-hour
  grant) OR the literal ``"revoked-only"`` (always-allow-for-this-thread) —
  anything else is skipped by the proxy, fail-closed, so an accidental null
  can never mean "permanent".
- ``client-map.json`` — egress-network IP → tid, written by SandboxManager
  at sandbox start (see ``client_map.py``); the proxy requires a grant's
  ``origin_tid`` to match the connecting client's tid.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

from assist.keyed_store import KeyedJsonStore

logger = logging.getLogger(__name__)

REQUESTS_FILE = "egress-requests.json"
APPROVALS_SUBDIR = "approvals"
PROJECTION_FILE = "approved-hosts.json"
REVOKED_ONLY = "revoked-only"
GRANT_TTL = timedelta(hours=1)


def request_key(tid: str, host: str, port: int) -> str:
    return f"{tid}:{host}:{port}"


@dataclass(frozen=True)
class EgressRequest:
    host: str
    port: int
    task: str
    origin_tid: str
    created_at: str = ""
    state: str = "pending"          # pending | approved | declined
    expires_at: str = ""            # ISO or REVOKED_ONLY when approved

    @property
    def key(self) -> str:
        return request_key(self.origin_tid, self.host, self.port)

    def to_dict(self) -> dict:
        return {"host": self.host, "port": self.port, "task": self.task,
                "origin_tid": self.origin_tid, "created_at": self.created_at,
                "state": self.state, "expires_at": self.expires_at}

    @staticmethod
    def from_dict(d: dict) -> "EgressRequest | None":
        try:
            host, port, tid = str(d["host"]), int(d["port"]), str(d["origin_tid"])
            if not (host and tid and 0 < port < 65536):
                return None
            return EgressRequest(
                host=host, port=port, task=str(d.get("task", "")),
                origin_tid=tid, created_at=str(d.get("created_at", "")),
                state=str(d.get("state", "pending")),
                expires_at=str(d.get("expires_at", "")))
        except Exception:
            logger.warning("egress: malformed request record skipped: %.120r", d)
            return None


def _grant_live(rec: EgressRequest, now: datetime) -> bool:
    """Fail-closed liveness: an approved grant is live iff its duration
    marker is the revoked-only literal or a still-future aware ISO stamp."""
    if rec.state != "approved":
        return False
    if rec.expires_at == REVOKED_ONLY:
        return True
    try:
        exp = datetime.fromisoformat(rec.expires_at)
        return exp.tzinfo is not None and exp > now
    except Exception:
        return False


class EgressStore(KeyedJsonStore[EgressRequest]):
    """Sole owner of the request file AND the approvals projection: every
    locked mutation regenerates the projection before releasing the lock, so
    the two can never drift. Expired grants are lazily pruned on mutation."""

    FILENAME = REQUESTS_FILE

    def __init__(self, egress_dir: str):
        os.makedirs(os.path.join(egress_dir, APPROVALS_SUBDIR), exist_ok=True)
        super().__init__(egress_dir)
        self._projection = os.path.join(
            egress_dir, APPROVALS_SUBDIR, PROJECTION_FILE)

    def _key(self, rec: EgressRequest) -> str:
        return rec.key

    def _to_dict(self, rec: EgressRequest) -> dict:
        return rec.to_dict()

    def _from_dict(self, d: dict) -> EgressRequest | None:
        return EgressRequest.from_dict(d)

    # --- internals (lock held) ---------------------------------------------
    def _mutate(self, recs: dict[str, EgressRequest]) -> None:
        """Prune expired grants, persist, regenerate the projection whole."""
        now = datetime.now(timezone.utc)
        live = {k: r for k, r in recs.items()
                if not (r.state == "approved" and not _grant_live(r, now))}
        self._write(live)
        import json
        tmp = f"{self._projection}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump({k: {"host": r.host, "port": r.port,
                           "origin_tid": r.origin_tid,
                           "expires_at": r.expires_at}
                       for k, r in live.items() if r.state == "approved"}, f)
        os.replace(tmp, self._projection)

    # --- API ----------------------------------------------------------------
    def add_pending(self, rec: EgressRequest) -> EgressRequest:
        with self._lock:
            recs = self._load()
            recs[rec.key] = rec
            self._mutate(recs)
            return rec

    def resolve(self, key: str, decision: str) -> EgressRequest | None:
        """Approve ("hour" | "always") or decline a pending request. Returns
        the updated record, or None when the key is unknown/not pending."""
        with self._lock:
            recs = self._load()
            rec = recs.get(key)
            if rec is None or rec.state != "pending":
                return None
            if decision == "hour":
                exp = (datetime.now(timezone.utc) + GRANT_TTL).isoformat()
                rec = replace(rec, state="approved", expires_at=exp)
            elif decision == "always":
                rec = replace(rec, state="approved", expires_at=REVOKED_ONLY)
            elif decision == "decline":
                rec = replace(rec, state="declined", expires_at="")
            else:
                return None
            recs[key] = rec
            self._mutate(recs)
            return rec

    def revoke(self, key: str) -> bool:
        """Remove a grant (user revoke on /egress, or the agent's
        remove_allowed_host — privilege reduction either way)."""
        with self._lock:
            recs = self._load()
            rec = recs.get(key)
            if rec is None or rec.state != "approved":
                return False
            del recs[key]
            self._mutate(recs)
            return True

    def remove_thread(self, tid: str) -> int:
        """Thread deletion: a grant never outlives its scope."""
        with self._lock:
            recs = self._load()
            kept = {k: r for k, r in recs.items() if r.origin_tid != tid}
            n = len(recs) - len(kept)
            if n:
                self._mutate(kept)
            return n

    def for_thread(self, tid: str) -> list[EgressRequest]:
        now = datetime.now(timezone.utc)
        with self._lock:
            return [r for r in self._load().values()
                    if r.origin_tid == tid
                    and (r.state != "approved" or _grant_live(r, now))]

    def live_grants(self) -> list[EgressRequest]:
        now = datetime.now(timezone.utc)
        with self._lock:
            return [r for r in self._load().values() if _grant_live(r, now)]
