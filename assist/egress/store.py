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

import json
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
# Declined records exist for the don't-re-ask corrective; a decline's
# steering value decays with the conversation, so once ANNOUNCED
# (dispatched) they are pruned after this window — the store stays bounded
# by live traffic, not history.
DECLINED_RETENTION = timedelta(days=7)


def request_key(tid: str, host: str, port: int) -> str:
    return f"{tid}:{host}:{port}"


def approvals_dir_is_safe(egress_dir: str) -> bool:
    """The self-approval guard, applied by EVERY reader of
    ``ASSIST_EGRESS_APPROVALS_DIR`` — the web wiring (tools/routes) AND the
    sandbox layer (proxy mount + client-map writes). An approvals dir under
    the thread root would sit inside some thread's rw ``/workspace`` bind
    mount, letting a prompt-injected sandbox write its own grants
    (self-approval with no tool, no card, no user). A guard applied on only
    one path is theater — the exploit rides whichever reader skipped it.
    Residual, named: the comparison uses the env-derived thread root; a
    ThreadManager rooted programmatically without the env var is out of this
    guard's scope."""
    root = os.path.realpath(
        os.getenv("ASSIST_THREADS_DIR", "/tmp/assist_threads"))
    try:
        return os.path.commonpath(
            [os.path.realpath(egress_dir), root]) != root
    except ValueError:      # different drives (never on this deploy) — refuse
        return False


@dataclass(frozen=True)
class EgressRequest:
    host: str
    port: int
    task: str
    origin_tid: str
    created_at: str = ""
    state: str = "pending"          # pending | approved | declined
    expires_at: str = ""            # ISO or REVOKED_ONLY when approved
    # Consumed by the ONE resolution turn that fires when the thread's
    # pending set empties: only never-dispatched resolutions are enumerated,
    # so an old decline or a long-lived always-grant is not re-announced (or
    # its task re-run) on every later batch.
    dispatched: bool = False

    @property
    def key(self) -> str:
        return request_key(self.origin_tid, self.host, self.port)

    def to_dict(self) -> dict:
        return {"host": self.host, "port": self.port, "task": self.task,
                "origin_tid": self.origin_tid, "created_at": self.created_at,
                "state": self.state, "expires_at": self.expires_at,
                "dispatched": self.dispatched}

    @staticmethod
    def from_dict(d: dict) -> "EgressRequest | None":
        try:
            host, port, tid = str(d["host"]), int(d["port"]), str(d["origin_tid"])
            if not (host and tid and 0 < port < 65536):
                return None
            state = d.get("state", "pending")
            if state not in ("pending", "approved", "declined"):
                return None     # unknown state: skip, don't persist garbage
            return EgressRequest(
                host=host, port=port, task=str(d.get("task", "")),
                origin_tid=tid, created_at=str(d.get("created_at", "")),
                state=state,
                expires_at=str(d.get("expires_at", "")),
                # strict identity check: bool("false") is True — only a real
                # JSON true may mark a record announced/prunable
                dispatched=(d.get("dispatched") is True))
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
        """Prune expired grants + stale announced declines, persist,
        regenerate the projection whole."""
        now = datetime.now(timezone.utc)

        def _stale_decline(r: EgressRequest) -> bool:
            if r.state != "declined" or not r.dispatched:
                return False
            try:
                return (now - datetime.fromisoformat(r.created_at)
                        ) > DECLINED_RETENTION
            except Exception:
                return True     # unparseable stamp: prune rather than hoard
        live = {k: r for k, r in recs.items()
                if not (r.state == "approved" and not _grant_live(r, now))
                and not _stale_decline(r)}
        self._write(live)
        try:
            tmp = f"{self._projection}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                json.dump({k: {"host": r.host, "port": r.port,
                               "origin_tid": r.origin_tid,
                               "expires_at": r.expires_at}
                           for k, r in live.items() if r.state == "approved"}, f)
            os.replace(tmp, self._projection)
        except OSError:
            # Fail CLOSED, not stale: a projection the store can't rewrite
            # (disk full, permissions) could keep honoring a just-revoked
            # grant. Remove it — the proxy degrades to base-allowlist-only —
            # then surface the failure to the caller.
            try:
                os.remove(self._projection)
            except OSError:
                pass
            logger.error("egress projection write failed — removed so the "
                         "proxy fails closed to the base allowlist",
                         exc_info=True)
            raise

    # --- API ----------------------------------------------------------------
    PER_THREAD_PENDING_CAP = 3
    GLOBAL_PENDING_CAP = 10

    def add_pending(self, rec: EgressRequest) -> str | None:
        """Insert a pending request, counting BOTH caps under the same lock
        as the insert (two parallel tool calls can otherwise land 4+ pending
        — the threat model's fatigue bound). Returns an error reason string
        when refused, None on success."""
        with self._lock:
            recs = self._load()
            pending = [r for r in recs.values() if r.state == "pending"]
            if sum(1 for r in pending if r.origin_tid == rec.origin_tid) \
                    >= self.PER_THREAD_PENDING_CAP:
                return "thread-cap"
            if len(pending) >= self.GLOBAL_PENDING_CAP:
                return "global-cap"
            recs[rec.key] = rec
            self._mutate(recs)
            return None

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

    def take_undispatched(self, tid: str) -> list[EgressRequest]:
        """Collect the thread's never-dispatched resolutions (approved live +
        declined) and mark them dispatched, atomically — the ONE resolution
        turn enumerates exactly this batch and no later batch repeats it."""
        now = datetime.now(timezone.utc)
        with self._lock:
            recs = self._load()
            # Consumed even if the process dies before the turn runs — at-most-
            # once by design (re-running an approved grant's task on retry
            # would be worse than losing one announcement; the BackgroundTask-
            # turn precedent already accepts crash loss).
            batch = [r for r in recs.values()
                     if r.origin_tid == tid and not r.dispatched
                     and (r.state == "declined"
                          or (r.state == "approved" and _grant_live(r, now)))]
            for r in batch:
                recs[r.key] = replace(r, dispatched=True)
            if batch:
                self._mutate(recs)
            return batch

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


def remaining_lifetime(rec: EgressRequest) -> str:
    """One wording for a grant's remaining lifetime (the /egress page and the
    list_allowed_hosts tool both show it — a shared helper so the copies
    can't diverge)."""
    if rec.expires_at == REVOKED_ONLY:
        return "until revoked or the thread is deleted"
    try:
        mins = max(0, int((datetime.fromisoformat(rec.expires_at)
                           - datetime.now(timezone.utc)).total_seconds() // 60))
        return f"~{mins} min left"
    except Exception:
        return "expiring"


def resolution_prompt(batch: list[EgressRequest]) -> str | None:
    """The resolution turn's prompt over ONE take_undispatched batch — pure,
    shared with the eval so the eval can't silently pin a stale shape."""
    lines = []
    for r in batch:
        if r.state == "approved":
            lines.append(f'- {r.host}:{r.port} APPROVED. Your recorded task: '
                         f'"{r.task}"')
        elif r.state == "declined":
            lines.append(f"- {r.host}:{r.port} DECLINED — do not re-ask; "
                         "proceed without it.")
    if not lines:
        return None
    return ("[Egress requests resolved] The user has resolved this thread's "
            "network access requests:\n" + "\n".join(lines)
            + "\nFor approved hosts, carry out the recorded task now — if "
            "the work already succeeded, just confirm the result. "
            "Acknowledge any declined hosts in your answer.")
