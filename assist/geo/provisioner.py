"""``Provisioner`` — runs approved region jobs off-loop and tells the agent when done.

Deliberately a THIN executor, not a ``Scheduler`` clone (design doc, cross-lens shift):
region jobs are event-driven — submitted exactly once, on the user's approval — so
there is no poll loop and no cron math. Serialization: ``_submit_lock`` makes the
whole check→claim→enqueue atomic (no double-enqueue from a double-clicked approve),
``max_workers=1`` runs one job at a time, and the scripts' shared flock excludes the
weekly refresh cron. Add/remove claim via the persisted ``state:importing``; transit
claims in memory only (it never changes serveability, so it must not risk a
reconcile→failed on a restart).

All web coupling is INJECTED callables (the ``Scheduler(store, dispatch, health_check)``
/ ``notify_tools(mark_urgent)`` precedent — this module never imports ``manage.web``):

- ``run_job(op, slug) -> bool`` — the heavy work; the web wires a blocking
  ``subprocess.run`` of the provisioning script (add / remove / transit). Bounded,
  validate-before-swap, blocks to completion (T5).
- ``on_complete(tid, message)`` — post a completion message to the originating thread
  (the web wires ``_scheduled_dispatch`` + ``_mark_urgent``). The message CARRIES the
  user's original request (A1): a bare "download finished" would make the small model
  re-derive intent from stale history. It is NOT an automatic re-run — the agent
  decides what to do with it.
- ``health_check() -> bool`` — is the LLM reachable. Completion delivery is held and
  retried while it isn't (D4): the undelivered flag lives in the REGISTRY
  (``completion_delivered``) and the request context in the PROPOSAL record, so
  delivery survives a web restart (C1). ``deliver_pending()`` re-attempts; the web
  calls it after startup reconcile and on a periodic tick.

Everything here runs on the executor thread or a caller's worker thread — nothing
blocks the asyncio event loop; ``submit`` is a registry write + a non-blocking
executor submit.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from typing import Callable

from assist.geo.model import (
    Region, STATE_FAILED, STATE_IMPORTING, STATE_READY)
from assist.geo.proposals import ProposalStore
from assist.geo.registry import RegionRegistry

logger = logging.getLogger(__name__)

OPS = ("add", "remove", "transit")


def _completion_message(display_name: str, user_request: str) -> str:
    ask = f' The user originally asked: "{user_request}".' if user_request else ""
    return (f"The {display_name} region finished downloading and is now loaded — "
            f"travel, directions, and address lookups work there.{ask} Pick that "
            "request back up now using the new region data.")


def _failure_message(display_name: str) -> str:
    return (f"The {display_name} region download failed, so the region was NOT added. "
            "Let the user know; they can ask to retry.")


class Provisioner:
    """One import at a time; states through the registry; completion via callbacks."""

    def __init__(self, registry: RegionRegistry, proposals: ProposalStore,
                 run_job: Callable[[str, str], bool],
                 on_complete: Callable[[str, str], None],
                 health_check: Callable[[], bool]):
        self._registry = registry
        self._proposals = proposals
        self._run_job = run_job
        self._on_complete = on_complete
        self._health_check = health_check
        self._executor = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix="geo-provisioner")
        # Makes the whole submit body (check → claim → enqueue) atomic, so two
        # concurrent approve POSTs can't both enqueue the same job. Held only for cheap
        # local-file I/O + a non-blocking executor submit — never loop-adjacent.
        self._submit_lock = threading.Lock()
        # Serializes completion delivery: _run's inline delivery vs a concurrent
        # deliver_pending() tick can't both dispatch the same completion.
        self._deliver_lock = threading.Lock()
        # In-memory transit-job claims. Transit never changes a region's serveability
        # (it stays READY), so — unlike add/remove — it must NOT move to importing:
        # a restart mid-transit would leave a fully-served region wrongly failed and
        # hidden from coverage. An in-memory claim dies with the process exactly when
        # the job does, so there is nothing to reconcile.
        self._transit_active: set[str] = set()

    # --- enqueue (called from a web worker thread; never blocks) -----------------
    def submit(self, op: str, slug: str) -> str:
        """Enqueue a region job. Returns a short human-readable status ("queued" /
        why-refused). For an ``add``, the proposal record must exist (the HITL gate:
        approve fires the RECORDED slug); its catalog metadata seeds the registry
        entry. The whole body is atomic under ``_submit_lock`` (no double-enqueue)."""
        if op not in OPS:
            return f"unknown operation: {op}"
        with self._submit_lock:
            current = self._registry.get(slug)
            if current is not None and current.state == STATE_IMPORTING:
                return f"{slug} already has a job running"
            if op == "transit":
                if current is None or current.state != STATE_READY:
                    return f"{slug} isn't loaded; can't add transit"
                if slug in self._transit_active:
                    return f"{slug} already has a transit job running"
                self._transit_active.add(slug)   # stays READY; in-memory dedup only
                self._executor.submit(self._run, op, slug, None)
                return "queued"
            if op == "add":
                p = self._proposals.get(slug)
                if p is None:
                    return f"no pending proposal for {slug}"
                if current is not None and current.state == STATE_READY:
                    self._proposals.remove(slug)
                    return f"{slug} is already loaded"
                self._registry.put(Region(
                    slug=slug, display_name=p.display_name, bbox=p.bbox,
                    state=STATE_IMPORTING, completion_delivered=True,
                    added_at=datetime.now(timezone.utc).isoformat()))
                self._executor.submit(self._run, op, slug, None)
                return "queued"
            # remove
            if current is None:
                return f"{slug} is not loaded"
            prior = current.state
            self._registry.set_state(slug, STATE_IMPORTING)
            self._executor.submit(self._run, op, slug, prior)
            return "queued"

    # --- execute (the single executor thread) ------------------------------------
    def _run(self, op: str, slug: str, prior: str | None) -> None:
        try:
            ok = bool(self._run_job(op, slug))
        except Exception:
            logger.exception("geo: %s job for %s raised", op, slug)
            ok = False
        if op == "transit":
            try:
                if ok:   # region was and stays READY; just gains transit
                    self._registry.update(slug, lambda r: replace(r, has_transit=True))
            finally:
                with self._submit_lock:
                    self._transit_active.discard(slug)
            return
        if op == "remove":
            if ok:
                self._registry.remove(slug)
            else:
                # the script validates-before-swap, so a returned failure means the
                # data is unchanged → restore the PRIOR state (READY or FAILED), never
                # promote a never-validated region to READY. (A killed job is caught by
                # reconcile → FAILED instead.)
                self._registry.set_state(slug, prior or STATE_READY)
            return
        # add — flip the state, then deliver (or record) the completion
        if ok:
            self._registry.update(slug, lambda r: replace(
                r, state=STATE_READY, completion_delivered=False))
        else:
            self._registry.set_state(slug, STATE_FAILED)
        self._deliver(slug, failed=not ok)

    # --- completion delivery (D4 hold-and-retry; C1 durable) ----------------------
    def _deliver(self, slug: str, *, failed: bool) -> None:
        # Read → dispatch → remove under one lock so a concurrent deliver_pending()
        # can't dispatch the same completion twice (it re-reads inside → finds the
        # proposal gone → returns).
        with self._deliver_lock:
            p = self._proposals.get(slug)
            if p is None or not p.origin_tid:
                return
            if not self._health_check():
                logger.info("geo: LLM not reachable; holding completion for %s", slug)
                return   # stays undelivered; deliver_pending() retries
            msg = (_failure_message(p.display_name) if failed
                   else _completion_message(p.display_name, p.user_request))
            try:
                self._on_complete(p.origin_tid, msg)
            except Exception:
                logger.exception("geo: completion dispatch for %s failed; will retry", slug)
                return
            if not failed:
                self._registry.update(slug, lambda r: replace(r, completion_delivered=True))
            self._proposals.remove(slug)

    def deliver_pending(self) -> None:
        """Re-attempt held completions (LLM was down, or the web restarted between
        'ready' and delivery). Failed adds keep their proposal until delivered too."""
        undelivered = {r.slug for r in self._registry.all()
                       if r.state == STATE_READY and not r.completion_delivered}
        for p in self._proposals.all():
            if p.slug in undelivered:
                self._deliver(p.slug, failed=False)
            else:
                r = self._registry.get(p.slug)
                if r is not None and r.state == STATE_FAILED:
                    self._deliver(p.slug, failed=True)

    def reconcile(self) -> list[str]:
        """Startup: flip orphaned ``importing`` entries (the restart killed their
        subprocess) to ``failed``, then let deliver_pending() notify."""
        return self._registry.reconcile()
