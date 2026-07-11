"""``Provisioner`` — runs approved region jobs off-loop and tells the agent when done.

Deliberately a THIN executor, not a ``Scheduler`` clone (design doc, cross-lens shift):
region jobs are event-driven — submitted exactly once, on the user's approval — so
there is no poll loop, no cron math, and no in-flight dedup set (``max_workers=1``
serializes execution; the ``state:importing`` pre-write refuses a duplicate submit; the
scripts' shared flock excludes the weekly refresh cron).

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

    # --- enqueue (called from a web worker thread; never blocks) -----------------
    def submit(self, op: str, slug: str) -> str:
        """Enqueue a region job. Returns a short human-readable status ("queued" /
        why-refused). For an ``add``, the proposal record must exist (the HITL gate:
        approve fires the RECORDED slug); its catalog metadata seeds the registry
        entry."""
        if op not in OPS:
            return f"unknown operation: {op}"
        current = self._registry.get(slug)
        if current is not None and current.state == STATE_IMPORTING:
            return f"{slug} already has a job running"
        if op == "add":
            p = self._proposals.get(slug)
            if p is None:
                return f"no pending proposal for {slug}"
            if self._registry.is_loaded(slug):
                self._proposals.remove(slug)
                return f"{slug} is already loaded"
            self._registry.put(Region(
                slug=slug, display_name=p.display_name, bbox=p.bbox,
                state=STATE_IMPORTING, completion_delivered=True,
                added_at=datetime.now(timezone.utc).isoformat()))
        elif current is None:
            return f"{slug} is not loaded"
        else:
            self._registry.set_state(slug, STATE_IMPORTING)
        self._executor.submit(self._run, op, slug)
        return "queued"

    # --- execute (the single executor thread) ------------------------------------
    def _run(self, op: str, slug: str) -> None:
        try:
            ok = bool(self._run_job(op, slug))
        except Exception:
            logger.exception("geo: %s job for %s raised", op, slug)
            ok = False
        if op == "remove":
            if ok:
                self._registry.remove(slug)
            else:
                # data unchanged (validate-before-swap) — the region is still served
                self._registry.set_state(slug, STATE_READY)
            return
        if op == "transit":
            self._registry.update(slug, lambda r: replace(
                r, state=STATE_READY, has_transit=r.has_transit or ok))
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
