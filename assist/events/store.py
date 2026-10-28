"""Subscription persistence — the same per-thread, lock-serialized, atomic-write store the
schedules use (:class:`assist.record_store.PerThreadJsonStore`). A subscription lives in
``<root>/<tid>/subscriptions.json`` and dies with the thread. This subclass adds the record
type, cap, and the sender-routing query.
"""
from __future__ import annotations

from datetime import datetime

from assist.record_store import (
    PerThreadJsonStore,
    RecordCapExceeded,
    RecordNotFound,
)
from assist.events.model import Subscription

SUBSCRIPTIONS_FILE = "subscriptions.json"
CAP_PER_THREAD = 10


class SubscriptionCapExceeded(RecordCapExceeded):
    """Creating would exceed CAP_PER_THREAD on this thread."""


class SubscriptionNotFound(RecordNotFound):
    """No subscription with that id on that thread."""


class SubscriptionStore(PerThreadJsonStore[Subscription]):
    """Disk-backed subscription store, keyed by thread like the schedule store."""

    FILENAME = SUBSCRIPTIONS_FILE
    CAP = CAP_PER_THREAD
    CAP_EXC = SubscriptionCapExceeded
    NOTFOUND_EXC = SubscriptionNotFound

    @staticmethod
    def _from_dict(d: dict) -> Subscription:
        return Subscription.from_dict(d)

    def route(self, sender: str, event_kind: str = "sms", *, occurred_at: str | None = None) -> Subscription | None:
        """The subscription handling ``event_kind`` from ``sender``: the FIRST match by
        creation order (earliest ``created_at``) across all threads. When an arrival
        time is supplied, only subscriptions created strictly before it qualify.
        Missing/invalid subscription times cannot establish eligibility. Returns None when
        nothing matches — the caller records + logs the message but fires no turn (a
        catch-all is expressible as the regexp ``.*``)."""
        def precedes_arrival(subscription):
            if occurred_at is None:
                return True
            if subscription.created_at is None:
                return False
            try:
                arrival = datetime.fromisoformat(occurred_at)
                created = datetime.fromisoformat(subscription.created_at)
                return arrival.tzinfo is not None and created.tzinfo is not None and created < arrival
            except (TypeError, ValueError):
                return False
        matches = [s for s in self.all() if s.event_kind == event_kind and s.matches(sender) and precedes_arrival(s)]
        if not matches:
            return None
        return min(matches, key=lambda s: (s.created_at or ""))
