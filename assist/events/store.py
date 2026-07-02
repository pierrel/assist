"""Subscription persistence — the same per-thread, lock-serialized, atomic-write store the
schedules use (:class:`assist.record_store.PerThreadJsonStore`). A subscription lives in
``<root>/<tid>/subscriptions.json`` and dies with the thread. This subclass adds the record
type, cap, and the sender-routing query.
"""
from __future__ import annotations

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

    def route(self, sender: str) -> Subscription | None:
        """The subscription that handles a message from ``sender``: the FIRST match by
        creation order (earliest ``created_at``) across all threads. Returns None when
        nothing matches — the caller records + logs the message but fires no turn (a
        catch-all is expressible as the regexp ``.*``)."""
        matches = [s for s in self.all() if s.matches(sender)]
        if not matches:
            return None
        return min(matches, key=lambda s: (s.created_at or ""))
