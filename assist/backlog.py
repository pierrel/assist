"""Durable follow-up message backlog — messages accepted while a thread is busy.

A follow-up sent to a busy thread used to live only in-memory (a parked
BackgroundTask waiter, or the resume scheduler's queue) — a web restart silently
dropped it. Each such message is now journaled to
``<root_dir>/<tid>/pending_messages.json`` at submit and removed (claimed, by
id) when its turn actually starts, so at every instant it is durable in at
least one place: this journal while waiting, ``status.json``'s
``pending_message`` once its turn runs. Startup recovery re-dispatches whatever
is still journaled, in submit order. See
``docs/2026-07-13-durable-message-queue.org``.

The first message of an idle thread is NOT journaled — ``_mark_pending`` makes
it durable in ``status.json`` before the POST returns; the journal is
follow-ups only.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field

from assist.record_store import PerThreadJsonStore, RecordNotFound

logger = logging.getLogger(__name__)

BACKLOG_FILE = "pending_messages.json"


class PendingMessageNotFound(RecordNotFound):
    """No pending message with that id on that thread (a double-claim is benign)."""


@dataclass
class PendingMessage:
    """One accepted-but-not-yet-started follow-up. ``rider`` holds the four raw
    submit-form fields (sent_at/tz/lat/lon) so recovery rebuilds the ContextRider
    with the existing ``_build_rider``; ``sender`` is set for inbound-SMS
    follow-ups (it decides triage on re-dispatch)."""
    thread_id: str
    text: str
    sender: str | None = None
    rider: dict | None = None
    enqueued_at: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> dict:
        return {"id": self.id, "thread_id": self.thread_id, "text": self.text,
                "sender": self.sender, "rider": self.rider,
                "enqueued_at": self.enqueued_at}

    @staticmethod
    def from_dict(d: dict) -> "PendingMessage":
        return PendingMessage(
            thread_id=str(d.get("thread_id", "")), text=str(d.get("text", "")),
            sender=d.get("sender") or None, rider=d.get("rider") or None,
            enqueued_at=str(d.get("enqueued_at", "")),
            id=str(d.get("id") or uuid.uuid4().hex))


class MessageBacklog(PerThreadJsonStore[PendingMessage]):
    """Disk-backed follow-up journal. ``root_dir`` is the thread root
    (``MANAGER.root_dir``), matching ScheduleStore/SubscriptionStore. Uncapped on
    purpose: inflow is human-paced, and an over-cap behavior would mean refusing
    a user's message."""

    FILENAME = BACKLOG_FILE
    NOTFOUND_EXC = PendingMessageNotFound

    @staticmethod
    def _from_dict(d: dict) -> PendingMessage:
        return PendingMessage.from_dict(d)

    def _read(self, tid: str) -> list[PendingMessage]:
        """The base read swallows a parse failure to [] — here that would
        SILENTLY drop user messages, so make it loud: atomic tmp+rename makes a
        partial write impossible, meaning a parse failure is real disk
        corruption. The corrupt file is moved aside to ``<name>.corrupt`` for
        inspection (a later ``add``'s read-modify-write would otherwise replace
        it, destroying the evidence), and [] is returned so recovery proceeds
        for other threads."""
        path = self._path(tid)
        try:
            with open(path) as f:
                data = json.load(f)
        except FileNotFoundError:
            return []
        except json.JSONDecodeError:
            logger.error("message backlog for %s is unreadable — its queued "
                         "messages will NOT be recovered", tid, exc_info=True)
            try:
                os.replace(path, f"{path}.corrupt")
                logger.error("corrupt backlog moved to %s.corrupt for inspection",
                             path)
            except OSError:
                logger.error("corrupt backlog could NOT be moved aside; left at %s",
                             path)
            return []
        return [self._from_dict(d) for d in data]

    def claim(self, tid: str, rid: str) -> None:
        """Remove a journaled message whose turn is now running (claim by id).
        Idempotent — a recovery dedupe or a double-claim finds it already gone."""
        try:
            self.remove(tid, rid)
        except PendingMessageNotFound:
            pass
