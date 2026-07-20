"""Durable pending-work journal — the thread's job queue.

An entry is a turn that has been accepted but not yet started. Two origins:
- ``origin=None`` — a USER follow-up sent while the thread was busy (used to
  live only in-memory; a restart silently dropped it).
- ``origin="continuation"`` — AGENT-scheduled background work (the
  ``continue_later`` tool; docs/2026-07-19-progressive-responses-design.org),
  bounded by the 5-per-user-message chain cap enforced at that tool.

Entries live in ``<root_dir>/<tid>/pending_messages.json``, journaled before
the accepting call returns and removed (claimed, by id) when the turn actually
starts — so at every instant the work is durable in at least one place: this
journal while waiting, ``status.json``'s ``pending_message`` once it runs.
Startup recovery re-dispatches whatever is still journaled, in submit order.
See ``docs/2026-07-13-durable-message-queue.org`` (durability/recovery) and the
blank-slate assessment doc (this journal is the nascent job queue).

The first message of an idle thread is NOT journaled — ``_mark_pending`` makes
it durable in ``status.json`` before the POST returns.
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
    """One accepted-but-not-yet-started unit of work. ``rider`` holds the four
    raw submit-form fields (sent_at/tz/lat/lon) so recovery rebuilds the
    ContextRider with the existing ``_build_rider``; ``sender`` is set for
    inbound-SMS follow-ups (it decides triage on re-dispatch); ``origin`` is
    ``None`` for a user follow-up or ``"continuation"`` for agent-scheduled
    background work (render + dispatch key on it)."""
    thread_id: str
    text: str
    sender: str | None = None
    rider: dict | None = None
    enqueued_at: str = ""
    origin: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> dict:
        return {"id": self.id, "thread_id": self.thread_id, "text": self.text,
                "sender": self.sender, "rider": self.rider,
                "enqueued_at": self.enqueued_at, "origin": self.origin}

    @staticmethod
    def from_dict(d: dict) -> "PendingMessage":
        return PendingMessage(
            thread_id=str(d.get("thread_id", "")), text=str(d.get("text", "")),
            sender=d.get("sender") or None, rider=d.get("rider") or None,
            enqueued_at=str(d.get("enqueued_at", "")),
            origin=d.get("origin") or None,
            id=str(d.get("id") or uuid.uuid4().hex))


class MessageBacklog(PerThreadJsonStore[PendingMessage]):
    """Disk-backed pending-work journal (see the module docstring for the two
    origins). ``root_dir`` is the thread root (``MANAGER.root_dir``), matching
    ScheduleStore/SubscriptionStore. No store-level cap: USER follow-ups are
    human-paced and refusing one would mean 500ing a message, while the
    model-authored continuation inflow is bounded upstream by ``continue_later``'s
    5-per-user-message chain cap (the tool is the only continuation writer)."""

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

    def peek(self, tid: str) -> list[PendingMessage]:
        """LOCK-FREE, side-effect-free read for the event-loop thread (the web
        render): atomic tmp+rename writes mean a bare read sees whole-old or
        whole-new, never partial — the ``_get_status`` discipline. A corrupt
        file returns [] WITHOUT the move-aside (that mutation belongs to the
        locked ``_read`` path; the next locked reader preserves the evidence).
        Never call the locking APIs (``for_thread``/``all``) on the loop."""
        try:
            with open(self._path(tid)) as f:
                data = json.load(f)
            return [self._from_dict(d) for d in data]
        except Exception:
            # Maximally defensive — this runs on the event loop's render path,
            # where a wrong-shaped-but-parseable value (a non-dict entry) must
            # degrade to "nothing to show", never a 500. The locked readers
            # keep the strict path + the corrupt-file move-aside.
            return []

    def claim(self, tid: str, rid: str) -> None:
        """Remove a journaled message whose turn is now running (claim by id).
        Idempotent — a recovery dedupe or a double-claim finds it already gone."""
        try:
            self.remove(tid, rid)
        except PendingMessageNotFound:
            pass
