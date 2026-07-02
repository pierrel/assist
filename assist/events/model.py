"""The message-event subscription record.

A subscription matches an inbound message's *sender* by regexp and, on a match, fires an
agent turn in the subscription's thread using ``template`` — the whole self-instruction the
agent authors (guidance/rules woven in however it likes), with ``{sender}`` / ``{text}``
slots for the event data. Deliberately just a regexp + a template (Pierre's review): one
free-form template gives the agent freedom to structure the instruction its own way, and
routing is a sender-regexp rather than per-sender threads.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace


class InvalidRegexp(Exception):
    """The sender_regexp doesn't compile."""


@dataclass(frozen=True)
class Subscription:
    """One message-event subscription bound to one thread."""
    id: str
    thread_id: str
    sender_regexp: str
    template: str
    enabled: bool = True
    created_at: str | None = None   # ISO-8601 UTC; also the routing tie-break order

    def matches(self, sender: str) -> bool:
        """True iff this (enabled) subscription's regexp is found in ``sender``. A bad
        regexp never matches (validated at creation, but stay defensive at match time)."""
        if not self.enabled:
            return False
        try:
            return re.search(self.sender_regexp, sender) is not None
        except re.error:
            return False

    def render(self, sender: str, text: str) -> str:
        """Render the template for one event. Literal-token replacement (not str.format)
        so a template containing other braces — JSON, code — is left untouched."""
        return self.template.replace("{sender}", sender).replace("{text}", text)

    def with_enabled(self, enabled: bool) -> "Subscription":
        return replace(self, enabled=enabled)

    def to_dict(self) -> dict:
        return {"id": self.id, "thread_id": self.thread_id,
                "sender_regexp": self.sender_regexp, "template": self.template,
                "enabled": self.enabled, "created_at": self.created_at}

    @classmethod
    def from_dict(cls, d: dict) -> "Subscription":
        return cls(id=d["id"], thread_id=d["thread_id"],
                   sender_regexp=d["sender_regexp"], template=d["template"],
                   enabled=d.get("enabled", True), created_at=d.get("created_at"))


def validate_regexp(pattern: str) -> None:
    """Raise :class:`InvalidRegexp` if ``pattern`` doesn't compile — so the setup tool can
    return a corrective string to the model instead of storing a dead subscription."""
    try:
        re.compile(pattern)
    except re.error as e:
        raise InvalidRegexp(str(e))
