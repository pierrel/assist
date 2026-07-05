"""The ``notify`` agent tool — flags THIS thread as URGENT, a stronger signal than the
plain "unseen" badge: the web thread list shows an "urgent" pill on the thread and the iOS
PWA home-screen icon shows a badge dot until every urgent thread has been opened.

Thread-scoped like the schedule/subscription tools (reads ``thread_id`` from the run
config). Built by ``notify_tools(mark_urgent)`` and wired into the web ``AgentSpec`` NORMAL
tool set — deliberately NOT the untrusted SMS-triage set (an inbound text must not be able to
force a badge on the user's phone), and not a core built-in (the urgent state + badge live in
the web deployment only; emacsos/CLI have no Manager). Never raises into the agent loop.
"""
from __future__ import annotations

import html

from langgraph.config import get_config


def _thread_id() -> str | None:
    return ((get_config() or {}).get("configurable") or {}).get("thread_id")


def notify_tools(mark_urgent) -> list:
    """Return the notify tool, closing over the injected ``mark_urgent(tid)`` callback (so
    this module never imports web state — no import cycle)."""

    def notify(reason: str) -> str:
        """Flag THIS thread as URGENT / time-sensitive so the user is alerted it needs a
        fast look: the web thread list shows an "urgent" badge on this thread, and the
        home-screen app icon shows a badge dot until they open it.

        Use this ONLY for genuinely time-sensitive things the user must see soon (a message
        that needs a reply before a deadline, an event happening imminently) — NOT for
        routine responses. Pass a short REASON for why it's urgent; it's shown in the
        confirmation and stays in the conversation.
        """
        tid = _thread_id()
        if not tid:
            return "Couldn't flag this urgent: no active thread."
        mark_urgent(tid)
        # Escape reason: the tool result renders through markdown (raw HTML passes),
        # and reason could carry agent-echoed untrusted content — don't let it inject.
        return f"Flagged this thread as urgent: {html.escape(reason)}"

    return [notify]
