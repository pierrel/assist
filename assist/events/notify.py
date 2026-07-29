"""The ``notify`` agent tool — flags THIS thread urgent and texts its configured owner.

The web thread list shows an "urgent" pill on the thread and the iOS PWA home-screen icon
shows a badge dot until every urgent thread has been opened.  When host configuration is
complete, the same call immediately sends the agent-provided message followed by a thread
link to the fixed recipient.

Thread-scoped like the schedule/subscription tools (reads ``thread_id`` from the run
config). Built by ``notify_tools(mark_urgent)`` and wired into the web ``AgentSpec`` NORMAL
tool set — deliberately NOT the untrusted SMS-triage set (an inbound text must not be able to
force a badge on the user's phone), and not a core built-in (the urgent state + badge live in
the web deployment only; emacsos/CLI have no Manager). Never raises into the agent loop.
"""
from __future__ import annotations

import html
import logging
import os
from urllib.parse import quote

from langgraph.config import get_config

from assist.events.reply import send_outbound_sms

logger = logging.getLogger(__name__)

_RECIPIENT_ENV = "URGENT_SMS_RECIPIENT"
_THREAD_URL_BASE_ENV = "URGENT_SMS_THREAD_URL_BASE"


def _thread_id() -> str | None:
    return ((get_config() or {}).get("configurable") or {}).get("thread_id")


def _thread_url(thread_id: str) -> str | None:
    base = os.getenv(_THREAD_URL_BASE_ENV, "").rstrip("/")
    if not base:
        return None
    return f"{base}/thread/{quote(thread_id, safe='')}"


def notify_tools(mark_urgent) -> list:
    """Return the notify tool, closing over the injected ``mark_urgent(tid)`` callback (so
    this module never imports web state — no import cycle)."""

    def notify(message: str) -> str:
        """Flag THIS thread as URGENT / time-sensitive so the user is alerted it needs a
        fast look: the web thread list shows an "urgent" badge on this thread, and the
        home-screen app icon shows a badge dot until they open it. It also immediately texts
        the configured owner with MESSAGE and a link to this thread.

        Use this ONLY for genuinely time-sensitive things the user must see soon (a message
        that needs a reply before a deadline, an event happening imminently) — NOT for
        routine responses. Pass a short, informative MESSAGE that says what needs attention;
        it is sent as text and shown in the confirmation.
        """
        tid = _thread_id()
        if not tid:
            return "Couldn't flag this urgent: no active thread."
        mark_urgent(tid)
        # Escape message: the tool result renders through markdown (raw HTML passes),
        # and message could carry agent-echoed untrusted content — don't let it inject.
        confirmation = f"Flagged this thread as urgent: {html.escape(message)}"
        recipient = os.getenv(_RECIPIENT_ENV)
        if not recipient:
            logger.warning("urgent SMS disabled: %s is not configured", _RECIPIENT_ENV)
            return f"{confirmation} SMS not sent: recipient isn't configured."
        thread_url = _thread_url(tid)
        if not thread_url:
            logger.warning("urgent SMS disabled: %s is not configured", _THREAD_URL_BASE_ENV)
            return f"{confirmation} SMS not sent: thread link isn't configured."
        failure = send_outbound_sms(recipient, f"{message}\n{thread_url}")
        if failure:
            return f"{confirmation} SMS not sent: {html.escape(failure)}."
        return f"{confirmation} SMS sent."

    return [notify]
