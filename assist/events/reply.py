"""Outbound SMS helpers plus the HITL-gated ``send_reply`` tool.

``send_reply`` is the one action a message-triage turn can take. It is gated by deepagents
human-in-the-loop
(:data:`REPLY_INTERRUPT_ON`): the agent calls ``send_reply(text)``, the graph interrupts
BEFORE the tool body runs, the user approves/edits/rejects, and only on approve does the
body execute — POSTing the reply to the phone's outbound-SMS endpoint.

The sender of the message being triaged rides the run config (``sms_sender``), set by the
inbound dispatch, the same way the schedule tools read ``thread_id``. On a normal thread
(no inbound message) there is no sender, so the tool declines.
"""
from __future__ import annotations

import logging
import os
import hashlib
import sqlite3
import uuid
from dataclasses import dataclass
from urllib.parse import urlsplit

from assist.events.sms_archive import configured_archive

import requests
from langgraph.config import get_config

logger = logging.getLogger(__name__)

# Run-config key carrying the inbound sender into the triage turn.
SMS_SENDER_KEY = "sms_sender"

# HITL config for the web agent build: pause before send_reply, offering approve/edit/reject
# (edit lets the user tweak the draft before it sends).
REPLY_INTERRUPT_ON = {
    "send_reply": {"allowed_decisions": ["approve", "edit", "reject"]},
}


def _sender() -> str | None:
    return ((get_config() or {}).get("configurable") or {}).get(SMS_SENDER_KEY)


@dataclass(frozen=True)
class SmsResult:
    """Observed outcome of one immutable send attempt, never a delivery guarantee."""

    status: str
    detail: str = ""


def send_outbound_sms(recipient: str, text: str, *, attempt_id: str | None = None) -> SmsResult:
    """Journal before sending; repeat attempts only read their existing outcome.

    A phone timeout or malformed response is unknown, including after a phone
    accepted the SMS. The phone's durable receipt can later settle that same id.
    Neither caller retries an uncertain send to repair its archive.
    """
    url = os.getenv("ASSIST_SMS_OUTBOUND_URL")
    secret = os.getenv("ASSIST_SMS_SECRET")
    if not url or not secret:
        return SmsResult("not_sent", "outbound SMS isn't configured")
    if urlsplit(url).path != "/outbound/sms/v1" or urlsplit(url).query or urlsplit(url).fragment:
        return SmsResult("not_sent", "the configured phone endpoint lacks the journal protocol; no send attempted")
    attempt_id = attempt_id or uuid.uuid4().hex
    try:
        archive = configured_archive()
        row, created = archive.record(attempt_id, "outgoing", recipient, text, "unknown")
    except (ValueError, OSError, sqlite3.Error):
        return SmsResult("not_sent", "couldn't safely journal this attempt; no send attempted")
    if not created:
        return SmsResult(row["status"], "existing attempt; no SMS repeated")
    status = "unknown"
    detail = "the phone's outcome isn't confirmed; do not resend"
    try:
        response = requests.post(
            url, json={"message_id": attempt_id, "to": recipient, "text": text, "occurred_at": row["occurred_at"]},
            headers={"X-Assist-SMS-Secret": secret}, timeout=(5, 70),
            allow_redirects=False)
        if response.status_code == 200:
            result = response.json()
            if result.get("message_id") == attempt_id and result.get("status") in {"sent", "not_sent", "unknown"}:
                status = result["status"]
                detail = "" if status != "unknown" else detail
    except (requests.RequestException, ValueError, AttributeError):
        # Do not log request exceptions: their string can contain endpoint secrets.
        logger.warning("outbound SMS response unavailable or invalid")
    try:
        row, _ = archive.record(attempt_id, "outgoing", recipient, text, status)
        return SmsResult(row["status"], detail)
    except (ValueError, OSError, sqlite3.Error):
        # The phone may already have sent it. Durable phone sync owns reconciliation.
        return SmsResult(status, "receipt storage failed; do not repeat this send")


def send_reply(text: str) -> str:
    """Reply to the sender of the message you're triaging, by text.

    The user approves this before it is sent — you are proposing the reply, not sending it
    yourself. Pass the exact message text to send. Only valid while triaging an inbound
    message (there must be a sender); otherwise it declines.
    """
    sender = _sender()
    if not sender:
        return "There is no inbound message to reply to in this turn."
    config = ((get_config() or {}).get("configurable") or {})
    approval = config.get("approval_id")
    if not approval:
        return "Reply not sent: this send has no confirmed approval identity."
    identity = f"{config.get('thread_id', '')}:{approval}"
    attempt_id = "reply_" + hashlib.sha256(identity.encode()).hexdigest()
    result = send_outbound_sms(sender, text, attempt_id=attempt_id)
    if result.status == "sent":
        return f"Reply sent to {sender}."
    if result.status == "unknown":
        return "Reply outcome unknown; do not resend. " + result.detail
    return f"Reply not sent: {result.detail or 'the phone confirmed no send'}."


def reply_tools() -> list:
    """The reply tool(s) for the web AgentSpec."""
    return [send_reply]
