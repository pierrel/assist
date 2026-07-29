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


def send_outbound_sms(recipient: str, text: str) -> str | None:
    """Send one SMS through the configured phone endpoint.

    Return ``None`` after a confirmed send, otherwise a corrective detail.  Callers own
    their recipient-specific policy and presentation; this shared host-side boundary owns
    endpoint authentication, bounded delivery, and redirect refusal.
    """
    url = os.getenv("ASSIST_SMS_OUTBOUND_URL")
    secret = os.getenv("ASSIST_SMS_SECRET")
    if not url or not secret:
        return "outbound SMS isn't configured (ASSIST_SMS_OUTBOUND_URL / ASSIST_SMS_SECRET)"
    try:
        response = requests.post(
            url, json={"to": recipient, "text": text},
            headers={"X-Assist-SMS-Secret": secret}, timeout=10,
            allow_redirects=False)
    except requests.RequestException as error:
        logger.warning("outbound SMS failed: %s", error)
        return "couldn't reach the phone"
    if response.status_code == 200:
        return None
    logger.warning("outbound SMS returned HTTP %s", response.status_code)
    return f"the phone returned HTTP {response.status_code}"


def send_reply(text: str) -> str:
    """Reply to the sender of the message you're triaging, by text.

    The user approves this before it is sent — you are proposing the reply, not sending it
    yourself. Pass the exact message text to send. Only valid while triaging an inbound
    message (there must be a sender); otherwise it declines.
    """
    sender = _sender()
    if not sender:
        return "There is no inbound message to reply to in this turn."
    failure = send_outbound_sms(sender, text)
    if failure is None:
        return f"Reply sent to {sender}."
    return f"Reply not sent: {failure}."


def reply_tools() -> list:
    """The reply tool(s) for the web AgentSpec."""
    return [send_reply]
