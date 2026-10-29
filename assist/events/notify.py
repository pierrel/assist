"""Queue a reasoned phone alert for this thread, without sending an SMS."""
from __future__ import annotations

import html
from langgraph.config import get_config

MAX_REASON_CHARS = 200


def valid_reason(message) -> bool:
    """Whether MESSAGE is a nonempty, bounded, display-safe notification reason."""
    return (isinstance(message, str) and bool(message.strip())
            and len(message) <= MAX_REASON_CHARS
            and all(ord(char) >= 32 and ord(char) != 127 for char in message))


def _thread_id() -> str | None:
    return ((get_config() or {}).get('configurable') or {}).get('thread_id')


def notify_tools(mark_urgent) -> list:
    """Bind notify to the host's durable, thread-scoped phone attention marker."""
    def notify(message: str) -> str:
        """Alert the user to something time-sensitive in THIS thread.

        Supply a short meaningful reason, at most 200 characters. A phone buzz is
        queued until the phone connects; physical vibration is not confirmed.
        Calls while this thread is still urgent update its reason without another
        buzz. Opening the thread clears that urgent cycle. Use this for important
        attention, not routine responses.
        """
        if not valid_reason(message):
            return 'Phone alert unavailable: provide a nonempty, display-safe reason of at most 200 characters.'
        tid = _thread_id()
        if not tid:
            return 'Phone alert unavailable: no active thread.'
        try:
            status = mark_urgent(tid, message.strip())
        except (OSError, ValueError):
            return 'Phone alert unavailable: its reason could not be saved.'
        reason = html.escape(message.strip())
        if status == 'handled':
            return f'Phone alert already handled for this urgent thread; reason updated: {reason}. Physical vibration is unverified.'
        if status == 'pending':
            return f'Phone alert already pending; reason updated: {reason}. Physical vibration is unverified.'
        return f'Phone alert queued: {reason}. Physical vibration is unverified.'
    return [notify]
