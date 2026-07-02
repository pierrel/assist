"""The agent-facing subscription tools — the conversational surface for message-event
triage, thread-scoped like the schedule tools (each reads its ``thread_id`` from the run
config, so a subscription belongs to the thread it's created in and its matching messages
fire turns there).

Built by ``subscription_tools(store)`` and wired into the web ``AgentSpec`` (not core
built-ins): a subscription's effect needs the inbound-SMS route + the phone daemon, which
only the web deployment has. Tools never raise into the agent loop — every failure returns a
corrective string.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from langgraph.config import get_config

from assist.events.model import Subscription, validate_regexp, InvalidRegexp
from assist.events.store import SubscriptionCapExceeded, SubscriptionNotFound


def _thread_id() -> str | None:
    return ((get_config() or {}).get("configurable") or {}).get("thread_id")


def _line(s: Subscription) -> str:
    head = f"[{s.id}] sender ~ /{s.sender_regexp}/{'' if s.enabled else ' (paused)'}"
    preview = s.template.strip().splitlines()[0] if s.template.strip() else "(empty template)"
    return f"{head}\n    template: {preview[:80]}"


def subscription_tools(store) -> list:
    """Return the thread-scoped subscription tools closing over ``store``."""

    def create_subscription(sender_regexp: str, template: str) -> str:
        """Subscribe THIS thread to inbound messages whose SENDER matches SENDER_REGEXP.

        On each matching message a turn runs here with TEMPLATE rendered — put your whole
        instruction to yourself in TEMPLATE (how to decide what to do, your rules), using
        the literal slots {sender} and {text} where the message's sender and body go. You
        can only propose a reply (send_reply), which the user approves before it sends.

        sender_regexp is a Python regexp searched against the sender string (e.g.
        ``^\\+1555`` for a number prefix, ``.*`` for every sender). If the sender rule is at
        all complex, load the 'regexp' skill first to get it right. Returns the saved
        subscription.
        """
        tid = _thread_id()
        if not tid:
            return "Couldn't subscribe: no active thread."
        try:
            validate_regexp(sender_regexp)
        except InvalidRegexp as e:
            return (f"Couldn't subscribe: sender_regexp doesn't compile ({e}). "
                    f"Load the 'regexp' skill for help writing it.")
        if not template.strip():
            return "Couldn't subscribe: template is empty — write your triage instruction."
        sub = Subscription(id=os.urandom(6).hex(), thread_id=tid,
                           sender_regexp=sender_regexp, template=template,
                           created_at=datetime.now(timezone.utc).isoformat())
        try:
            store.add(sub)
        except SubscriptionCapExceeded as e:
            return f"Couldn't subscribe: {e}"
        return f"Subscribed. {_line(sub)}"

    def list_subscriptions() -> str:
        """List THIS thread's message-event subscriptions."""
        tid = _thread_id()
        if not tid:
            return "No active thread."
        subs = store.for_thread(tid)
        if not subs:
            return "This thread has no subscriptions."
        return "\n".join(_line(s) for s in subs)

    def modify_subscription(subscription_id: str, sender_regexp: str | None = None,
                            template: str | None = None) -> str:
        """Change an existing subscription. Pass ONLY the field(s) you're changing + the id;
        omitted fields keep their current value."""
        tid = _thread_id()
        if not tid:
            return "No active thread."
        if sender_regexp is None and template is None:
            return "Nothing to change — pass sender_regexp and/or template."
        if sender_regexp is not None:
            try:
                validate_regexp(sender_regexp)
            except InvalidRegexp as e:
                return f"Couldn't change it: sender_regexp doesn't compile ({e})."
        if template is not None and not template.strip():
            return "Couldn't change it: template can't be empty."

        def _apply(s: Subscription) -> Subscription:
            from dataclasses import replace
            return replace(
                s,
                sender_regexp=sender_regexp if sender_regexp is not None else s.sender_regexp,
                template=template if template is not None else s.template)
        try:
            saved = store.update(tid, subscription_id, _apply)
        except SubscriptionNotFound:
            return f"No subscription {subscription_id} on this thread."
        return f"Updated. {_line(saved)}"

    def _set_enabled(subscription_id: str, enabled: bool, verb: str) -> str:
        tid = _thread_id()
        if not tid:
            return "No active thread."
        try:
            saved = store.update(tid, subscription_id, lambda s: s.with_enabled(enabled))
        except SubscriptionNotFound:
            return f"No subscription {subscription_id} on this thread."
        return f"{verb}. {_line(saved)}"

    def pause_subscription(subscription_id: str) -> str:
        """Stop a subscription from firing (keep it; can be resumed)."""
        return _set_enabled(subscription_id, False, "Paused")

    def resume_subscription(subscription_id: str) -> str:
        """Resume a paused subscription."""
        return _set_enabled(subscription_id, True, "Resumed")

    def delete_subscription(subscription_id: str) -> str:
        """Delete a subscription from this thread permanently."""
        tid = _thread_id()
        if not tid:
            return "No active thread."
        try:
            store.remove(tid, subscription_id)
        except SubscriptionNotFound:
            return f"No subscription {subscription_id} on this thread."
        return f"Deleted subscription {subscription_id}."

    return [create_subscription, list_subscriptions, modify_subscription,
            pause_subscription, resume_subscription, delete_subscription]
