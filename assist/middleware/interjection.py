"""Deliver journaled messages at the running Deep turn's model boundary.

The web adapter scopes owner follow-ups and host egress decisions to their
logical-work target, preserving browser execution authority through tool
completion and fairness. Ordinary matching-sender messages retain their existing
scope; a host event can enter only an owner turn. Scheduled continuations stay
separate work. Framing distinguishes user intent from a host decision and its
untrusted recorded task; notification does not grant permissions.

Before-model delivery follows completed tool results. Injection carries each
Run ID but does not claim it until a subsequent checkpoint contains that ID.
The terminal sweep covers a last injection with no later model boundary. Claimed
Runs cannot execute again as queued fallback. Unread entries keep FIFO fallback;
terminal-error retries preserve origin and logical target. Checkpoint kwargs
must remain available to collect_interjection_ids, including after summaries.

The web embedder registers peek, consume and frame callbacks. Without callbacks
this middleware is inert. Durable journal, browser admission and native Hermes
receipt handling remain owned by their existing components.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from assist.thread_queue import active_handle

logger = logging.getLogger(__name__)

# The kwargs key carrying journal ids on injected messages — THE claim-scan
# invariant (Pierre, PR #199 note 2): every claim site recovers ids via
# collect_interjection_ids so the scans can never drift apart.
INTERJECTION_IDS_KEY = "interjection_ids"


def collect_interjection_ids(msgs) -> set:
    """Journal ids carried by messages already in (checkpointed) state."""
    ids: set = set()
    for m in msgs:
        ids.update((getattr(m, "additional_kwargs", None) or {}).get(
            INTERJECTION_IDS_KEY, []))
    return ids


# Set by the web layer at import time; None => middleware is inert.
#   peek(tid) -> list of journal records (this hook runs on the turn's worker
#                thread, so the locked read is fine)
#   consume(tid, ids) -> None  (claim by id + turn-scoped claim tracking for
#                               the web layer's error-exit re-journal + the
#                               deferred continuation clear; idempotent)
#   frame(record) -> str  (framing + message text; owner entries also
#                          enumerate pending continuations and snapshot their
#                          ids for the clear-at-claim)
_CALLBACKS: dict[str, Callable] | None = None


def register_interjection_callbacks(peek, consume, frame) -> None:
    global _CALLBACKS
    _CALLBACKS = {"peek": peek, "consume": consume, "frame": frame}


class InterjectionMiddleware(AgentMiddleware):
    """Deliver + claim journaled interjections at model-call boundaries.

    Claim tracking for the fate-sharing re-journal (a user's steering intent
    must survive a failed turn — Pierre, PR #199 note 5) lives WEB-SIDE inside
    the ``consume`` callback (a per-tid turn-scoped record list): turns
    serialize per thread, so the web layer can clear it at turn start and read
    it at terminal error exits without reaching into this compiled graph."""

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        if _CALLBACKS is None:
            return None
        handle = active_handle()
        if handle is None:
            return None
        tid = handle.thread_id
        try:
            # 1. CLAIM ids already durably in state (a prior superstep).
            in_state = collect_interjection_ids(state.get("messages", []))
            if in_state:
                _CALLBACKS["consume"](tid, in_state)

            # 2. INJECT unconsumed, sender-eligible entries not already in state.
            sender = _turn_sender(runtime)
            pending = [r for r in _CALLBACKS["peek"](tid)
                       if (r.origin is None or r.origin == "egress" and sender is None)
                       and r.sender == sender
                       and r.id not in in_state]
            if not pending:
                return None
            new_msgs = []
            for rec in pending:
                new_msgs.append(HumanMessage(
                    content=_CALLBACKS["frame"](rec),
                    additional_kwargs={INTERJECTION_IDS_KEY: [rec.id]}))
            logger.info("interjection: injected %d message(s) into %s",
                        len(new_msgs), tid)
            return {"messages": new_msgs}
        except Exception:
            # Delivery is best-effort per boundary: a journal hiccup must never
            # fail the running turn — the fallback follow-up path still delivers.
            logger.error("interjection hook failed for %s (fallback delivery "
                         "still applies)", tid, exc_info=True)
            return None


def _turn_sender(runtime: Runtime) -> str | None:
    """The running turn's sender (None = owner web/scheduled/continuation turn;
    an SMS triage turn carries it in the run config). Read via langgraph's
    get_config() — the run config is NOT on ``runtime`` in this langchain
    (the ContextRiderMiddleware lesson). FAIL CLOSED: on any read failure
    return "" — it matches no entry (the store coerces "" to None on every
    read via from_dict, so no record it returns ever carries it), degrading
    to inject-nothing rather than treating a possibly-triage turn as owner
    scope."""
    try:
        from langgraph.config import get_config
        from assist.events.reply import SMS_SENDER_KEY
        cfg = get_config() or {}
        return (cfg.get("configurable") or {}).get(SMS_SENDER_KEY) or None
    except Exception:
        return ""
