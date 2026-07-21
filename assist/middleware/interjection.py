"""Mid-turn interjection — deliver journaled user messages to the RUNNING turn.

A message sent while a turn is running lands durably in the pending-work
journal (=MESSAGE_BACKLOG=). This middleware's ``before_model`` hook peeks that
journal at every main-loop model-call boundary and, when unconsumed eligible
entries exist, appends them to graph state as individually framed
``HumanMessage``s — the model's very next call sees them and chooses redirect /
incorporate / defer / stop in its own voice (the four outcomes are DESCRIPTIVE,
never an interface; docs/2026-07-20-mid-turn-interjection-design.org).

This is a DELIVERY CHANNEL, not a behavior guard: no prompt can make the model
see a message that is not in its context (the guidance-first rule's "provably
can't" case). Everything behavioral lives in the injected framing.

Mechanics (all verified empirically at design time):
- ``before_model`` ONLY: at that boundary every prior tool call has its
  ToolMessage in state, so the appended user message lands after complete
  AI/Tool pairs — a sequence the live Qwen template accepts and honors.
- CLAIM IS DEFERRED: the hook never claims what it just injected — the message
  is only in memory until its superstep checkpoint commits. It claims entry
  ids found in messages ALREADY IN STATE (a prior superstep ⇒ durably
  checkpointed), via ``additional_kwargs["interjection_ids"]``; a second sweep
  runs at the turn's terminal exits. The shipped entry-gone gate then makes the
  queued fallback dispatcher skip a consumed entry — exactly-once, no new
  dedup machinery. An entry the turn never consumes runs as today's follow-up
  turn: nothing is ever lost.
- SCOPING is two gates: ``entry.origin is None`` (a user follow-up — an
  agent-scheduled continuation entry is work, not steering) and
  ``entry.sender == turn sender`` — the security posture: an untrusted SMS
  entry can never enter a full-privilege owner turn, an owner entry never
  lands inside a reply-to-stranger triage context, and owner entries
  naturally steer scheduled/continuation turns.
- CLAIM-SCAN INVARIANT (Pierre, PR #199): the journal id must be recoverable
  from the durable checkpoint at every claim site. This assumes message
  history is NON-MUTATING for kwargs: deepagents' summarization is (it
  summarizes via wrap_model_call without rewriting state); langchain's own
  SummarizationMiddleware rewrites history with RemoveMessage and would break
  the scan — do not swap it in without redesigning the claim.

Wired like ContextRider/notify: callbacks injected by the web layer
(``register_interjection_callbacks``); unregistered (CLI/emacsos/evals/
subagents) ⇒ inert by construction. The web layer owns the framing text (which
for an owner entry also enumerates + snapshots the pending continuations), the
claim side effects (clearing the snapshotted continuations — deferred to claim
so the clear is fate-shared with the message's durability), and the turn-scoped
claim tracking that feeds its fate-sharing re-journal on terminal error.
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
                       if r.origin is None and r.sender == sender
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
    return "" — it matches no entry (the store coerces "" to None on write,
    so no record ever carries it), degrading to inject-nothing rather than
    treating a possibly-triage turn as owner scope."""
    try:
        from langgraph.config import get_config
        from assist.events.reply import SMS_SENDER_KEY
        cfg = get_config() or {}
        return (cfg.get("configurable") or {}).get(SMS_SENDER_KEY) or None
    except Exception:
        return ""
