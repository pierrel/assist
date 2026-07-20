"""The ``continue_later`` agent tool — schedule background work and answer NOW.

The structural lever for progressive responses
(docs/2026-07-19-prd-progressive-responses.org): a turn that fans into fast
local work + slow follow-on work answers with the fast results immediately and
journals the slow work as a *continuation* — an ``origin="continuation"``
entry in the pending-work journal, dispatched by the web layer at the turn's
``ready`` exit as an ordinary self-message turn (queue / fairness / sandbox /
recovery / unseen-badge all inherited; docs/2026-07-19-progressive-responses-
design.org).

Chain cap: at most 5 continuation turns between user messages, enforced HERE
(this tool is the only continuation writer, and turns serialize per thread, so
one gate suffices). The count is DERIVED — trailing continuation-marked turns
in the conversation plus unclaimed journal entries — via the injected
``chain_len`` callback; there is no counter file to desync or corrupt.

Wired like ``notify_tools``: thread-scoped via the run config, callbacks
injected (this module never imports web state), NORMAL web tool set only —
deliberately NOT the untrusted SMS-triage set (an inbound text must not be
able to schedule agent-invented background work). Never raises into the agent
loop — every outcome is a corrective/directive string.
"""
from __future__ import annotations

from langgraph.config import get_config

CHAIN_CAP = 5   # max agent-initiated turns between user messages (Pierre, PRD)


def _thread_id() -> str | None:
    return ((get_config() or {}).get("configurable") or {}).get("thread_id")


def continuation_tools(journal, chain_len) -> list:
    """Return the continue_later tool, closing over two injected callbacks:
    ``journal(tid, task) -> None`` appends the continuation to the pending-work
    journal (+ event log); ``chain_len(tid) -> int`` returns the current chain
    length (trailing continuation turns + already-journaled continuations)."""

    def continue_later(task: str) -> str:
        """Schedule background work to run AFTER you answer, and follow up with the
        user when it completes. Use this when part of the answer is ready NOW and the
        rest needs slow work (like research): give the user what you have, and put the
        slow part here instead of doing it in this turn.

        ``task`` must be a COMPLETE, self-contained instruction for your future self,
        who will NOT remember this turn's plan: say exactly what to find out or do, AND
        what you already told the user, AND that the results must be reported back in a
        follow-up message. The task text is shown to the user as the pending work.

        After this tool returns, FINISH YOUR ANSWER and end the turn: answer from what
        you already have and tell the user you'll follow up. Do NOT also do the
        background work in this turn — that defeats the point.
        """
        tid = _thread_id()
        if not tid:
            return "Couldn't schedule background work: no active thread."
        task = " ".join((task or "").split())
        if not task:
            return ("Nothing scheduled: `task` was empty. Pass a complete, "
                    "self-contained instruction for the background work.")
        if chain_len(tid) >= CHAIN_CAP:
            return (f"Cap reached ({CHAIN_CAP} background turns since the user's "
                    "last message) — nothing scheduled. Do NOT promise more "
                    "background work: end your answer honestly, tell the user "
                    "there is more to do, and wait for their go-ahead.")
        journal(tid, task)
        return ("Background work scheduled — it runs after this turn and the "
                "results will be posted here as a follow-up. Now finish your "
                "answer from what you already have, tell the user you'll follow "
                "up, and END this turn without doing the scheduled work.")

    return [continue_later]
