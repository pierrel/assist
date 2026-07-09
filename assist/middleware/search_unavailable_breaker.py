"""Circuit breaker for a dead search backend.

When the self-hosted SearXNG backend is down/rate-limited, ``search_internet``
RETURNS a fixed string (``_SEARCH_UNAVAILABLE_MESSAGE``), not an exception — by
design, so a broken backend fails loud as a tool result the agent relays rather
than an exception that crashes the turn (see ``assist/tools.py``).  The string
tells the model to "stop", but the slow local model does NOT reliably obey: it
issues ANOTHER search with a DIFFERENT query, and another, grinding for minutes
against a confirmed-down backend until the ``recursion_limit`` trips.  Observed
live twice (30 min, then 14 min) on a rate-limited backend.

``LoopDetectionMiddleware`` does not catch this: ``search_internet`` is
read-only (so its same-tool-same-error Pattern A is transparent to it), and the
*distinct* queries defeat its same-args Pattern B.  That gap was left open
deliberately — distinct-arg exploration is normally legitimate.  This breaker
closes it for the ONE case where continuing is provably pointless: the tool
returned the EXACT, single ``_SEARCH_UNAVAILABLE_MESSAGE`` constant, which means
the backend is down and every further call will fail identically.  Keying on
that exact constant (not a fuzzy "looks like the same error" heuristic) makes
this a coarse REAL bound, not the ambiguous-signal heuristic that loop detection
was deliberately rolled back to avoid — so it stays a separate, isolable
middleware rather than a new branch in ``_detect_loop``.

Mechanism (FINALIZE-not-kill, mirroring ``ReadUrlRereadBreaker`` /
``UrlProvenanceMiddleware``): when the threshold is reached, ``wrap_tool_call``
refuses the next ``search_internet`` and returns a corrective ``ToolMessage``
instead of executing it — telling the model to write its best answer NOW from
what it already gathered this turn (or to say plainly it couldn't look this up
if it gathered nothing usable).  The turn CONTINUES to one more model turn, so
any partial/earlier results already in the message history are synthesized into
the final answer rather than discarded by a canned total-failure terminal (the
2026-06-05 volume-cap-destroys-output lesson).  Stateless — every decision is
read from the message tail, so it composes with checkpointing/rollback.

Scope: ``search_internet`` only.  ``read_url`` errors are deliberately NOT
handled here — they are ``f"Error fetching URL: {e}"`` (a distinct string per
URL, no clean equality to count) and distinct-URL read streaks are the
deliberately-tolerated exploration case.  The detection predicate is isolated in
``_count_search_unavailable`` so a future version could broaden it.
"""
import logging
from typing import Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

# Import the EXACT constant (single source of truth) so a wording change in
# tools.py can't silently desync this breaker.  Only the string constant is
# imported — never tools.py's blocking helpers (this hook runs inline on the
# event-loop thread; it must stay pure CPU).
from assist.tools import _SEARCH_UNAVAILABLE_MESSAGE
# Share loop detection's per-turn event extraction (windowless here — we want
# the cumulative full-turn count).  Same import precedent as
# empty_response_recovery importing `_last_successful_artifact`.
from assist.middleware.loop_detection import _extract_events

logger = logging.getLogger(__name__)

_SEARCH_TOOL = "search_internet"


def _count_search_unavailable(messages: list) -> int:
    """Count completed ``search_internet`` calls whose result is EXACTLY the
    unavailable constant, across the WHOLE current turn.

    ``window=None`` -> NO recency window (a backend being down is turn-global
    state; a windowed count could let heavy read_url interleaving push earlier
    unavailable searches out of view and undercount).  Exact-equality: the
    constant is returned verbatim from one code path, so a genuine empty result
    (``[]``) or any other content is NOT counted.  Pending calls (no result
    yet) have ``completed=False`` and are not counted."""
    return sum(
        1 for e in _extract_events(messages, window=None)
        if e["completed"]
        and e["tool_name"] == _SEARCH_TOOL
        and e["result_content"] == _SEARCH_UNAVAILABLE_MESSAGE
    )


def _correction() -> str:
    """The corrective tool result returned in place of the refused search — a
    FINALIZE nudge, not a total-failure terminal.  It tells the model to write
    its best answer NOW from what it already gathered this turn (never from
    memory), or to say plainly it couldn't look this up if it gathered nothing
    usable.  This preserves any partial/earlier results already in the message
    history (they are synthesized on the model's next turn) instead of the old
    strip-to-dead-end that discarded them (2026-06-05 volume-cap lesson).
    Deliberately no "try again in N minutes" framing — a down backend is an
    outage, not a rate-limit to wait out (see assist/tools.py)."""
    return (
        "Web search is unavailable and every further search will fail the same "
        "way — do NOT search again. Write your best answer NOW from the results "
        "you already gathered this turn (do NOT answer from your own knowledge), "
        "and note honestly that some sources couldn't be reached. If you gathered "
        "nothing usable at all, say plainly that you couldn't look this up right now."
    )


class SearchUnavailableBreakerMiddleware(AgentMiddleware):
    """Stop a turn from re-searching a confirmed-down backend — FINALIZE, not kill.

    After ``threshold`` completed ``search_internet`` results have come back as the
    exact unavailable constant this turn, the next ``search_internet`` call is
    refused: ``wrap_tool_call`` returns a corrective ``ToolMessage`` (instead of
    executing the search) that nudges the model to finalize from what it already
    gathered.  The turn CONTINUES, so partial/earlier results are synthesized into
    the answer rather than discarded (unlike the old strip-to-dead-end terminal).

    Stateless: every check inspects the message tail (no cross-turn instance
    state), so it is checkpoint/rollback safe.  Complementary to
    ``LoopDetectionMiddleware`` — that catches exact-repeat (same args / same
    error); this catches the distinct-query streak against a uniform unavailable
    result that loop detection deliberately ignores.

    Args:
        threshold: Max completed unavailable searches to allow before the next
            one is refused.  Default 4 — the prompt (the unavailable message +
            sub_research guidance) is the first line of defense and should make
            the model stop on its own well before this; the breaker is the HARD
            backstop for when the small model ignores the prompt.  Tunable via
            ``ASSIST_SEARCH_UNAVAILABLE_THRESHOLD`` at the install site.  Clamped
            to a minimum of 1 so a misconfigured 0/negative knob can't refuse
            every (even healthy) search request.
    """

    def __init__(self, threshold: int = 4):
        super().__init__()
        # Floor at 1: with threshold <= 0, `count < threshold` is never true and
        # the breaker would refuse EVERY search request, even on a healthy backend.
        self.threshold = max(1, threshold)
        self.tools = []
        self._intervention_count = 0

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], "ToolMessage | Command"],
    ) -> "ToolMessage | Command":
        if request.tool_call.get("name", "") != _SEARCH_TOOL:
            return handler(request)

        state = request.state or {}
        messages = state.get("messages", []) if isinstance(state, dict) \
            else getattr(state, "messages", [])
        if _count_search_unavailable(messages) < self.threshold:
            return handler(request)

        self._intervention_count += 1
        logger.warning(
            "SearchUnavailableBreaker: intervention #%d — >= %d unavailable search "
            "results this turn; refusing the next search_internet and nudging the "
            "model to finalize from what it has.",
            self._intervention_count, self.threshold,
        )
        return ToolMessage(
            content=_correction(),
            tool_call_id=request.tool_call.get("id", ""),
            name=_SEARCH_TOOL,
            status="error",
        )
