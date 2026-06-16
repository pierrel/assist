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

Mechanism mirrors ``LoopDetectionMiddleware.after_model`` exactly (the
established clean turn-ender in this stack): when the threshold is reached and
the latest ``AIMessage`` requests yet another ``search_internet``, strip its
tool calls and replace the content with a composed status report.  The agent
loop ends because no tool calls remain to dispatch.  Stateless — every decision
is read from the message tail, so it composes with checkpointing/rollback.

Scope: ``search_internet`` only.  ``read_url`` errors are deliberately NOT
handled here — they are ``f"Error fetching URL: {e}"`` (a distinct string per
URL, no clean equality to count) and distinct-URL read streaks are the
deliberately-tolerated exploration case.  The detection predicate is isolated in
``_count_search_unavailable`` so a future version could broaden it.
"""
import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

# Import the EXACT constant (single source of truth) so a wording change in
# tools.py can't silently desync this breaker.  Only the string constant is
# imported — never tools.py's blocking helpers (this hook runs inline on the
# event-loop thread; it must stay pure CPU).
from assist.tools import _SEARCH_UNAVAILABLE_MESSAGE
# Share loop detection's per-turn event extraction (windowless here — we want
# the cumulative full-turn count) and its turn-terminator.  Same import
# precedent as empty_response_recovery importing `_last_successful_artifact`.
from assist.middleware.loop_detection import _extract_events, strip_tool_calls_to_end_turn

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


def _compose_terminal_message() -> str:
    """First-person status report (the SUBAGENT's voice) for the stripped AI
    message.  This becomes the subagent's report to the orchestrator, so it is
    a status report, NOT the raw ``_SEARCH_UNAVAILABLE_MESSAGE`` (which is a
    second-person model-directive that reads wrong across the agent boundary).
    Worded to lexically match the orchestrator's existing "research-agent
    reports that search is unavailable -> relay and stop" prompt branch
    (research_instructions.txt.j2:21).  The USER-facing message + what to tell
    the user is owned by the prompts, NOT this middleware: the main agent
    relays it per general_instructions.md.j2:72-87 ("tell the user plainly ...
    web search is currently unavailable, and end the turn").  Deliberately no
    "try again in N minutes" framing — a down backend is an outage to fix, not
    a rate-limit to wait out (see assist/tools.py)."""
    return (
        "I couldn't complete this research because web search is currently "
        "unavailable — the search backend could not be reached, so I couldn't "
        "look this up."
    )


class SearchUnavailableBreakerMiddleware(AgentMiddleware):
    """Terminate a turn that keeps calling ``search_internet`` against a
    confirmed-down backend.

    On the (``threshold``+1)th search request, after ``threshold`` results have
    already come back as the exact unavailable constant, the latest AI message's
    tool calls are stripped and its content replaced with a composed status
    report; the loop ends because no tool calls remain.

    Stateless: every check inspects the message tail (no cross-turn instance
    state), so it is checkpoint/rollback safe.  Complementary to
    ``LoopDetectionMiddleware`` — that catches exact-repeat (same args / same
    error); this catches the distinct-query streak against a uniform
    unavailable result that loop detection deliberately ignores.

    Args:
        threshold: Max completed unavailable searches to allow before the next
            one is short-circuited.  Default 4 — the prompt (the unavailable
            message + sub_research guidance) is the first line of defense and
            should make the model stop on its own well before this; the breaker
            is the HARD backstop for when the small model ignores the prompt.
            Tunable via ``ASSIST_SEARCH_UNAVAILABLE_THRESHOLD`` at the install
            site.  Clamped to a minimum of 1 so a misconfigured 0/negative knob
            can't terminate every (even healthy) search request.
    """

    def __init__(self, threshold: int = 4):
        super().__init__()
        # Floor at 1: with threshold <= 0, `count < threshold` is never true and
        # the breaker would strip EVERY search request, even on a healthy backend.
        self.threshold = max(1, threshold)
        self.tools = []
        self._intervention_count = 0

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        last = messages[-1]
        if not isinstance(last, AIMessage):
            return None
        if not getattr(last, "tool_calls", None):
            return None

        # Only act if the model is about to issue ANOTHER search — otherwise it
        # may already be giving up / moving to a different tool on its own.
        last_call_names = {(tc.get("name") or "") for tc in last.tool_calls}
        if _SEARCH_TOOL not in last_call_names:
            return None

        count = _count_search_unavailable(messages)
        if count < self.threshold:
            return None

        terminal_content = _compose_terminal_message()
        self._intervention_count += 1
        logger.warning(
            "SearchUnavailableBreaker: intervention #%d — %d unavailable "
            "search results this turn (threshold %d); stripping the next "
            "search_internet call and terminating the turn.",
            self._intervention_count,
            count,
            self.threshold,
        )

        return strip_tool_calls_to_end_turn(last, terminal_content)
