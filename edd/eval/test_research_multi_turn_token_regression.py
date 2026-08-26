"""Regression test for the SF Giants thread that overflowed context by 1 token.

On 2026-04-26, thread `20260426073544-71c17777` raised:

    BadRequestError: This model's maximum context length is 53616 tokens.
    However, you requested 0 output tokens and your prompt contains at
    least 53617 input tokens, for a total of at least 53617 tokens.

The thread was three sequential research-style requests through the
general agent (which delegates to the research subagent).  Each turn
caused the research subagent to accumulate web-search results and draft
reports.  By turn 3 the message history exceeded the model server's
53,616-token cap by exactly one token.

Historical root causes (resolved by the 2026-05-16 context-management
overhaul — see docs/2026-05-16-context-management-overhaul.org and the
original 2026-04-26 incident analysis at
docs/2026-04-26-token-max-mismatch-investigation.md):

1. `ContextAwareToolEvictionMiddleware` (now deleted) only acted on
   incoming tool results, not on the full message list before send.
2. That middleware estimated tokens with `len(content) // 4`, which
   underestimates Qwen3-Coder tokenization.
3. `ModelRetryMiddleware` does not retry on `BadRequestError` (only
   transient 5xx / network errors).
4. `BadRequestRetryMiddleware` exists but was not wired into the general
   or research agents.

Today this test guards against the SAME failure mode under a different
architecture: deepagents 0.6.1's auto-installed `SummarizationMiddleware`
(trigger=fraction 0.85, real LLM-summarization, offload to
`/conversation_history/`) handles compaction; `BadRequestRetryMiddleware`
handles terminal sanitize-and-truncate.  If summarization regresses or
its plumbing breaks, this test should fail again.

What this test pins down: a three-turn run of the exact prompts runs under the
eval runner's fixed 480-second OS-level cap.  A natural pytest completion must
not leak `BadRequestError`; a timeout is recorded as a failed stress trial, not
silently treated as an overflow pass.

The agent's *research quality* is not asserted — the failure under test
is about context-size handling, not answer correctness.  Lenient
`assertTrue` on response presence is included only as a sanity check
that turns landed at all.

The research subagent is MOCKED (a large canned report per turn via
``stub_research_subagent``) — see ``AGENTS.md`` testing guideline #5:
real search rate-limits SearXNG, and this eval tests the caller-facing
overflow guard, not search behavior.  The fixture explicitly constructs the
historical synchronous general-agent composition; ordinary web runs are now
async and have separate lifecycle coverage.  The canned report drives the
orchestrator's context/summarization path across the three turns without
SearXNG, so the run is rate-limit-free and deterministic.  (Historical note:
the original ``BadRequestError`` was a 53k-context older model; on the current
131k model this run passes without forcing overflow — it is a plumbing
regression guard for the summarizer/retry wiring, which the mocked context
still exercises.)
"""
import logging
import os
import sys
import tempfile
import shutil
from unittest import TestCase

from openai import BadRequestError

from assist.spec import AgentSpec
from assist.thread import Thread
from assist.thread_manager import ThreadManager
from edd.eval.utils import stub_research_subagent

# A substantial, prompt-relevant canned research report drives the historical
# synchronous context/summarization path across the three turns without hitting
# SearXNG. Numbered source notes keep the bulk varied so the parent can use the
# findings rather than mistaking its own fixture for a failed research loop.
_CANNED_REPORT = (
    "# Fixture research report: San Francisco Giants tickets\n\n"
    "## Plan findings\n"
    "The fixture records full-season, partial-plan, and flexible-membership "
    "options. Exact weekend or 1 PM bundles depend on that season's inventory, "
    "so the customer should ask the club about current availability.\n\n"
    "## Flexible-membership calculation\n"
    "The fixture's example has a $500 membership credit, a $50 single-game seat, "
    "and a 30 percent member discount, making the member seat $35 and the saving "
    "$15 per game. $500 divided by $15 is 33.34, so 34 attended games offset the "
    "membership fee through the discount alone; the included $500 credit changes "
    "the overall value comparison.\n\n"
    "## Recorded sources\n"
    "https://www.mlb.com/giants/tickets\n"
    "https://www.mlb.com/giants/tickets/season-tickets\n\n"
    + "\n".join(
        f"Supporting ticket record {number}: plan availability, seat pricing, "
        f"credits, and discounts are recorded for comparison {number}."
        for number in range(1, 201)
    )
)


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


# Verbatim from the failing thread (user's report).  Curly quotes and
# punctuation preserved — the small model's tokenization of unicode
# punctuation is part of the 1-token overshoot story.
TURN_1 = (
    "How do “season tickets” for mlb (sf giants in particular) "
    "work? Is it possible to buy a bundle of the “chap seat” "
    "tickets only for Saturday or Sunday, 1PM games?"
)

TURN_2 = (
    "Please look into “Checking if the Giants offer flexible "
    "ticketing options for specific game times”. What do they "
    "offer? Please include verified links in your response."
)

TURN_3 = (
    "Can you do the calculations for me on how much I can save on the "
    "$500 flexible membership option when I save 30% per seat? Look up "
    "seat prices with and without the discount and credit and let me "
    "know how many games I have to go to to “break even” "
    "compared to just buying single-game tickets."
)


class TestGiantsThreadTokenRegression(TestCase):
    """Three-turn run of thread 20260426073544-71c17777.

    The runner, not pytest-timeout, supplies this row's 480-second process
    cap (see ``scripts/run-evals.sh``).  Its retained log is also the evidence
    source for peak approximate context, tool-call count, and completion
    status.  The cap is intentionally external: it kills a real agent loop
    instead of adding a test-only branch that could hide one.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tm = ThreadManager(root_dir=self.tmpdir)

    def tearDown(self):
        try:
            self.tm.close()
        finally:
            if os.path.isdir(self.tmpdir):
                shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _send(self, turn_label: str, text: str) -> str:
        """Send a turn.  Translate BadRequestError into a test failure
        (this is the regression target).  Anything else propagates and
        is reported as a test error.
        """
        try:
            return self.thread.message(text)
        except BadRequestError as exc:
            self.fail(
                f"{turn_label}: BadRequestError leaked past retry/rollback. "
                f"This is the regressed failure mode. Error: {exc}"
            )

    def test_three_turns_no_token_overflow(self):
        # Explicitly select the incident's synchronous composition.  A bare
        # ThreadManager.new() now creates an async web-main agent, whose task tools
        # require a durable web Run and therefore cannot represent this regression.
        with stub_research_subagent(_CANNED_REPORT):
            thread_id = self.tm.reserve()
            working_dir = self.tm.make_default_working_dir(self.tm.thread_dir(thread_id))
            self.thread = Thread(
                working_dir, thread_id=thread_id, checkpointer=self.tm.checkpointer,
                model=self.tm.model, spec=AgentSpec())
            r1 = self._send("turn 1", TURN_1)
            logger.info("turn 1 response (first 300 chars): %s", str(r1)[:300])
            r2 = self._send("turn 2", TURN_2)
            logger.info("turn 2 response (first 300 chars): %s", str(r2)[:300])
            r3 = self._send("turn 3", TURN_3)
            logger.info("turn 3 response (first 300 chars): %s", str(r3)[:300])

        # Sanity: the thread actually progressed.  Empty replies on every
        # turn would suggest the agent is silently bailing rather than
        # hitting the bug.
        self.assertTrue(r1, "turn 1 returned no content")
        self.assertTrue(r2, "turn 2 returned no content")
        self.assertTrue(r3, "turn 3 returned no content")
