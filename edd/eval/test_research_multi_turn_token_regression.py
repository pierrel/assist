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

Root causes (full analysis: docs/2026-04-26-token-max-mismatch-investigation.md):

1. `ContextAwareToolEvictionMiddleware` only acts on incoming tool
   results, not on the full message list before send.
2. The middleware estimates tokens with `len(content) // 4`, which
   underestimates Qwen3-Coder tokenization.
3. `ModelRetryMiddleware` does not retry on `BadRequestError` (only
   transient 5xx / network errors).
4. `BadRequestRetryMiddleware` exists but was not wired into the general
   or research agents.

What this test pins down: a three-turn run of the exact prompts that
failed must complete without `BadRequestError` reaching the caller.

The agent's *research quality* is not asserted — the failure under test
is about context-size handling, not answer correctness.  Lenient
`assertTrue` on response presence is included only as a sanity check
that turns landed at all.

This test is intentionally network-bound: the failing pattern only
appears when real research tools accumulate real web-search results.
Mocking the tools would not reproduce the message-buildup that
overflows the cap.
"""
import logging
import os
import sys
import tempfile
import shutil
from unittest import TestCase

from openai import BadRequestError

from assist.thread import ThreadManager


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

    Pass criterion: no `openai.BadRequestError` reaches the caller across
    all three turns.  Other failures (network, recursion limits, rate
    limits) are not the bug under test — they are reported as test errors
    rather than failures so the regression signal stays clean.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tm = ThreadManager(root_dir=self.tmpdir)
        self.thread = self.tm.new()

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
