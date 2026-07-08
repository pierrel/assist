"""Cooperative cancellation + fair-scheduling pause for :class:`ThreadAffinityQueue`.

The queue's tick timer flips two flags on the holder handle, which this middleware
reads in ``after_model`` (fired between LLM calls, so it yields at a superstep
boundary without corrupting an in-flight slot):

- ``expired`` — cumulative-active hold hit the 2h cap -> raise
  :class:`ThreadHoldExpired` (TERMINAL runaway backstop).  Checked FIRST.
- ``pause_requested`` — the quantum expired with a waiter present -> raise
  :class:`ThreadPauseRequested` (NON-terminal: the run path pauses the turn and
  resumes it later from its durable checkpoint).

The cap wins over the quantum (the tick never sets both), and ``expired`` is
checked first here as belt-and-suspenders so a kill is never demoted to a pause.
"""

import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

from assist.thread_queue import (
    ThreadHoldExpired,
    ThreadPauseRequested,
    active_handle,
)

logger = logging.getLogger(__name__)


class ThreadQueueMiddleware(AgentMiddleware):
    def after_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        handle = active_handle()
        if handle is None:
            return None
        if handle.expired:
            logger.info("thread-queue: yielding %s — hold cap hit (kill)", handle.thread_id)
            raise ThreadHoldExpired(
                f"thread {handle.thread_id} held the LLM queue past its cap; "
                "killed to avoid starving other threads"
            )
        if handle.pause_requested:
            logger.info("thread-queue: yielding %s — quantum pause for a waiting turn",
                        handle.thread_id)
            raise ThreadPauseRequested(
                f"thread {handle.thread_id} paused at its quantum to let a waiting "
                "turn run; will resume from its checkpoint"
            )
        return None
