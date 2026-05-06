"""Cooperative cancellation for :class:`ThreadAffinityQueue`.

The queue's watchdog flips ``handle.expired`` after ``hold_timeout_s``.
This middleware reads that flag in ``after_model`` and raises
:class:`ThreadHoldExpired` between LLM calls so the holder yields the
queue without corrupting an in-flight slot.
"""

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

from assist.thread_queue import ThreadHoldExpired, active_handle


class ThreadQueueMiddleware(AgentMiddleware):
    def after_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        handle = active_handle()
        if handle is not None and handle.expired:
            raise ThreadHoldExpired(
                f"thread {handle.thread_id} held the LLM queue past its cap; "
                "killed to avoid starving other threads"
            )
        return None
