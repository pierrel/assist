"""Middleware to record good states in the execution history.

Small models (e.g. vLLM-served Ministral) hallucinate in a variety of cases.
This causes wasted execution time retrying requests that will never succeed
and ultimately error out with no clear reason to the user.

This middleware records the checkpoint ID present in runtime.config each time
the model returns a valid response (i.e. no exception was raised).  These IDs
are passed to invoke_with_rollback so that on a BadRequestError it can jump
directly to the most recent known-good checkpoint instead of stepping back one
checkpoint at a time through potentially-corrupt history.
"""
import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


class GoodStateTrackerMiddleware(AgentMiddleware):
    """Record checkpoint IDs where the model returned successfully.

    Usage::

        tracker = GoodStateTrackerMiddleware()
        agent   = create_agent(..., good_state_tracker=tracker)
        result  = invoke_with_rollback(agent, input, config,
                                       good_state_tracker=tracker)
    """

    def __init__(self):
        # Ordered list of checkpoint IDs known to be in a good state.
        # Newest checkpoints are appended; best_rollback_target() scans
        # newest-first so we land as close to the failure as possible.
        self.good_states: list[str] = []

    # ------------------------------------------------------------------
    # after_model: no error occurred, so record the current checkpoint ID
    # ------------------------------------------------------------------
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Called after every successful model response.

        The checkpoint_id in runtime.config is the checkpoint that was active
        going *into* this step — i.e. the last committed good state.  We record
        it so that rollback can target it directly instead of walking back one
        step at a time.
        """
        config = getattr(runtime, "config", None)
        if not config or not isinstance(config, dict):
            return None
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")
        if checkpoint_id and checkpoint_id not in self.good_states:
            self.good_states.append(checkpoint_id)
            logger.debug(
                "GoodStateTracker: recorded checkpoint %s (total tracked: %d)",
                checkpoint_id,
                len(self.good_states),
            )
        return None

    # ------------------------------------------------------------------
    # Helper used by invoke_with_rollback
    # ------------------------------------------------------------------
    def best_rollback_target(self, history: list) -> Any | None:
        """Return the most recent good StateSnapshot from *history*, or None.

        *history* is ordered newest-first, as returned by
        ``agent.get_state_history(config)``.  We scan newest-first so we
        land as close to the point of failure as possible.
        """
        if not self.good_states:
            return None
        good_set = set(self.good_states)
        for snapshot in history:
            cid = snapshot.config.get("configurable", {}).get("checkpoint_id", "")
            if cid in good_set:
                return snapshot
        return None
