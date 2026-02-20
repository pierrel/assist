"""Checkpoint-based rollback for unrecoverable model errors.

When a model API call returns a BadRequestError (400), the graph state
typically contains messages with content that the model API can't parse
(null bytes, malformed JSON, etc.).  Simple retries won't help because
the same bad state is sent every time.

Instead, this module rolls back to a *previous* checkpoint — one that
existed before the bad content was produced — and re-invokes the graph
from there.  Because LLM output is non-deterministic, the model will
often take a different path and avoid the bad content entirely.

Usage:
    result = invoke_with_rollback(agent, input_data, config)
"""
import logging
from typing import Any

from openai import BadRequestError
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


def invoke_with_rollback(
    agent: CompiledStateGraph,
    input_data: dict[str, Any] | None,
    config: dict[str, Any],
    *,
    max_retries_per_step: int = 2,
    max_rollback_depth: int = 3,
    rollback_on: tuple[type[Exception], ...] = (BadRequestError,),
) -> dict[str, Any]:
    """Invoke an agent with checkpoint-based rollback for state errors.

    Args:
        agent: A compiled LangGraph agent with a checkpointer.
        input_data: Initial input (e.g. ``{"messages": [...]}``) or ``None``
            to resume from the current checkpoint.
        config: LangGraph config — must include ``configurable.thread_id``.
        max_retries_per_step: How many times to retry from the same
            checkpoint before rolling back one step further.
        max_rollback_depth: Maximum number of checkpoints to go back
            through before giving up.
        rollback_on: Exception types that trigger a rollback (default:
            ``BadRequestError``).

    Returns:
        The agent's invoke result (dict with ``messages`` key).

    Raises:
        The original exception if all rollback attempts are exhausted.
    """
    current_input = input_data
    current_config = config

    # Captured once on the first failure so subsequent retries
    # roll back through the *original* history, not through
    # checkpoints created by failed retry attempts.
    original_history = None

    depth = 0                # index into original_history (skipping [0])
    retries_at_depth = 0     # attempts at the current depth

    max_attempts = 1 + max_rollback_depth * max_retries_per_step
    for _attempt in range(max_attempts):
        try:
            return agent.invoke(current_input, current_config)

        except BaseException as exc:
            if not isinstance(exc, rollback_on):
                raise

            # --- First failure: capture the checkpoint history ----------
            if original_history is None:
                original_history = list(
                    agent.get_state_history(current_config)
                )
                if len(original_history) < 2:
                    # Only the bad (or no) checkpoint — can't roll back.
                    raise

            # --- Decide where to roll back to --------------------------
            retries_at_depth += 1

            if retries_at_depth > max_retries_per_step:
                # Exhausted retries at this depth — go one step deeper.
                depth += 1
                retries_at_depth = 1

            if depth >= max_rollback_depth:
                raise

            # history[0] is the most recent (likely bad) checkpoint.
            # history[1] is one step before it, etc.
            target_idx = depth + 1
            if target_idx >= len(original_history):
                raise

            target = original_history[target_idx]
            cp_id = target.config["configurable"].get("checkpoint_id")
            logger.warning(
                "Rollback: depth=%d retry=%d/%d checkpoint=%s",
                depth, retries_at_depth, max_retries_per_step, cp_id,
            )

            current_config = target.config
            current_input = None   # resume from the checkpoint

    # Should not reach here, but just in case:
    raise RuntimeError("invoke_with_rollback: max attempts exceeded")
