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

Subagent rollback (Option A — implemented):
    Wrap each subagent's ``runnable`` with ``RollbackRunnable`` so that
    when the subagent's internal ``invoke()`` hits a BadRequestError the
    *subagent* graph is rolled back independently from the parent.  This
    is safe for read-only subagents (context-agent) and additive-only
    ones (research-agent).  The dev-agent is excluded for now because it
    writes to the real filesystem and Docker sandbox — after rollback it
    would not know what files it already modified.

Future option (Option B — not yet implemented):
    Use a ``wrap_tool_call`` middleware on the *parent* agent to catch
    errors from the ``task`` tool (which invokes subagents).  On failure,
    the parent agent's ``task`` tool result is replaced with an error
    message so the parent can decide how to proceed (retry, skip, or
    re-prompt).  This avoids subagent-level rollback entirely — the
    parent stays in control — but it means the subagent's partial work
    (file writes, Docker commands) still persists.  Worth exploring if
    Option A proves insufficient for agents with real-world side effects.
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
    thread_id = config.get("configurable", {}).get("thread_id", "?")

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
                logger.warning(
                    "Rollback[%s]: %s on first attempt, "
                    "%d checkpoints available (max_depth=%d, retries_per_step=%d)",
                    thread_id, type(exc).__name__,
                    len(original_history), max_rollback_depth, max_retries_per_step,
                )
                if len(original_history) < 2:
                    logger.warning(
                        "Rollback[%s]: only %d checkpoint(s) — cannot roll back, re-raising",
                        thread_id, len(original_history),
                    )
                    raise

            # --- Decide where to roll back to --------------------------
            retries_at_depth += 1

            if retries_at_depth > max_retries_per_step:
                # Exhausted retries at this depth — go one step deeper.
                depth += 1
                retries_at_depth = 1

            if depth >= max_rollback_depth:
                logger.error(
                    "Rollback[%s]: exhausted all %d depths — giving up",
                    thread_id, max_rollback_depth,
                )
                raise

            # history[0] is the most recent (likely bad) checkpoint.
            # history[1] is one step before it, etc.
            target_idx = depth + 1
            if target_idx >= len(original_history):
                raise

            target = original_history[target_idx]
            cp_id = target.config["configurable"].get("checkpoint_id")
            step = target.metadata.get("step", "?")
            logger.warning(
                "Rollback[%s]: depth=%d retry=%d/%d → checkpoint=%s (step %s)",
                thread_id, depth, retries_at_depth, max_retries_per_step,
                cp_id, step,
            )

            current_config = target.config
            current_input = None   # resume from the checkpoint

    # Should not reach here, but just in case:
    raise RuntimeError("invoke_with_rollback: max attempts exceeded")


class RollbackRunnable:
    """Wrap a compiled LangGraph agent so that ``invoke()`` uses rollback.

    This is designed to wrap subagent runnables passed to ``CompiledSubAgent``.
    When the subagent's own ``invoke()`` hits a rollback-eligible error, it
    rolls back through the *subagent's* checkpoints independently of the parent.

    All other attributes (``get_state``, ``get_state_history``, ``stream``, etc.)
    are proxied to the underlying agent so the wrapper is transparent.

    Args:
        agent: The compiled subagent graph.
        max_retries_per_step: Passed to ``invoke_with_rollback``.
        max_rollback_depth: Passed to ``invoke_with_rollback``.
        rollback_on: Exception types that trigger rollback.
    """

    def __init__(
        self,
        agent: CompiledStateGraph,
        *,
        max_retries_per_step: int = 2,
        max_rollback_depth: int = 3,
        rollback_on: tuple[type[Exception], ...] = (BadRequestError,),
    ):
        self._agent = agent
        self._max_retries_per_step = max_retries_per_step
        self._max_rollback_depth = max_rollback_depth
        self._rollback_on = rollback_on

    def invoke(self, input_data, config=None, **kwargs):
        return invoke_with_rollback(
            self._agent,
            input_data,
            config or {},
            max_retries_per_step=self._max_retries_per_step,
            max_rollback_depth=self._max_rollback_depth,
            rollback_on=self._rollback_on,
        )

    def __getattr__(self, name):
        # Proxy everything else to the underlying agent
        return getattr(self._agent, name)
