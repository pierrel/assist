"""Give the agent one trusted chance to finish before its graph limit.

RemainingSteps is relative to the current LangGraph execution, even when its
checkpoint step numbers include earlier turns. After delivery the warning stays
in model context; only its delivery flag is checkpointed. Continuations retain
that flag, and a new graph entry resets it. Tools and the terminal recursion error keep
their ordinary behavior.
"""
from typing import Annotated

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import SystemMessage
from langgraph.managed.is_last_step import RemainingStepsManager


FINALIZATION_STEPS = 50
EXECUTION_STEP_LIMIT = 500 + FINALIZATION_STEPS
LIMIT_WARNING = (
    "You are nearing the execution limit. If you have a clear path to finish "
    "within the remaining budget, finish. Otherwise stop investigating and "
    "honestly report completed work, unfinished work and blockers. Never claim "
    "pending actions succeeded."
)


class ExecutionLimitState(AgentState):
    # LangGraph recognizes a managed value only as the last annotation;
    # LangChain also needs it excluded from graph input/output schemas.
    remaining_steps: Annotated[int, PrivateStateAttr, RemainingStepsManager]
    execution_limit_warned: Annotated[bool, PrivateStateAttr]


class ExecutionLimitMiddleware(AgentMiddleware):
    state_schema = ExecutionLimitState

    def before_agent(self, state, runtime):
        return {"execution_limit_warned": False}

    def before_model(self, state, runtime):
        if (state["remaining_steps"] <= FINALIZATION_STEPS
                and not state.get("execution_limit_warned", False)):
            return {"execution_limit_warned": True}

    def wrap_model_call(self, request, handler):
        if not request.state.get("execution_limit_warned", False):
            return handler(request)
        base = request.system_message
        system = (base.model_copy(update={"content": [
            *base.content_blocks, {"type": "text", "text": f"\n\n{LIMIT_WARNING}"},
        ]}) if base else SystemMessage(content=LIMIT_WARNING))
        return handler(request.override(system_message=system))
