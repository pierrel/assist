"""Example showing how to pass a ReflexionState to a single plan_check_node."""

from assist.reflexion_agent import (
    build_plan_check_node,
    Plan,
    PlanRetrospective,
    Step,
    StepResolution,
    ReflexionState,
)
from langchain_core.runnables import Runnable


class DummyLLM(Runnable):
    """Minimal LLM returning a fixed PlanRetrospective."""

    def __init__(self):
        self._schema = None

    def with_structured_output(self, schema):
        self._schema = schema
        return self

    def invoke(self, messages, opts=None):
        if self._schema is PlanRetrospective:
            # Always indicate no replan is needed
            return PlanRetrospective(needs_replan=False, learnings=None)
        raise ValueError("Unexpected schema: {self._schema}")


# Build the node using the dummy llm
llm = DummyLLM()
plan_check_node = build_plan_check_node(llm, callbacks=None)

# Create a sample plan and history for the state
plan = Plan(
    goal="Test goal",
    steps=[Step(action="do something", objective="achieve something")],
    assumptions=[],
    risks=[],
)

history = [
    StepResolution(
        action="do something",
        objective="achieve something",
        resolution="completed",
    )
]

state: ReflexionState = {
    "messages": [],
    "plan": plan,
    "step_index": 1,
    "history": history,
    "needs_replan": False,
    "learnings": [],
}

# Invoke the node with the prepared ReflexionState
result = plan_check_node(state)
print(result)
