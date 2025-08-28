"""Example showing how to pass a ReflexionState to a plan_check_node from YAML."""

import pathlib
import sys
import yaml

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from assist.reflexion_agent import (
    build_plan_check_node,
    Plan,
    PlanRetrospective,
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
        raise ValueError(f"Unexpected schema: {self._schema}")


# Build the node using the dummy llm
llm = DummyLLM()
plan_check_node = build_plan_check_node(llm, callbacks=None)

# Load the ReflexionState from a YAML file
data = yaml.safe_load(
    pathlib.Path(__file__).with_name("plan_check_state.yaml").read_text()
)

plan = Plan.model_validate(data["plan"])
history = [StepResolution.model_validate(h) for h in data["history"]]

state: ReflexionState = {
    "messages": data.get("messages", []),
    "plan": plan,
    "step_index": data["step_index"],
    "history": history,
    "needs_replan": data.get("needs_replan", False),
    "learnings": data.get("learnings", []),
}

if __name__ == "__main__":
    # Invoke the node with the prepared ReflexionState
    result = plan_check_node(state)
    print(result)
