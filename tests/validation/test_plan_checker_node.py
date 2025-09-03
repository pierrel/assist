import pytest

from assist.reflexion_agent import (
    build_plan_check_node,
    Plan,
    Step,
    StepResolution,
    ReflexionState,
)
from eval.types import Validation

from .utils import run_validation, DummyLLM


def check_needs_replan(state: ReflexionState) -> bool:
    return state["needs_replan"] is False


LLM = DummyLLM()
GRAPH = build_plan_check_node(LLM, [])

PLAN = Plan(
    goal="Prepare tea",
    steps=[
        Step(action="Boil water", objective="Boil water"),
        Step(action="Steep tea", objective="Steep tea"),
    ],
    assumptions=[],
    risks=[],
)

STATE: ReflexionState = {
    "messages": [],
    "plan": PLAN,
    "step_index": 1,
    "history": [
        StepResolution(action="Boil water", objective="Boil water", resolution="done")
    ],
    "needs_replan": False,
    "learnings": [],
}


VALIDATIONS = [Validation(input=STATE, check=check_needs_replan)]


@pytest.mark.validation
@pytest.mark.parametrize("validation", VALIDATIONS)
def test_plan_checker_node(validation: Validation) -> None:
    assert run_validation(GRAPH, validation)

