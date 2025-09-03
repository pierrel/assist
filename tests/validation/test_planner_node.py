import pytest
from langchain_core.messages import HumanMessage

from assist.reflexion_agent import build_plan_node, ReflexionState
from eval.types import Validation

from .utils import run_validation, DummyLLM


def check_plan(state: ReflexionState) -> bool:
    plan = state["plan"]
    has_assumptions = bool(plan.assumptions)
    has_risks = bool(plan.risks)
    has_over_2_steps = len(plan.steps) > 2
    uses_tavily = any("tavily" in s.action for s in plan.steps)
    return has_assumptions and has_risks and has_over_2_steps and uses_tavily


LLM = DummyLLM()
GRAPH = build_plan_node(LLM, [], [])


VALIDATIONS = [
    Validation(
        input={"messages": [HumanMessage(content="How do I brew a cup of tea?")]},
        check=check_plan,
    ),
    Validation(
        input={"messages": [HumanMessage(content="Make a short plan for brewing a cup of tea.")]},
        check=check_plan,
    ),
]


@pytest.mark.validation
@pytest.mark.parametrize("validation", VALIDATIONS)
def test_planner_node(validation: Validation) -> None:
    assert run_validation(GRAPH, validation)

