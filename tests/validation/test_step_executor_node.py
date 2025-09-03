import pytest
from langchain_core.messages import SystemMessage, HumanMessage

from assist.reflexion_agent import build_execute_node, Plan, Step, ReflexionState
from eval.types import Validation

from .utils import run_validation, DummyLLM, DummyAgent


def has_resolution(state: ReflexionState) -> bool:
    return state["history"][0].resolution not in (None, "")


LLM = DummyLLM()

def fake_general_agent(_llm, _tools):
    return DummyAgent(message="step done")

AGENT = fake_general_agent(LLM, [])
GRAPH = build_execute_node(AGENT, [])

PLAN = Plan(
    goal="Greet user",
    steps=[Step(action="Greet the user politely", objective="Provide a friendly greeting")],
    assumptions=[],
    risks=[],
)

STATE: ReflexionState = {
    "messages": [SystemMessage("You are a helpful assistant"), HumanMessage("Hello, how are you?")],
    "plan": PLAN,
    "step_index": 0,
    "history": [],
    "needs_replan": False,
    "learnings": [],
}


VALIDATIONS = [Validation(input=STATE, check=has_resolution)]


@pytest.mark.validation
@pytest.mark.parametrize("validation", VALIDATIONS)
def test_step_executor_node(validation: Validation) -> None:
    assert run_validation(GRAPH, validation)

