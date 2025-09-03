import pytest
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from assist.reflexion_agent import build_summarize_node, StepResolution, ReflexionState
from eval.types import Validation

from .utils import run_validation, DummyLLM


def check_messages(output: ReflexionState) -> bool:
    out_message = output["messages"][-1]
    is_aimessage = isinstance(out_message, AIMessage)
    has_china = "china" in out_message.content.lower()
    return is_aimessage and has_china


LLM = DummyLLM(message="Tea originated in China.")
GRAPH = build_summarize_node(LLM, [])

STATE: ReflexionState = {
    "messages": [SystemMessage("You are a helpful assistant"), HumanMessage("What's up with tea?")],
    "history": [
        StepResolution(action="Greet", objective="Say hi", resolution="Hi there!"),
        StepResolution(action="Share fact", objective="Inform user", resolution="Tea originated in China."),
    ],
}


VALIDATIONS = [Validation(input=STATE, check=check_messages)]


@pytest.mark.validation
@pytest.mark.parametrize("validation", VALIDATIONS)
def test_summarizer_node(validation: Validation) -> None:
    assert run_validation(GRAPH, validation)

