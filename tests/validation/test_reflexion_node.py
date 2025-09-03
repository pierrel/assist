import pytest
from langchain_core.messages import HumanMessage, AIMessage

import assist.reflexion_agent as reflexion_agent
from assist.reflexion_agent import build_reflexion_graph, ReflexionState
from eval.types import Validation

from .utils import run_validation, DummyLLM, DummyAgent


def check_result(result: ReflexionState) -> bool:
    message = result["messages"][-1]
    is_aimessage = isinstance(message, AIMessage)
    does_not_say_summary = "ummary" not in message.content
    says_france = "France" in message.content
    return is_aimessage and does_not_say_summary and says_france


LLM = DummyLLM(message="The capital of France is Paris.")


def fake_general_agent(_llm, _tools):
    return DummyAgent(message="Paris is the capital of France.")


reflexion_agent.general_agent = fake_general_agent
GRAPH = build_reflexion_graph(LLM, [], [], LLM)


VALIDATIONS = [
    Validation(
        input={
            "messages": [
                HumanMessage(
                    content="Identify the capital of France and provide one fact about it."
                )
            ]
        },
        check=check_result,
    )
]


@pytest.mark.validation
@pytest.mark.parametrize("validation", VALIDATIONS)
def test_reflexion_node(validation: Validation) -> None:
    assert run_validation(GRAPH, validation)

