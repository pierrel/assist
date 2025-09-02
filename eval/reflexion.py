from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from assist.reflexion_agent import build_reflexion_graph, Plan, Step, StepResolution, ReflexionState
from assist.tools.base import base_tools
from eval.types import Validation
from langchain_openai import ChatOpenAI


def check_result(result: dict) -> bool:
    message = result["messages"][-1]
    is_aimessage = isinstance(message, AIMessage)
    doest_say_summary = "ummary" not in message.content
    says_france = "France" in message.content

    return is_aimessage and doest_say_summary and says_france

GRAPH = build_reflexion_graph(ChatOpenAI(model="gpt-4o-mini"),
                              base_tools("~/.cache/assist/dbs"),
                              [],
                              ChatOpenAI(model="gpt-4o-mini"))

VALIDATIONS = [
    Validation(
        input={"messages": [HumanMessage(content="Identify the capital of France and provide one fact about it.")]},
        check=check_result,
    )
]
