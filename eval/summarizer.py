import re
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from assist.reflexion_agent import build_summarize_node, StepResolution, ReflexionState
from .types import Validation
from assist.reflexion_agent import build_summarize_node
from langchain_openai import ChatOpenAI
from eval.types import Validation


llm = ChatOpenAI(model="gpt-4o-mini")

GRAPH = build_summarize_node(llm, [])

STATE = {
    "messages": [SystemMessage("You are a helpful assistant"),
                 HumanMessage("What's up with tea?")],
    "history": [
        StepResolution(action="Greet", objective="Say hi", resolution="Hi there!"),
        StepResolution(action="Share fact", objective="Inform user", resolution="Tea originated in China."),
    ],
}


def check_messages(output: ReflexionState) -> bool:
    out_message = output["messages"][-1]
    is_aimessage = isinstance(out_message, AIMessage)
    has_china = "china" in out_message.content or "China" in out_message.content

    return is_aimessage and has_china


VALIDATIONS = [
    Validation(
        input=STATE,
        check=check_messages,
    )
]
