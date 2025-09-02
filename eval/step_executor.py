from langchain_core.messages import HumanMessage, SystemMessage
from assist.reflexion_agent import (
    build_execute_node,
    Plan,
    Step,
    ReflexionState,
    StepResolution,
    general_agent
)
from langchain_openai import ChatOpenAI
from assist.tools.base import base_tools
from eval.types import Validation

llm = ChatOpenAI(model="gpt-4o-mini")
tools = base_tools("~/.cache/assist/dbs/")


def has_resolution(state: ReflexionState) -> bool:
    return state["history"][0].resolution not in (None, "")


GRAPH = build_execute_node(general_agent(llm,
                                         tools),
                           [])

PLAN = Plan(
    goal="Greet user",
    steps=[Step(action="Greet the user politely", objective="Provide a friendly greeting")],
    assumptions=[],
    risks=[],
)

STATE = ReflexionState(
    messages=[SystemMessage("You are a helpful assistant"),
              HumanMessage("Hello, how are you?")],
    plan=PLAN,
    step_index=0,
    history=[],
    needs_replan=False,
    learnings=[],
)

VALIDATIONS = [
    Validation(
        input=STATE,
        check=has_resolution
    )
]
