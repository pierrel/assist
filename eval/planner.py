from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from assist.reflexion_agent import build_plan_node, Plan, Step, StepResolution, ReflexionState
from assist.tools.base import base_tools
from eval.types import Validation
from langchain_openai import ChatOpenAI


def check_plan(state: ReflexionState) -> bool:
    has_assumptions = state["plan"].assumptions
    has_risks = state["plan"].risks
    has_over_2_steps = len(state["plan"].steps) > 2
    uses_tavily = any(["tavily" in s.action for s in state["plan"].steps])

    return bool(has_assumptions and has_risks and has_over_2_steps and uses_tavily)


GRAPH = build_plan_node(ChatOpenAI(model="gpt-4o-mini"),
                        base_tools("~/.cache/assist/dbs"),
                        [])


VALIDATIONS = [
    Validation(
        input={"messages": [HumanMessage(content="How do I brew a cup of tea?")]},
        check=check_plan
    ),
    Validation(
        input={"messages": [HumanMessage(content="Make a short plan for brewing a cup of tea.")]},
        check=check_plan
    )
]
