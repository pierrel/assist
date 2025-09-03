import pytest
from langchain_core.messages import HumanMessage

from assist.reflexion_agent import build_plan_node, ReflexionState
from assist.tools.base import base_tools
from eval.types import Validation

from .utils import run_validation, thinking_llm, base_tools_for_test, graphiphy

def test_tea_brew():
    llm = thinking_llm("")
    graph = graphiphy(build_plan_node(llm,
                                      base_tools_for_test(),
                                      []))
    state = graph.invoke({"messages": [HumanMessage(content="How do I brew a cup of tea?")]})

    plan = state["plan"]
    has_assumptions = bool(plan.assumptions)
    has_risks = bool(plan.risks)
    has_over_2_steps = len(plan.steps) > 2
    uses_tavily = any("tavily" in s.action for s in plan.steps)

    assert has_assumptions and has_risks and has_over_2_steps and uses_tavily

