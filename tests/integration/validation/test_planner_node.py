import pytest
from unittest import TestCase
from langchain_core.messages import HumanMessage

from assist.reflexion_agent import build_plan_node, ReflexionState
from assist.tools.base import base_tools
from eval.types import Validation

from .utils import run_validation, thinking_llm, base_tools_for_test, graphiphy

class TestPlannerNode(TestCase):
    def setUp(self):
        llm = thinking_llm("")
        self.graph = graphiphy(build_plan_node(llm,
                                               base_tools_for_test(),
                                               []))

    def ask_node(self, query: str) -> ReflexionState:
        message = HumanMessage(content=query)
        return self.graph.invoke({"messages": [message]})


    def test_search_website(self):
        state = self.ask_node("I remember seeing something about college campuses with the best food on this website: https://www.mentalfloss.com. What's the URL for that article?")

        self.assertTrue(any(["site_search" in s.action.lower() for s in state["plan"].steps]))



        def test_search_webpage(self):
        state = self.ask_node("Which campus has the best food according to this website: https://www.mentalfloss.com/food/best-and-worst-college-campus-food?utm_source=firefox-newtab-en-us ?")

        self.assertTrue(any(["page_search" in s.action.lower() for s in state["plan"].steps]))


    def test_tea_brew(self):
        state = self.graph.invoke({"messages": [HumanMessage(content="How do I brew a cup of tea?")]})
        
        plan = state["plan"]
        has_assumptions = bool(plan.assumptions)
        has_risks = bool(plan.risks)
        has_over_2_steps = len(plan.steps) > 2
        uses_tavily = any("tavily" in s.action for s in plan.steps)
        
        assert has_assumptions and has_risks and has_over_2_steps and uses_tavily

