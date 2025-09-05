from unittest import TestCase
from langchain_core.messages import HumanMessage

from assist.reflexion_agent import build_plan_node

from .utils import thinking_llm, graphiphy

class TestPlannerNode(TestCase):
    def setUp(self) -> None:
        llm = thinking_llm("")
        self.graph = graphiphy(build_plan_node(llm,
                                               [],
                                               []))

    def test_tea_brew(self) -> None:
        state = self.graph.invoke({"messages": [HumanMessage(content="How do I brew a cup of tea?")]})
        
        plan = state["plan"]
        has_assumptions = bool(plan.assumptions)
        has_risks = bool(plan.risks)
        has_over_2_steps = len(plan.steps) > 2
        uses_tavily = any("tavily" in s.action for s in plan.steps)

        self.assertTrue(has_assumptions, "Has assumptions")
        self.assertTrue(has_risks, "Has risks")
        self.assertGreater(len(plan.steps), 2, "Should have more than 2 steps")
        self.assertTrue(uses_tavily, "Mentions tavily in any step")

    def test_rewrite_more_professional(self) -> None:
        query = "Rewrite this to be more professional."
        examples = [
            "hey—need that report asap. thx.",
            "We kinda dropped the ball on the Q3 metrics.",
        ]
        for example in examples:
            state = self.graph.invoke({
                "messages": [HumanMessage(content=f"{query} {example}")]
            })
            plan = state["plan"]
            self.assertGreater(len(plan.steps), 1, "has at least 2 steps")

    def test_rephrase_for_ninth_grade(self) -> None:
        query = "Rephrase for a 9th-grade reading level."
        examples = [
            "The municipality’s fiscal posture necessitates austerity measures.",
            "Our platform leverages distributed systems to optimize throughput.",
        ]
        for example in examples:
            state = self.graph.invoke({
                "messages": [HumanMessage(content=f"{query} {example}")]
            })
            plan = state["plan"]
            self.assertGreater(len(plan.steps), 1, "Has at least 2 steps")
            self.assertNotIn("tavily",
                             [step.action for step in plan.steps],
                             "It should not try to use the tavily tool")

    def test_extract_entities_to_json(self) -> None:
        query = "Extract all dates, people, and organizations from this text into JSON."
        examples = [
            "On March 2, 2024, Mayor London Breed met with leaders from SFUSD.",
            "Apple hired Sam Patel on 2023-11-14 after interviews at UCSF.",
        ]
        for example in examples:
            state = self.graph.invoke({
                "messages": [HumanMessage(content=f"{query} {example}")]
            })
            plan = state["plan"]
            self.assertGreater(len(plan.steps), 1, "Has at least 2 steps")
            self.assertNotIn("tool",
                             [step.action for step in plan.steps],
                             "No tool is available to be used for this task")

    def test_classify_customer_messages(self) -> None:
        query = (
            "Classify these customer messages into issue categories; return CSV."
        )
        examples = [
            "App crashes when I upload a photo.",
            "How do I reset my password?",
            "Please cancel my subscription.",
            "Charged twice for August.",
            "Search results are super slow.",
            "Two-factor code never arrives.",
            "Dark mode text is unreadable.",
        ]
        for example in examples:
            state = self.graph.invoke({
                "messages": [HumanMessage(content=f"{query} {example}")]
            })
            plan = state["plan"]
            self.assertGreater(len(plan.steps), 1, "Has at least 2 steps")
            self.assertNotIn("tool",
                             [step.action for step in plan.steps],
                             "No tool is available to be used for this task")


