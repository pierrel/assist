from unittest import TestCase
from langchain_core.messages import HumanMessage

from assist.reflexion_agent import build_plan_node, Plan, Step
from assist.tools.filesystem import write_file

from .utils import graphiphy, DummyLLM


class _FilePlanLLM:
    """Minimal LLM stub that plans file-writing steps based on the request."""

    def __init__(self):
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages, _opts=None):
        if self.schema is Plan:
            request = messages[-1].content if messages else ""
            if "README.md" in request and "CHANGELOG.md" in request:
                steps = [
                    Step(action="draft_readme", objective="compose README"),
                    Step(action="write_file", objective="save README"),
                    Step(action="draft_changelog", objective="compose changelog"),
                    Step(action="write_file", objective="save changelog"),
                ]
            elif "notes.txt" in request:
                steps = [
                    Step(action="draft_content", objective="prepare file text"),
                    Step(action="write_file", objective="save text to disk"),
                ]
            else:
                steps = []
            return Plan(goal="goal", steps=steps, assumptions=["assumption"], risks=["risk"])
        raise ValueError("Unexpected schema")

class TestPlannerNode(TestCase):
    def setUp(self) -> None:
        llm = DummyLLM("")
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

    def test_plan_includes_write_file(self) -> None:
        """Planner should propose using write_file for file creation tasks."""
        llm = _FilePlanLLM()
        graph = graphiphy(build_plan_node(llm, [write_file], []))
        state = graph.invoke({
            "messages": [HumanMessage(content="Create notes.txt summarizing the meeting")]
        })
        plan = state["plan"]
        actions = [s.action for s in plan.steps]
        self.assertIn("write_file", actions, "Planner includes write_file step")
        self.assertGreater(len(plan.steps), 1, "Plan has multiple steps")

    def test_plan_multiple_write_files(self) -> None:
        """Planner should handle requests requiring multiple file writes."""
        llm = _FilePlanLLM()
        graph = graphiphy(build_plan_node(llm, [write_file], []))
        state = graph.invoke({
            "messages": [HumanMessage(content="Create README.md and CHANGELOG.md for the project")]
        })
        plan = state["plan"]
        actions = [s.action for s in plan.steps]
        self.assertEqual(actions.count("write_file"), 2, "Two write_file steps expected")

