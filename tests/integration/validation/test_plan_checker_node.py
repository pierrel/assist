from unittest import TestCase

from assist.reflexion_agent import (
    build_plan_check_node,
    Plan,
    Step,
    StepResolution,
    ReflexionState,
)

from .utils import DummyLLM, graphiphy


class TestPlanCheckerNode(TestCase):
    def setUp(self):
        llm = DummyLLM()
        self.graph = graphiphy(build_plan_check_node(llm, []))

        plan = Plan(
            goal="Prepare tea",
            steps=[
                Step(action="Boil water", objective="Boil water"),
                Step(action="Steep tea", objective="Steep tea"),
            ],
            assumptions=[],
            risks=[],
        )

        self.state: ReflexionState = {
            "messages": [],
            "plan": plan,
            "step_index": 1,
            "history": [
                StepResolution(action="Boil water", objective="Boil water", resolution="done")
            ],
            "needs_replan": False,
            "learnings": [],
        }

    def test_no_replan_needed(self):
        result = self.graph.invoke(self.state)
        self.assertFalse(result["needs_replan"])

