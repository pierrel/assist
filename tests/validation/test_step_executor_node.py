from unittest import TestCase
from langchain_core.messages import SystemMessage, HumanMessage

from assist.reflexion_agent import build_execute_node, Plan, Step, ReflexionState

from .utils import DummyLLM, DummyAgent, graphiphy


class TestStepExecutorNode(TestCase):
    def setUp(self):
        llm = DummyLLM()

        def fake_general_agent(_llm, _tools):
            return DummyAgent(message="step done")

        agent = fake_general_agent(llm, [])
        self.graph = graphiphy(build_execute_node(agent, []))

        plan = Plan(
            goal="Greet user",
            steps=[
                Step(
                    action="Greet the user politely",
                    objective="Provide a friendly greeting",
                )
            ],
            assumptions=[],
            risks=[],
        )

        self.state: ReflexionState = {
            "messages": [
                SystemMessage("You are a helpful assistant"),
                HumanMessage("Hello, how are you?"),
            ],
            "plan": plan,
            "step_index": 0,
            "history": [],
            "needs_replan": False,
            "learnings": [],
        }

    def test_step_execution_adds_resolution(self):
        result = self.graph.invoke(self.state)
        self.assertTrue(result["history"][0].resolution)

