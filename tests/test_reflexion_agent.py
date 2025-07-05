from unittest import TestCase
from unittest.mock import patch
from langchain_core.messages import HumanMessage, AIMessage
from assist.reflexion_agent import PlannerAgent, reflexion_agent
from assist.fake_runnable import FakeRunnable

class TestPlannerAgent(TestCase):
    def test_make_plan(self):
        llm = FakeRunnable([[AIMessage(content="1. foo\n2. bar")]])
        planner = PlannerAgent(llm, [])
        plan = planner.make_plan("do stuff")
        self.assertIn("foo", plan)

class TestReflexionAgent(TestCase):
    def test_invoke_runs_plan(self):
        plan_llm = FakeRunnable([[AIMessage(content="1. foo")]])
        react_resp = [AIMessage(content="all done")] 
        with patch('assist.reflexion_agent.general_agent') as mock_general:
            mock_general.return_value = FakeRunnable([react_resp])
            agent = reflexion_agent(plan_llm, [])
            resp = agent.invoke({'messages': [HumanMessage(content='hi')]})
            self.assertEqual(resp['messages'][-1].content, 'all done')
            mock_general.assert_called_once()

