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
        plan_llm = FakeRunnable([[AIMessage(content="1. foo\n2. bar")]])
        step_agent = FakeRunnable(
            [[AIMessage(content="step1 done")], [AIMessage(content="step2 done")]],
            append=True,
        )
        summary_agent = FakeRunnable([[AIMessage(content="final summary")]])

        with patch('assist.reflexion_agent.general_agent') as mock_general:
            mock_general.side_effect = [step_agent, summary_agent]
            agent = reflexion_agent(plan_llm, [])
            resp = agent.invoke({'messages': [HumanMessage(content='hi')]})
            self.assertEqual(resp['messages'][-1].content, 'final summary')
            self.assertEqual(mock_general.call_count, 2)
            # ensure step executions include prior context
            self.assertEqual(len(step_agent.calls), 2)
            first_call = step_agent.calls[0]["messages"]
            second_call = step_agent.calls[1]["messages"]
            self.assertEqual([m.content for m in first_call], ["hi", "foo"])
            self.assertEqual(
                [m.content for m in second_call],
                ["hi", "foo", "step1 done", "bar"],
            )

