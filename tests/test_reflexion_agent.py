from unittest import TestCase
from unittest.mock import patch

from langchain_core.messages import HumanMessage, AIMessage

from assist.reflexion_agent import reflexion_agent
from assist.fake_runnable import FakeRunnable


class TestReflexionAgent(TestCase):
    def test_invoke_runs_plan(self):
        llm = FakeRunnable([
            [AIMessage(content="1. foo")],
            [AIMessage(content="summary")],
        ])
        react_resp = [AIMessage(content="all done")]
        with patch('assist.reflexion_agent.general_agent') as mock_general:
            mock_general.return_value = FakeRunnable([react_resp])
            agent = reflexion_agent(llm, [])
            resp = agent.invoke({'messages': [HumanMessage(content='hi')]})
            self.assertEqual(resp['messages'][-1].content, 'summary')
            mock_general.assert_called_once()

