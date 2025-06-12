from unittest import TestCase
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import pdb
import server  # under test


class TestServer(TestCase):
    def test_simple_work_messages(self):
        langchain_messages = [HumanMessage(content="What is 2 plus 2?")]
        agent = server.get_agent('mistral', 0.4)
        resp = agent.invoke({"messages": langchain_messages})
        work_messages = server.work_messages(resp["messages"])
        
        work_message_types = list(map(__class__, work_messages))
        self.assertListEqual(work_message_types,
                             [AIMessage])
        self.assertEqual(len(work_messages), 1, "There should be 1 \
        work message when tools are not required")        

    def test_tool_work_messages(self):
        langchain_messages = [HumanMessage(content="What is the size \
        of the capital of Colorado in the united states?")]
        agent = server.get_agent('mistral', 0.4)
        resp = agent.invoke({"messages": langchain_messages})
        work_messages = server.work_messages(resp["messages"])
        work_message_types = list(map(__class__, work_messages))
        self.assertListEqual(work_message_types, [])
        self.assertEqual(len(work_messages), 3, "There should be 1 \
        work message when tools are not required")        
