from unittest import TestCase
from langchain_ollama import ChatOllama
from typing import Union
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
import os
import general_agent

AnyMessage = Union[HumanMessage, AIMessage, ToolMessage]

def tool_messages(resp_messages: AnyMessage) -> list[ToolMessage]:
    return [x for x in resp_messages if isinstance(x, ToolMessage)]


class TestGeneralAgent(TestCase):
    @classmethod
    def setUpClass(cls):
        llm = ChatOllama(model="mistral", temperature=0.4)
        cls.agent = general_agent.general_agent(llm, [])

    def test_fs_tools_list_files(self):
        current_dir = os.getcwd()
        current_file = os.path.basename(__file__)
        cont = f"List the files inside of {current_dir}"
        message = HumanMessage(content=cont)
        resp = self.agent.invoke({"messages": message})
        resp_messages = resp['messages']

        self.assertEqual(len(tool_messages(resp_messages)), 1)
        self.assertTrue(current_file in resp_messages[-1].content)

    def test_fs_tools_show_file_contents(self):
        current_dir = os.getcwd()
        current_file = os.path.basename(__file__)

        cont = f"What is in the file {current_file} within the directory {current_dir}? What does it do?"
        message = HumanMessage(content=cont)
        resp = self.agent.invoke({'messages': message})
        resp_messages = resp['messages']

        self.assertEqual(len(tool_messages(resp_messages)), 2)
        self.assertTrue("show_file_contents" in resp_messages[-1].content)
