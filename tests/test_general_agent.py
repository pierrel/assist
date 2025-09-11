from unittest import TestCase
from typing import Union
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
import os
from .utils import make_test_agent
from assist.general_agent import _guard_tool, TRUNCATE_MSG, general_agent, CONTEXT_BUFFER_RATIO
from langchain_core.tools import tool
from unittest.mock import patch

AnyMessage = Union[HumanMessage, AIMessage, ToolMessage]

def tool_messages(resp_messages: AnyMessage) -> list[ToolMessage]:
    return [x for x in resp_messages if isinstance(x, ToolMessage)]


class TestGeneralAgent(TestCase):
    @classmethod
    def setUpClass(cls):
        current_file = os.path.basename(__file__)
        responses = [
            [
                ToolMessage(content=f"{current_file}", tool_call_id="1"),
                AIMessage(content=f"Files include {current_file}")
            ],
            [
                ToolMessage(content="listing", tool_call_id="1"),
                ToolMessage(content="file contents", tool_call_id="2"),
                AIMessage(content="The show_file_contents tool reads files")
            ],
        ]
        cls.agent = make_test_agent(responses)

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


def test_guard_tool_truncates_output():
    @tool
    def long_tool() -> str:
        """Return a big string"""
        return "x" * 120

    guarded = _guard_tool(long_tool, 50)
    out = guarded.invoke({})
    assert out.endswith(TRUNCATE_MSG)
    assert len(out) <= 50 + len(TRUNCATE_MSG)


def test_general_agent_uses_model_manager_limit():
    @tool
    def dummy() -> str:
        """Return a short string"""
        return "ok"

    with patch("assist.general_agent._guard_tool", side_effect=lambda t, l: t) as guard, \
            patch("assist.general_agent.create_react_agent") as creator, \
            patch("assist.general_agent.get_context_limit", return_value=55):
        general_agent(object(), [dummy])
        guard.assert_called_once()
        expected = int(55 * CONTEXT_BUFFER_RATIO) - len(TRUNCATE_MSG)
        assert guard.call_args.args[1] == expected
        creator.assert_called_once()
