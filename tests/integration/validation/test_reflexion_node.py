from unittest import TestCase
from langchain_core.messages import HumanMessage, AIMessage

import assist.reflexion_agent as reflexion_agent
from assist.reflexion_agent import build_reflexion_graph

from .utils import (
    DummyLLM,
    DummyAgent,
    fake_general_agent,
    actual_llm,
    base_tools_for_test,
)


class TestReflexionNode(TestCase):
    def setUp(self):
        self.llm = DummyLLM(message="The capital of France is Paris.")
        self.orig_agent = reflexion_agent.general_agent
        reflexion_agent.general_agent = fake_general_agent
        self.graph = build_reflexion_graph(self.llm, [], [], self.llm)

    def tearDown(self):
        reflexion_agent.general_agent = self.orig_agent

    def test_reflexion_node(self):
        state = self.graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Identify the capital of France and provide one fact about it."
                    )
                ]
            }
        )
        message = state["messages"][-1]
        self.assertIsInstance(message, AIMessage)
        self.assertNotIn("ummary", message.content)
        self.assertIn("France", message.content)


def test_agent_summarizes_file_from_context(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSIST_SERVER_PROJECT_ROOT", "")
    llm = actual_llm()
    graph = build_reflexion_graph(llm, base_tools_for_test())

    file_path = tmp_path / "sample.txt"
    file_path.write_text("Rutabaga is a root vegetable.\nIt is nutritious.")

    messages = [
        HumanMessage(content="Where is the sample file?"),
        AIMessage(content=f"It is in file: {file_path}"),
        HumanMessage(content="Please summarize that file from before."),
    ]

    result = graph.invoke({"messages": messages})
    final_content = result["messages"][-1].content.lower()
    assert "rutabaga" in final_content

