from unittest import TestCase
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from assist.reflexion_agent import build_summarize_node, StepResolution, ReflexionState

from .utils import DummyLLM, graphiphy


class TestSummarizerNode(TestCase):
    def setUp(self):
        llm = DummyLLM(message="Tea originated in China.")
        self.graph = graphiphy(build_summarize_node(llm, []))
        self.state: ReflexionState = {
            "messages": [
                SystemMessage("You are a helpful assistant"),
                HumanMessage("What's up with tea?"),
            ],
            "history": [
                StepResolution(
                    action="Greet", objective="Say hi", resolution="Hi there!"
                ),
                StepResolution(
                    action="Share fact",
                    objective="Inform user",
                    resolution="Tea originated in China.",
                ),
            ],
        }

    def test_summary_mentions_china(self):
        result = self.graph.invoke(self.state)
        out_message = result["messages"][-1]
        self.assertIsInstance(out_message, AIMessage)
        self.assertIn("china", out_message.content.lower())

