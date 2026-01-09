import tempfile
import shutil
import uuid

from unittest import TestCase
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


from assist.model_manager import select_chat_model
from assist.deepagents_agent import deepagents_agent

from .utils2 import send_message


class TestDeepagentsAgent(TestCase):
    def setUp(self):
        self.working_dir = tempfile.mkdtemp()
        self.model = select_chat_model("gpt-oss-20b", 0.3)
        self.agent = deepagents_agent(self.model,
                                      checkpointer=InMemorySaver(),
                                      log_dir=self.working_dir)

    def tearDown(self):
        shutil.rmtree(self.working_dir)

    def test_basic_research(self):
        thread_id = uuid.uuid1()
        messages = send_message(self.agent,
                                "What is langgraph?",
                                thread_id)
        print(messages)
        self.assertTrue(messages)
                                      
