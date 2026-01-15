import tempfile
import shutil
import uuid

from unittest import TestCase
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


from assist.model_manager import select_chat_model
from assist.agent import create

from .utils2 import send_message


class TestAgent(TestCase):
    def setUp(self):
        self.working_dir = tempfile.mkdtemp()
        self.model = select_chat_model("gpt-oss-20b", 0.3)
        self.agent = create(self.model,
                            checkpointer=InMemorySaver(),
                            log_dir=self.working_dir)


    def tearDown(self):
        shutil.rmtree(self.working_dir)

        
    def test_basic_research(self):
        thread_id = uuid.uuid1()
        messages = send_message(self.agent,
                                "What is langgraph?",
                                thread_id)
        state = self.agent.get_state({"configurable": {"thread_id": thread_id}})
        report_content = state.values.get("files", {}).get("/final_report.md", {}).get("content", None)
        self.assertTrue(messages)
        self.assertTrue(report_content)


    def test_hard_research(self):
        thread_id = uuid.uuid1()
        messages = send_message(self.agent,
                                "In emacs, I want to create a buffer in a window with ui controls that, when clicked, do not move the focus to that window (it should stay in the buffer/window its in). How do I achieve this?",
                                thread_id)
        self.assertTrue(messages)
        state = self.agent.get_state({"configurable": {"thread_id": thread_id}})
        report_content = state.values.get("files", {}).get("/final_report.md", {}).get("content", None)
        self.assertTrue(messages)
        self.assertTrue(report_content)

    def test_backpack_search(self):
        thread_id = uuid.uuid1()
        messages = send_message(self.agent,
                                "I’m in the market for a new backpack. I want something that will basically last forever and preferably vegan-friendly. It should also be very sleek, understated, and minimal. It should hold things like a laptop (sometimes 2 13” laptops), notebook, sweater, and have a pocket for smaller things. What I have right now is the Luis Vuitton Taiga Leather Antón, which lasted me almost 8 years. Only consider brands from the USA and preferably made in the USA. Provide at least one option that fits this description from a San Francisco brand (or more if they fit the description).",
                                thread_id)
        self.assertTrue(messages)
        state = self.agent.get_state({"configurable": {"thread_id": thread_id}})
        report_content = state.values.get("files", {}).get("/final_report.md", {}).get("content", None)
        self.assertTrue(messages)
        self.assertTrue(report_content)
    
