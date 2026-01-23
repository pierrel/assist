import os
import tempfile
import shutil

from unittest import TestCase

from assist.agent import create_user_expert_agent, AgentHarness

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from assist.model_manager import select_chat_model

from .utils import read_file, create_filesystem

class TestUserExpertAgent(TestCase):
    def create_agent(self, filesystem: dict):
        root = tempfile.mkdtemp()
        create_filesystem(root, filesystem)

        return AgentHarness(create_user_expert_agent(self.model,
                                                     root)), root
    
    def setUp(self):
        self.model = select_chat_model("gpt-oss-20b", 0.1)

    def test_reads_readme(self):
        agent, root = self.create_agent({"README.org": "All of my todos are in gtd/inbox.org",
                                         "gtd": {"inbox.org":
                                                 """* Tasks
                                           ** TODO Fold laundry
                                           Just get it done
                                           ** TODO Buy new pants"""}})
        res = agent.message("Where are my todos?")
        self.assertRegex(res, "inbox\\.org", "Should mention the inbox file")

    def test_adds_item_correctly(self):
        agent, root = self.create_agent({"README.org": "All of my todos are in gtd/inbox.org",
                                         "gtd": {"inbox.org":
                                                 "* Tasks\n** TODO Fold laundry\nJust get it done\n** TODO Buy new pants\nSize 31"}})
        inbox_contents_before = read_file(f"{root}/gtd/inbox.org")
        res = agent.message("I need a new washer/dryer")
        inbox_contents = read_file(f"{root}/gtd/inbox.org")
        self.assertRegex(res, "(?i)updated|added", "Should mention that a change was made.")
        self.assertRegex(inbox_contents,
                         "(?im)^\\*\\* TODO.*dryer",
                         "Should have added a TODO with dryer in the heading")
        self.assertRegex(inbox_contents,
                         "laundry\nJust get it done",
                         "Should not have split a TODO item")
        self.assertRegex(inbox_contents,
                         "pants\nSize",
                         "Should not have split a TODO item")

