import os
import tempfile
import shutil

from unittest import TestCase

from assist.agent import create_user_expert_agent, AgentHarness

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from assist.model_manager import select_chat_model

def read_file(path: str):
    """Returns the full contents of file at path"""
    with open(path, 'r') as f:
        return f.read()

def create_filesystem(root_dir: str,
                      structure: dict):
    """Creates a directory structure and files according to `structure`. For example:
    {"README.org": "This is the readme file",
    "gtd": {"inbox.org": "This is the inbox file"},
           {"projects": {"project1.org": "This is a project file"}}}

    Creates:
    a README.org file with content "This is the readme file"
    a gtd directory
    a gtd/inbox.org file with content "This is the inbox file"
    ..."""
    for name, content in structure.items():
        path = os.path.join(root_dir, name)

        if isinstance(content, str):
            # Create a file with the given content
            with open(path, 'w') as f:
                f.write(content)
        elif isinstance(content, dict):
            # Create a directory and recursively process its contents
            os.makedirs(path, exist_ok=True)
            create_filesystem(path, content)

class TestUserExpertAgent(TestCase):
    def create_agent(self, filesystem: dict):
        root = tempfile.mkdtemp()
        create_filesystem(root, filesystem)
                           
        return AgentHarness(create_user_expert_agent(self.model,
                                                     root))
    
    def setUp(self):
        self.model = select_chat_model("gpt-oss-20b", 0.1)

    def test_reads_readme(self):
        root = tempfile.mkdtemp()
        agent = self.create_agent({"README.org": "All of my todos are in gtd/inbox.org",
                                   "gtd": {"inbox.org":
                                           """* Tasks
                                           ** TODO Fold laundry
                                           Just get it done
                                           ** TODO Buy new pants"""}})
        res = agent.message("Where are my todos?")
        self.assertRegex(res, "inbox\\.org", "Should mention the inbox file")

    def test_adds_item_correctly(self):
        root = tempfile.mkdtemp()
        agent = self.create_agent({"README.org": "All of my todos are in gtd/inbox.org",
                                   "gtd": {"inbox.org":
                                           """* Tasks
                                           ** TODO Fold laundry
                                           Just get it done
                                           ** TODO Buy new pants
                                           Size 31"""}})
        inbox_contents_before = read_file(f"{root}/gtd/inbox.org")
        res = agent.message("I need a new washer/dryer")
        inbox_contents = read_file(f"{root}/gtd/inbox.org")
        self.assertRegex(res, "updated", "Should mention that a change was made.")
        self.assertRegex(inbox_contents,
                         "^\\*\\* TODO.*dryer",
                         "Should have added a TODO with dryer in the heading")
        self.assertRegex(inbox_contents,
                         "laundry\nJust get it done",
                         "Should not have split a TODO item")
        self.assertRegex(inbox_contents,
                         "pants\nSize",
                         "Should not have split a TODO item")

