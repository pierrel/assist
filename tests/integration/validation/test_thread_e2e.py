import os
import tempfile
import shutil
import uuid

from unittest import TestCase
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


from assist.model_manager import select_chat_model
from assist.agent import create_agent
from assist.thread import ThreadManager, Thread

def create_structure(root: str):
    os.makedirs(root, exist_ok=True)
    # README.org in root
    readme_path = os.path.join(root, "README.org")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("Main tasks are managed in the gtd directory. inbox.org contains the tasks")

    # gtd directory with inbox.org
    gtd_dir = os.path.join(root, "gtd")
    os.makedirs(gtd_dir, exist_ok=True)

    inbox_path = os.path.join(gtd_dir, "inbox.org")
    inbox_content = (
        "* Tasks\n"
        "** TODO Plan Yosemite vacation\n"
        "See https://www.nationalparkreservations.com/park/yosemite-national-park/?msclkid=0d80374b168f1b298d7b5e249ba16b5f\n"
        "** TODO Take out trash\n"
    )
    with open(inbox_path, "w", encoding="utf-8") as f:
        f.write(inbox_content)


class TestThreadE2E(TestCase):
    def setUp(self):
        self.working_dir = tempfile.mkdtemp()
        self.thread_manager = ThreadManager(self.working_dir)
        


    def teardown(self):
        shutil.rmtree(self.working_dir)


    def test_domain_creation(self):
        thread = self.thread_manager.new()
        self.assertTrue(os.path.isdir(thread.domain_manager.domain()))

    def test_task_list_find(self):
        thread = self.thread_manager.new()
        create_structure(thread.domain_manager.domain())
        resp = thread.message("Where is my tasks list?")
        self.assertRegex(resp, "inbox\\.org", "Does not contain inbox.")

    def test_first_task_find(self):
        thread = self.thread_manager.new()
        create_structure(thread.domain_manager.domain())
        resp = thread.message("What is my next task?")
        self.assertRegex(resp, "Yosemite", "Does not contain Yosemite, the first task list.")
        
    def test_updates_task(self):
        thread = self.thread_manager.new()
        create_structure(thread.domain_manager.domain())
        resp = thread.message("Update my next task to be due on 11/7/2026.")
        self.assertTrue(resp, "Should respond")
        inbox_path = os.path.join(thread.domain_manager.domain(), "gtd", "inbox.org")
        with open(inbox_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("2026-11-07", content, "Inbox should contain normalized due date")
