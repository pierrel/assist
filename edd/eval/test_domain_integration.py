import os
import tempfile
import shutil

from unittest import TestCase
from langchain_core.messages import ToolMessage

from assist.thread import ThreadManager
from assist.domain_manager import DomainManager

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


class TestDomainIntegration(TestCase):
    """Integration tests for Thread + DomainManager working together."""

    def setUp(self):
        self.working_dir = tempfile.mkdtemp()
        self.thread_manager = ThreadManager(self.working_dir)

    def tearDown(self):
        if os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir)

    def test_domain_creation_with_thread(self):
        """Test that thread working directory is created and DomainManager can use it."""
        thread = self.thread_manager.new()
        # Verify thread's working directory exists
        self.assertTrue(os.path.isdir(thread.working_dir))

        # Create DomainManager for the thread's working directory
        dm = DomainManager(repo_path=thread.working_dir)
        self.assertTrue(os.path.isdir(dm.domain()))
        self.assertEqual(dm.domain(), thread.working_dir)

    def test_task_list_find(self):
        """Test finding task list in domain structure."""
        thread = self.thread_manager.new()
        dm = DomainManager(repo_path=thread.working_dir)
        create_structure(dm.domain())

        resp = thread.message("Where is my tasks list?")
        self.assertRegex(resp, "inbox\\.org", "Does not contain inbox.")

    def test_first_task_find(self):
        """Test finding first task in domain structure."""
        thread = self.thread_manager.new()
        dm = DomainManager(repo_path=thread.working_dir)
        create_structure(dm.domain())

        resp = thread.message("What is my next task?")
        self.assertRegex(resp, "Yosemite", "Does not contain Yosemite, the first task list.")

    def test_updates_task(self):
        """Test updating task in domain structure."""
        thread = self.thread_manager.new()
        dm = DomainManager(repo_path=thread.working_dir)
        create_structure(dm.domain())

        resp = thread.message("Update my next task to be due on 11/7/2026.")
        self.assertTrue(resp, "Should respond")

        inbox_path = os.path.join(dm.domain(), "gtd", "inbox.org")
        with open(inbox_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("2026-11-07", content, "Inbox should contain normalized due date")

    def test_find_pants(self):
        """Test research and task creation in domain structure."""
        message = "I need new pants. What are some good, custom made pants options. Provide both local (to san francisco) and over the internet. Also provide a price range."
        thread = self.thread_manager.new()
        dm = DomainManager(repo_path=thread.working_dir)
        create_structure(dm.domain())

        resp = thread.message(message)
        self.assertTrue(resp, "Should respond")

        inbox_path = os.path.join(dm.domain(), "gtd", "inbox.org")
        with open(inbox_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("pant", content, "Inbox should contain something about pants")

        tool_calls = [m.name for m in thread.get_raw_messages() if isinstance(m, ToolMessage)]
        self.assertIn("write_todos", tool_calls)
        self.assertIn("task", tool_calls)
