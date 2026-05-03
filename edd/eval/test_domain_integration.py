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

    def test_finds_and_updates_task(self):
        """Find the next task by name, then update its due date.

        Combines two assertions in one Thread+Domain run:
        (a) the agent surfaces "Yosemite" as the next task, and
        (b) the agent normalizes "11/7/2026" to ISO and writes it back.
        """
        thread = self.thread_manager.new()
        dm = DomainManager(repo_path=thread.working_dir)
        create_structure(dm.domain())

        resp = thread.message(
            "What is my next task? Then update it to be due on 11/7/2026."
        )
        self.assertRegex(resp, "Yosemite",
                         "Should surface Yosemite as the next task")

        inbox_path = os.path.join(dm.domain(), "gtd", "inbox.org")
        with open(inbox_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("2026-11-07", content,
                      "Inbox should contain normalized due date")
