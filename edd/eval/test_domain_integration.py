import os
import tempfile
import shutil

from unittest import TestCase
from unittest.mock import patch
from langchain_core.messages import ToolMessage

from assist.agent import AgentHarness, create_agent
from assist.thread_manager import ThreadManager
from assist.domain_manager import DomainManager
from assist.model_manager import select_assistant_model

from .test_async_subagents import reset_task_fixture
from .utils import complete_web_main_tasks, prompt_rewrite_web_main_spec

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


class TestPromptRewriteDomainTask(TestCase):
    """Compare the natural task journey in the exact prompt-rewrite web shape.

    ThreadManager deliberately always constructs the production web-main prompt,
    so its integration test cannot provide the legacy side of this comparison.
    This fixture retains the same workspace and natural user request while the
    shared prompt-rewrite helper switches only the static prompt composition.
    """

    def setUp(self):
        self.working_dir = tempfile.mkdtemp()
        self.agent_dir = tempfile.mkdtemp()
        create_structure(self.working_dir)
        reset_task_fixture()
        self.model = select_assistant_model(0.1)

    def tearDown(self):
        shutil.rmtree(self.working_dir, ignore_errors=True)
        shutil.rmtree(self.agent_dir, ignore_errors=True)

    def test_finds_and_updates_task_with_web_main_prompt(self):
        with patch("assist.tools.requests.get",
                   side_effect=AssertionError("task eval must not fetch URLs")) as get:
            agent = AgentHarness(create_agent(
                self.model, self.working_dir, agent_dir=self.agent_dir,
                spec=prompt_rewrite_web_main_spec()))
            agent.message(
                "What is my next task? Then update it to be due on 11/7/2026."
            )
            response = complete_web_main_tasks(agent)
        get.assert_not_called()

        self.assertRegex(response, "Yosemite",
                         "Should surface Yosemite as the next task")
        with open(os.path.join(self.working_dir, "gtd", "inbox.org"),
                  encoding="utf-8") as inbox:
            self.assertIn("2026-11-07", inbox.read(),
                          "Inbox should contain normalized due date")
