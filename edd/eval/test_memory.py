import os
import tempfile
import shutil
from textwrap import dedent

from unittest import TestCase

from assist.model_manager import select_chat_model
from assist.agent import create_agent, AgentHarness

from .utils import read_file, create_filesystem, AgentTestMixin

class TestMemory(AgentTestMixin, TestCase):
    def create_agent(self, filesystem: dict):
        root = tempfile.mkdtemp()
        create_filesystem(root, filesystem)

        return AgentHarness(create_agent(self.model,
                                         root)), root

    def setUp(self):
        self.model = select_chat_model(0.1)

    def test_reads_memory(self):
        agent, root = self.create_agent({"AGENTS.md": "I have 3 cats"})
        res = agent.message("How many cats do I have?")
        self.assertRegex(res, "3|three|Three", "Should correctly respond with the number 3")

    def test_doesnt_include_tags(self):
        agent, root = self.create_agent({"AGENTS.md": ""})
        agent.message("I have 3 cats. Commit this to memory.")
        memory_after = read_file(os.path.join(root, "AGENTS.md"))
        self.assertNotEqual(memory_after, "",
                            "Should have written something")
        self.assertNotRegex(memory_after, "<agent_memory>",
                            "Should not use the same format")

    def test_writes_memory_explicit(self):
        agent, root = self.create_agent({"AGENTS.md": ""})
        agent.message("I have 3 cats. Commit this to memory.")
        memory_after = read_file(os.path.join(root, "AGENTS.md"))
        self.assertRegex(memory_after, "cats",
                         "Should add the fact to memory")

    def test_writes_memory_implicit(self):
        agent, root = self.create_agent({"AGENTS.md": ""})
        agent.message("I have 3 cats")
        memory_after = read_file(os.path.join(root, "AGENTS.md"))
        self.assertRegex(memory_after, "cats",
                         "Should add the fact to memory")


