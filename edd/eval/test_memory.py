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

    def test_writes_memory_explicit(self):
        agent, root = self.create_agent({"AGENTS.md": ""})
        agent.message("I have 3 cats. Commit this to memory.")
        memory_after = read_file(os.path.join(root, "AGENTS.md"))
        self.assertRegex(memory_after, "cats",
                         "Should add the fact to memory")
        self.assertNotRegex(memory_after, "<agent_memory>",
                            "Should not write the literal <agent_memory> framing tag into the file")

    def test_writes_memory_implicit(self):
        agent, root = self.create_agent({"AGENTS.md": ""})
        agent.message("I have 3 cats")
        memory_after = read_file(os.path.join(root, "AGENTS.md"))
        self.assertRegex(memory_after, "cats",
                         "Should add the fact to memory")

    def test_writes_memory_explicit_feedback(self):
        """Multi-turn: user gives explicit forward-looking feedback.

        Turn 1 is a benign action; turn 2 gives the agent a rule
        ("in the future ...") that should land in memory as a
        persistent preference, not be acknowledged in prose only.
        """
        agent, root = self.create_agent({"AGENTS.md": ""})
        agent.message("Show me a quick hello world.")
        agent.message(
            "In the future, write all code examples in Python."
        )
        memory_after = read_file(os.path.join(root, "AGENTS.md"))
        self.assertRegex(
            memory_after, "Python",
            "Should capture the future-tense feedback as a "
            "persistent preference."
        )

    def test_writes_memory_two_consecutive_turns(self):
        """Both turns add a fact; the second turn must see the first.

        Exercises the cross-turn freshness contract: after the model
        writes via ``edit_file``/``write_file`` on turn 1, turn 2's
        rendered ``<agent_memory>`` block must reflect the new content
        — otherwise the second write's anchor mismatches and the fact
        is lost (or duplicated).
        """
        agent, root = self.create_agent({"AGENTS.md": ""})
        agent.message("I have 3 cats.")
        agent.message("I also have 2 dogs.")
        memory_after = read_file(os.path.join(root, "AGENTS.md"))
        self.assertRegex(memory_after, "cats",
                         "Turn 1's fact must survive into the final file")
        self.assertRegex(memory_after, "dogs",
                         "Turn 2's fact must be persisted, not lost to a "
                         "stale anchor mismatch")

    def test_writes_memory_preserves_existing(self):
        """Append, do not overwrite, when AGENTS.md already has content.

        The save path is now `edit_file` (not a closure-bound append
        tool), so the most plausible failure mode is the model
        clobbering existing memory with just the new fact.  This test
        seeds the file with two prior facts and expects all three
        survive after the write.
        """
        seed = "User has 3 cats.\nUser prefers Python.\n"
        agent, root = self.create_agent({"AGENTS.md": seed})
        agent.message("I also have 2 dogs. Commit this to memory.")
        memory_after = read_file(os.path.join(root, "AGENTS.md"))
        self.assertRegex(memory_after, "cats",
                         "Existing fact about cats should survive")
        self.assertRegex(memory_after, "Python",
                         "Existing fact about Python should survive")
        self.assertRegex(memory_after, "dogs",
                         "New fact about dogs should be appended")


