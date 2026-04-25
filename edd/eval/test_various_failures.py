import os
import pdb
import tempfile
import shutil
from textwrap import dedent

from unittest import TestCase

from assist.model_manager import select_chat_model
from assist.agent import create_agent, AgentHarness

from .utils import read_file, create_filesystem, AgentTestMixin

class TestVariousFailures(AgentTestMixin, TestCase):
    def create_agent(self, filesystem: dict = {}):
        root = tempfile.mkdtemp()
        create_filesystem(root, filesystem)

        return AgentHarness(create_agent(self.model,
                                         root)), root

    def setUp(self):
        self.model = select_chat_model(0.1)

    def test_reads_memory(self):
        agent, root = self.create_agent()
        res = agent.message("I’m going swimming right now after about a week of being outside of the pool. Can you research good workouts that are around 2000 yards and good to “getting back in”? Recommend one for me.")
        self.assertNotEqual(res, "", "Should respond")
