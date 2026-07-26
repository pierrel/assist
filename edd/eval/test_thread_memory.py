"""Natural behavior eval for repository versus thread memory scope."""
from __future__ import annotations

import os
import tempfile
from unittest import TestCase

from assist.agent import AgentHarness, create_agent
from assist.model_manager import select_assistant_model
from assist.spec import AgentSpec

from .utils import create_filesystem, read_file


class TestThreadMemory(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def test_infers_repo_and_thread_scope(self):
        root = tempfile.mkdtemp()
        agent_dir = tempfile.mkdtemp()
        create_filesystem(root, {"AGENTS.md": ""})
        graph = create_agent(
            self.model,
            root,
            agent_dir=agent_dir,
            spec=AgentSpec(async_subagent_tools=()),
        )
        agent = AgentHarness(graph, thread_id="thread-memory-eval")

        agent.message(
            "Across all conversations, use my label atlas-grid for compact "
            "tables. For this "
            "thread's application workflow, use the stage name "
            "candidate-replied after I answer.",
        )

        repo = read_file(os.path.join(root, "AGENTS.md"))
        memory_path = os.path.join(agent_dir, "memory.md")
        thread = read_file(memory_path) if os.path.exists(memory_path) else ""
        self.assertIn("atlas-grid", repo.lower(), agent.all_messages())
        self.assertNotIn("candidate-replied", repo)
        self.assertIn("candidate-replied", thread)
        self.assertNotIn("atlas-grid", thread.lower())
