import tempfile
from textwrap import dedent

from unittest import TestCase

from assist.agent import create_research_agent, AgentHarness

from assist.model_manager import select_chat_model

from .utils import read_file, create_filesystem, files_in_directory

class TestResearchAgent(TestCase):
    def create_agent(self, filesystem: dict):
        root = tempfile.mkdtemp()
        create_filesystem(root, filesystem)
        
        return AgentHarness(create_research_agent(self.model,
                                                  root)), root
    
    def setUp(self):
        self.model = select_chat_model("gpt-oss-20b", 0.1)
        
    def test_follows_result_guidance(self):
        agent, root = self.create_agent({"references": {"existing_research.org":"The capital of France is Paris"}})
        res = agent.message("Research the history of Paris. Place the result in references/paris_history.org")
        self.assertIn("paris_history.org", res, "Should mention the resulting file")
        self.assertIn("paris_history.org", files_in_directory(f"{root}/references"))

    def test_doesnt_leave_question(self):
        agent, root = self.create_agent({"references": {"existing_research.org":"The capital of France is Paris"}})
        res = agent.message("Research the history of Paris. Place the result in references/paris_history.org")
        self.assertNotIn("question.txt", files_in_directory(root))
