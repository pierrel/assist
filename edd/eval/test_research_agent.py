import tempfile, logging, sys
from textwrap import dedent
from unittest import TestCase

from assist.agent import create_research_agent, AgentHarness
from assist.model_manager import select_chat_model

from .utils import read_file, create_filesystem, files_in_directory

# debug logging by default
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.DEBUG)
logging.getLogger("openai").setLevel(logging.DEBUG)
logging.getLogger("langchain").setLevel(logging.DEBUG)
logging.getLogger("deepagents").setLevel(logging.DEBUG)


class TestResearchAgent(TestCase):
    def create_agent(self, filesystem: dict):
        root = tempfile.mkdtemp()
        create_filesystem(root, filesystem)
        
        return AgentHarness(create_research_agent(self.model,
                                                  root)), root
    
    def setUp(self):
        self.model = select_chat_model("gpt-oss-20b", 0.1)
        
    def test_follows_result_guidance(self):
        agent, root = self.create_agent({"reference": {"existing_research.org":"The capital of France is Paris"}})
        res = agent.message("What is langgraph? Place results into /reference/langgraph.org")
        self.assertIn("langgraph.org", res, "Should mention the resulting file")
        self.assertIn("langgraph.org", files_in_directory(f"{root}/reference"))

    def test_doesnt_leave_question(self):
        agent, root = self.create_agent({"references": {"existing_research.org":"The capital of France is Paris"}})
        agent.message("What is langgraph?")
        self.assertNotIn("question.txt", files_in_directory(root))

    def test_has_references_with_urls(self):
        agent, root = self.create_agent({"references": {"existing_research.org":"The capital of France is Paris"}})
        res = agent.message("What is langgraph? Place the result into a .org file in the references directory.")

        # Extract the filename from the response
        # The agent should return the path to the written file
        import re
        file_match = re.search(r'(\w+\.org)', res)
        self.assertIsNotNone(file_match, "Should return a filename in the response")

        filename = file_match.group(1)
        file_content = read_file(f"{root}/references/{filename}")

        # Check that the file has a Sources section
        self.assertRegex(file_content, "(?i)sources|reference", "Research output should have a Sources/Reference section")

        # Find the Sources/Reference section and extract all non-blank lines following it
        # up to the next fully blank line or end of file
        sources_regexp = r'(?i)(sources|references?)'
        sources_match = re.search(sources_regexp, file_content, re.MULTILINE)
        self.assertIsNotNone(sources_match, "Should have a sources or references section")
        # Get text after the Sources/Reference header
        text_after_sources = file_content[sources_match.end():]

        # Split into lines and process
        lines = text_after_sources.split('\n')
        source_lines = []

        # Collect non-blank lines until we hit a blank line or EOF
        for line in lines:
            stripped = line.strip()
            if not stripped:
                # Hit a blank line, stop collecting
                break
            source_lines.append(stripped)

        # Check that we have at least one source line
        self.assertGreater(len(source_lines), 0, "Should have at least one reference in Sources section")

        # Check that each line has a URL
        url_pattern = r'https?://[^\s]+'
        for line in source_lines:
            self.assertRegex(line, url_pattern, f"Each source line should contain a URL, found: {line}")
