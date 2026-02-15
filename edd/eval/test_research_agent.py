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
        files = files_in_directory(f"{root}/reference")
        self.assertIn("langgraph.org", files, "Should create the requested file")
        # The response should mention the file, but the agent may also return
        # the filename via the last tool call rather than in text content
        if res:
            self.assertIn("langgraph.org", res, "Should mention the resulting file")

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

        # Find the Sources/References section header (must be at start of line,
        # optionally preceded by org-mode stars or markdown hashes)
        sources_header = re.search(
            r'^[\s*#]*\b(Sources|References?)\b',
            file_content,
            re.MULTILINE | re.IGNORECASE,
        )
        self.assertIsNotNone(sources_header, "Research output should have a Sources/References section header")

        # Get text after the header line
        text_after_header = file_content[sources_header.end():]

        # Collect all URLs from the sources section (up to the next blank line
        # or next header)
        url_pattern = re.compile(r'https?://[^\s\]\)>]+')
        header_pattern = re.compile(r'^[\s*#]*\b[A-Z]', re.MULTILINE)

        urls = []
        for line in text_after_header.split('\n'):
            stripped = line.strip()
            if not stripped:
                if urls:  # stop at blank line after we've found URLs
                    break
                continue  # skip leading blank lines
            # Stop if we hit another section header
            if urls and header_pattern.match(line):
                break
            found = url_pattern.findall(stripped)
            urls.extend(found)

        self.assertGreater(len(urls), 0, "Sources section should contain at least one URL")
