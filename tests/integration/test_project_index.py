from unittest import TestCase
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.prebuilt import create_react_agent
from assist.tools import project_index
from langchain_core.messages import SystemMessage, HumanMessage
from tempfile import TemporaryDirectory, NamedTemporaryFile
from langchain_ollama import ChatOllama
from pathlib import Path


def setup_temp_files(contents: list[str]) -> TemporaryDirectory:
    """Sets up a series of files in a new temporary directory. Each
    file contains the contents of an element of ``contents``. Returns
    the path of the newly-created temporary directory"""
    tmpdir = TemporaryDirectory()
    for content in contents:
        file = NamedTemporaryFile(dir=tmpdir.name,
                                  suffix=".txt")
        Path(file.name).write_text(content)
    return tmpdir

class TestProjectIndex(TestCase):
    @classmethod
    def setUpClass(cls):
        # setup
        cls.pi = project_index.ProjectIndex()


    def test_retrieve_documents(self):
        tmp_file_contents = ["This is some text",
                             "Here is some longer text that I care about",
                             "Here is a todo list item: Take out the trash",
                             "This is a todo list item that needs to be done today: clean the kitchen"]

        # test that project search works by itself
        td = setup_temp_files(tmp_file_contents)
        res1 = self.pi.search(Path(td.name), "All of my todos")

    def test_agentic_retrieval(self):
        tmp_file_contents = ["This is some text",
                             "Here is some longer text that I care about",
                             "Here is a todo list item: Take out the trash",
                             "This is a todo list item that needs to be done today: clean the kitchen"]
        llm = ChatOllama(model="mistral", temperature=0.4)
        proj_tool = self.pi.search_tool()
        agent_executor = create_react_agent(llm,
                                            [proj_tool])
        res2 = None
        td = setup_temp_files(tmp_file_contents)
        project_root = Path(td.name)
        res2 = agent_executor.invoke({"messages": [
            SystemMessage(content=f'Check for files in {td}'),
            HumanMessage(content="What are all of the tasks that need to be done today?")
        ]})
