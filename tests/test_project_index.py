import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from assist.tools import project_index
from langchain_community.embeddings import FakeEmbeddings


class TestProjectIndex(TestCase):
    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.project_root = Path(self.tmpdir.name)
        (self.project_root / "sub").mkdir()
        (self.project_root / "sub/file1.txt").write_text("hello world")
        (self.project_root / "file2.txt").write_text("foo bar baz")

        project_index.set_embedding(FakeEmbeddings(size=4))
        project_index._retrievers = {}

    def tearDown(self):
        project_index.set_embedding(None)
        project_index._retrievers = {}
        self.tmpdir.cleanup()

    def test_index_and_search(self):
        retriever = project_index.get_project_retriever(self.project_root)
        docs = retriever.get_relevant_documents("hello")
        joined = "\n".join(d.page_content for d in docs)
        self.assertIn("hello world", joined)

        docs2 = retriever.get_relevant_documents("foo bar")
        joined2 = "\n".join(d.page_content for d in docs2)
        self.assertIn("foo bar baz", joined2)

