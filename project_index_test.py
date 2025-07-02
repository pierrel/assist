import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from tools import project_index
from langchain_community.embeddings import FakeEmbeddings


class TestProjectIndex(TestCase):
    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.project_root = Path(self.tmpdir.name)
        (self.project_root / "sub").mkdir()
        (self.project_root / "sub/file1.txt").write_text("hello world")
        (self.project_root / "file2.txt").write_text("foo bar baz")

        # patch project_index globals
        self.orig_root = project_index.PROJECT_ROOT
        self.orig_index_dir = project_index.INDEX_DIR
        project_index.PROJECT_ROOT = self.project_root
        project_index.INDEX_DIR = self.project_root / "index_store"
        project_index.set_embedding(FakeEmbeddings(size=4))
        project_index._retriever = None

    def tearDown(self):
        project_index.PROJECT_ROOT = self.orig_root
        project_index.INDEX_DIR = self.orig_index_dir
        project_index.set_embedding(None)
        project_index._retriever = None
        self.tmpdir.cleanup()

    def test_index_and_search(self):
        retriever = project_index.get_project_retriever()
        docs = retriever.get_relevant_documents("hello")
        joined = "\n".join(d.page_content for d in docs)
        self.assertIn("hello world", joined)

        docs2 = retriever.get_relevant_documents("foo bar")
        joined2 = "\n".join(d.page_content for d in docs2)
        self.assertIn("foo bar baz", joined2)
