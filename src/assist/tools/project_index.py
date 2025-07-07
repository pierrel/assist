from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Dict, Optional

from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool


class ProjectIndex:
    """Manage vector stores for arbitrary projects."""

    def __init__(self, embedding: Optional[Embeddings] = None) -> None:
        self._embedding = embedding
        self._retrievers: Dict[str, Chroma] = {}

    def set_embedding(self, embedding: Optional[Embeddings]) -> None:
        """Set the embedding function used for new vector stores."""
        self._embedding = embedding
        self._retrievers = {}

    def index_dir(self, project_root: Path) -> Path:
        """Return a unique directory for storing the vector index."""
        digest = hashlib.md5(str(project_root.resolve()).encode()).hexdigest()[:8]
        return Path(tempfile.gettempdir()) / f"assist_index_{digest}"

    def build_vectorstore(self, project_root: Path, index_dir: Path) -> Chroma:
        """Create a Chroma vector store for ``project_root``."""
        loader = DirectoryLoader(
            str(project_root),
            glob="**/*.*",
            recursive=True,
            loader_cls=TextLoader,
            exclude=[
                "**/.git/**",
                "**/.venv/**",
                "**/__pycache__/**",
                str(index_dir),
            ],
        )
        index_creator = VectorstoreIndexCreator(
            vectorstore_cls=Chroma,
            embedding=self._embedding,
            vectorstore_kwargs={"persist_directory": str(index_dir)},
        )
        index = index_creator.from_loaders([loader])
        vectorstore = index.vectorstore
        vectorstore.persist()
        return vectorstore

    def load_vectorstore(self, index_dir: Path) -> Chroma:
        """Load the persisted Chroma vector store."""
        return Chroma(persist_directory=str(index_dir), embedding_function=self._embedding)

    def get_retriever(self, project_root: Path | str):
        """Return a retriever for ``project_root``."""
        root = Path(project_root)
        key = str(root.resolve())
        if key in self._retrievers:
            return self._retrievers[key]

        index_dir = self.index_dir(root)
        if index_dir.exists():
            vectorstore = self.load_vectorstore(index_dir)
        else:
            index_dir.mkdir(parents=True, exist_ok=True)
            vectorstore = self.build_vectorstore(root, index_dir)

        retriever = vectorstore.as_retriever()
        self._retrievers[key] = retriever
        return retriever

    def project_search(self, project_root: Path | str, query: str) -> str:
        """Search ``project_root`` for relevant information."""
        retriever = self.get_retriever(project_root)
        docs = retriever.get_relevant_documents(query)
        return "\n".join(doc.page_content for doc in docs)


_DEFAULT_INDEX = ProjectIndex()


def set_embedding(embedding: Optional[Embeddings]) -> None:
    _DEFAULT_INDEX.set_embedding(embedding)


def get_project_retriever(project_root: Path | str):
    return _DEFAULT_INDEX.get_retriever(project_root)


@tool
def project_search(project_root: Path | str, query: str) -> str:
    """Search ``project_root`` for relevant information."""
    return _DEFAULT_INDEX.project_search(project_root, query)


__all__ = ["ProjectIndex", "set_embedding", "get_project_retriever", "project_search"]
