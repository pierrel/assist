from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Optional, Dict

from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool


_retrievers: Dict[str, Chroma] = {}
_EMBEDDING: Optional[Embeddings] = None


def set_embedding(embedding: Optional[Embeddings]) -> None:
    """Set the embedding function used for the vector store."""
    global _EMBEDDING, _retrievers
    _EMBEDDING = embedding
    _retrievers = {}


def _index_dir(project_root: Path) -> Path:
    """Return a unique directory for storing the vector index."""
    digest = hashlib.md5(str(project_root.resolve()).encode()).hexdigest()[:8]
    return Path(tempfile.gettempdir()) / f"assist_index_{digest}"


def _build_vectorstore(project_root: Path, index_dir: Path) -> Chroma:
    """Create a Chroma vector store for the project."""
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
        embedding=_EMBEDDING,
        vectorstore_kwargs={"persist_directory": str(index_dir)},
    )
    index = index_creator.from_loaders([loader])
    vectorstore = index.vectorstore
    vectorstore.persist()
    return vectorstore


def _load_vectorstore(index_dir: Path) -> Chroma:
    """Load the persisted Chroma vector store."""
    return Chroma(persist_directory=str(index_dir), embedding_function=_EMBEDDING)


def get_project_retriever(project_root: Path | str):
    """Return a retriever over ``project_root``."""
    global _retrievers
    root = Path(project_root)
    key = str(root.resolve())
    if key in _retrievers:
        return _retrievers[key]

    index_dir = _index_dir(root)
    if index_dir.exists():
        vectorstore = _load_vectorstore(index_dir)
    else:
        index_dir.mkdir(parents=True, exist_ok=True)
        vectorstore = _build_vectorstore(root, index_dir)

    retriever = vectorstore.as_retriever()
    _retrievers[key] = retriever
    return retriever


@tool
def project_search(project_root: Path | str, query: str) -> str:
    """Search ``project_root`` for relevant information."""
    retriever = get_project_retriever(project_root)
    docs = retriever.get_relevant_documents(query)
    return "\n".join(doc.page_content for doc in docs)


__all__ = ["set_embedding", "get_project_retriever", "project_search"]
