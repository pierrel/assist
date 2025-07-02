from __future__ import annotations
from pathlib import Path

from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "index_store"

_retriever = None
_EMBEDDING: Optional[Embeddings] = None


def set_embedding(embedding: Optional[Embeddings]) -> None:
    """Set the embedding function used for the vector store."""
    global _EMBEDDING, _retriever
    _EMBEDDING = embedding
    _retriever = None


def _build_vectorstore() -> Chroma:
    """Create a Chroma vector store for the project."""
    loader = DirectoryLoader(
        str(PROJECT_ROOT),
        glob="**/*.*",
        recursive=True,
        loader_cls=TextLoader,
        exclude=["**/.git/**", str(INDEX_DIR)],
    )
    index_creator = VectorstoreIndexCreator(
        vectorstore_cls=Chroma,
        embedding=_EMBEDDING,
        vectorstore_kwargs={"persist_directory": str(INDEX_DIR)},
    )
    index = index_creator.from_loaders([loader])
    vectorstore = index.vectorstore
    vectorstore.persist()
    return vectorstore


def _load_vectorstore() -> Chroma:
    """Load the persisted Chroma vector store."""
    return Chroma(persist_directory=str(INDEX_DIR), embedding_function=_EMBEDDING)


def get_project_retriever():
    """Return a retriever over the current project."""
    global _retriever
    if _retriever is not None:
        return _retriever

    if INDEX_DIR.exists():
        vectorstore = _load_vectorstore()
    else:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore = _build_vectorstore()

    _retriever = vectorstore.as_retriever()
    return _retriever


@tool
def project_search(query: str) -> str:
    """Search the current project files for relevant information."""
    retriever = get_project_retriever()
    docs = retriever.get_relevant_documents(query)
    return "\n".join(doc.page_content for doc in docs)
