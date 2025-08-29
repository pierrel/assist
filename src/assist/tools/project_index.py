from __future__ import annotations

import hashlib
import tempfile
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool
from vgrep.manager import Manager


class _DummyContextualizer:
    """Simple contextualizer that returns an empty string.

    The real ``vgrep`` library uses an LLM to provide additional context
    for each chunk.  For testing and lightweight usage we replace that
    behaviour with a no-op implementation so that no external services
    are required."""

    def contextualize(self, text: str, existing_context: str = "") -> str:  # pragma: no cover - trivial
        return ""


class _Retriever:
    """Wrap ``vgrep``'s ``Manager`` with the retriever interface used by tests."""

    def __init__(self, mgr: Manager) -> None:
        self._mgr = mgr

    def get_relevant_documents(self, query: str) -> List[Document]:
        results = self._mgr.query(query)
        return [Document(page_content=r["text"]) for r in results]

    def invoke(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)


class _DeterministicEmbedding:
    """Return reproducible pseudo-random vectors for texts.

    The embedding values are derived from a hash of each input string so that
    the same text will always produce the same vector without requiring network
    access or external models."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def __call__(self, input: List[str]) -> List[List[float]]:  # pragma: no cover - simple
        vectors: List[List[float]] = []
        for text in input:
            seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            vectors.append(rng.random(self._dim, dtype=np.float32).tolist())
        return vectors

    def name(self) -> str:  # pragma: no cover - simple
        return "deterministic"

    def is_legacy(self) -> bool:  # pragma: no cover - simple
        return True


class ProjectIndex:
    """Manage vector stores for arbitrary projects using ``vgrep``."""

    def __init__(self, base_dir: Path | str | None = None) -> None:
        self._retrievers: Dict[str, _Retriever] = {}
        self._base_dir = Path(base_dir) if base_dir is not None else Path(tempfile.gettempdir())

    def index_dir(self, project_root: Path) -> Path:
        """Return a unique directory for storing the vector index."""
        digest = hashlib.md5(str(project_root.resolve()).encode()).hexdigest()[:8]
        base = self._base_dir / "projects" / f"assist_index_{digest}"
        base.mkdir(parents=True, exist_ok=True)
        return base

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _create_manager(self, project_root: Path, index_dir: Path) -> Manager:
        embedding = _DeterministicEmbedding() if os.getenv("PYTEST_CURRENT_TEST") else None
        mgr = Manager(project_root, db_path=index_dir, embedding=embedding)
        # The default contextualizer uses an LLM; replace it to keep tests
        # lightweight and deterministic.
        mgr.db.contextualizer = _DummyContextualizer()
        return mgr

    def _build_index(self, project_root: Path, index_dir: Path) -> _Retriever:
        mgr = self._create_manager(project_root, index_dir)
        mgr.sync()
        return _Retriever(mgr)

    def _load_index(self, project_root: Path, index_dir: Path) -> _Retriever:
        mgr = self._create_manager(project_root, index_dir)
        return _Retriever(mgr)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_retriever(self, project_root: Path | str) -> _Retriever:
        """Return a retriever for ``project_root``."""
        root = Path(project_root)
        key = str(root.resolve())
        if key in self._retrievers:
            return self._retrievers[key]

        index_dir = self.index_dir(root)
        if any(index_dir.iterdir()):
            retriever = self._load_index(root, index_dir)
        else:
            retriever = self._build_index(root, index_dir)

        self._retrievers[key] = retriever
        return retriever

    def search(self, project_root: Path | str, query: str) -> str:
        retriever = self.get_retriever(project_root)
        docs = retriever.invoke(query)
        return "\n".join(doc.page_content for doc in docs)

    def search_tool(self) -> BaseTool:
        @tool
        def project_search(project_root: Path | str, query: str) -> str:
            """Search ``project_root`` for relevant information about the given ``query``.

            Args:
            ``project_root`` is a directory on the filesystem that at the top level of the project. The project contains information relevant to the user's current task.

            ``query`` is the query to be performed to learn more about the files in the project, which are vectorized for easy semantic search.

            Returns:
            str: A newline-separated list of file contents relevant to the query."""
            retriever = self.get_retriever(project_root)
            docs = retriever.invoke(query)
            return "\n".join(doc.page_content for doc in docs)

        return project_search


__all__ = ["ProjectIndex"]
