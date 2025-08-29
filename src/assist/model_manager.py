"""Model selection utilities for planner/executor separation."""
from __future__ import annotations

import subprocess
from typing import Dict, List, Tuple

from langchain_core.runnables import Runnable


class ModelManager:
    """Resolve appropriate planner and executor LLMs.

    The planner model is used for planning, plan checking and summarization,
    while the executor model handles individual step execution.
    """

    #: Mapping of planning model -> execution model
    PLAN_EXECUTION_MAP: Dict[str, str] = {
        "gpt-4o": "gpt-4o-mini",
        "gpt-4o-mini": "gpt-4o-mini",
        "llama3.2:8b": "llama3.2:3b",
    }

    def __init__(self) -> None:
        self.ollama_models = self._load_ollama_models()

    def _load_ollama_models(self) -> List[str]:
        """Return list of installed Ollama models.

        The function is conservative: any failure results in an empty list so
        that callers can still proceed in environments without Ollama.
        """
        try:
            out = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, check=True
            )
            # Skip the header line and grab model names (first column)
            lines = [l.strip().split()[0] for l in out.stdout.splitlines()[1:] if l.strip()]
            return lines
        except Exception:  # pragma: no cover - best effort only
            return []

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_openai_model(model: str) -> bool:
        return model.startswith("gpt")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_llms(self, model: str, temperature: float) -> Tuple[Runnable, Runnable]:
        """Return ``(planner_llm, executor_llm)`` for ``model``.

        ``planner_llm`` mirrors the requested model exactly, while
        ``executor_llm`` is looked up via :pyattr:`PLAN_EXECUTION_MAP`.
        """
        exec_model = self.PLAN_EXECUTION_MAP.get(model, model)

        if self._is_openai_model(model):
            from langchain_openai import ChatOpenAI

            planner_llm = ChatOpenAI(model=model, temperature=temperature)
            executor_llm = ChatOpenAI(model=exec_model, temperature=temperature)
        else:
            from langchain_ollama import ChatOllama

            planner_llm = ChatOllama(model=model, temperature=temperature)
            if exec_model not in self.ollama_models:
                exec_model = model
            executor_llm = ChatOllama(model=exec_model, temperature=temperature)

        return planner_llm, executor_llm
