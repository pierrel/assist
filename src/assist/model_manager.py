"""Model selection utilities for planner/executor separation."""
from __future__ import annotations

import subprocess
import re
from typing import List, Tuple

from langchain_core.runnables import Runnable


class ModelManager:
    """Resolve appropriate planner and executor LLMs.

    The planner model is used for planning, plan checking and summarization,
    while the executor model handles individual step execution.
    """

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
    # OpenAI handling
    # ------------------------------------------------------------------
    @staticmethod
    def _openai_step_model(model: str) -> str:
        """Return a cheaper OpenAI model similar to ``model``.

        Current heuristic simply selects the ``-mini`` variant when available.
        """
        return model if model.endswith("-mini") else f"{model}-mini"

    # ------------------------------------------------------------------
    # Ollama handling
    # ------------------------------------------------------------------
    def _ollama_step_model(self, model: str) -> str:
        family, size = model.split(":", 1) if ":" in model else (model, "")
        family_models = [m for m in self.ollama_models if m.startswith(family + ":")]
        if not family_models:
            return model

        def size_value(m: str) -> float:
            s = m.split(":", 1)[1]
            match = re.search(r"[0-9]+(?:\.[0-9]+)?", s)
            return float(match.group(0)) if match else 0

        family_models.sort(key=size_value)
        sizes = [m.split(":", 1)[1] for m in family_models]
        if size in sizes:
            idx = sizes.index(size)
            if idx > 0:
                return f"{family}:{sizes[idx - 1]}"
            else:
                return model
        else:
            return f"{family}:{sizes[0]}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_llms(self, model: str, temperature: float) -> Tuple[Runnable, Runnable]:
        """Return ``(planner_llm, executor_llm)`` for ``model``.

        ``planner_llm`` mirrors the requested model exactly, while
        ``executor_llm`` is a cheaper sibling model.
        """
        if self._is_openai_model(model):
            from langchain_openai import ChatOpenAI

            planner_llm = ChatOpenAI(model=model, temperature=temperature)
            exec_model = self._openai_step_model(model)
            executor_llm = ChatOpenAI(model=exec_model, temperature=temperature)
        else:
            from langchain_ollama import ChatOllama

            planner_llm = ChatOllama(model=model, temperature=temperature)
            exec_model = self._ollama_step_model(model)
            executor_llm = ChatOllama(model=exec_model, temperature=temperature)

        return planner_llm, executor_llm
