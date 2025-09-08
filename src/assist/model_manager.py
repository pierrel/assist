"""Utilities for managing chat models for the server.

This module encapsulates the logic for selecting chat models and mapping
planning models to their corresponding execution models.
"""
from __future__ import annotations

from typing import Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
try:  # pragma: no cover - optional dependency
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore

# Mapping of planning models to their corresponding execution models
MODEL_EXECUTION_MAP: dict[str, str] = {
    "gpt-4o": "gpt-4o-mini",
    "gpt-4o-mini": "gpt-4o-mini",
}


def select_chat_model(model: str, temperature: float) -> BaseChatModel:
    """Return the appropriate chat model for ``model``.

    If the model string indicates a ChatGPT model (``gpt-*``) a ``ChatOpenAI``
    instance is returned, otherwise a ``ChatOllama`` instance is used.
    """
    if model.startswith("gpt-"):
        if ChatOpenAI is None:  # pragma: no cover - environment dependent
            raise RuntimeError("ChatOpenAI is not available")
        return ChatOpenAI(model=model, temperature=temperature)
    return ChatOllama(model=model, temperature=temperature)


def get_model_pair(model: str, temperature: float) -> Tuple[BaseChatModel, BaseChatModel]:
    """Return a pair of (planning_llm, execution_llm) for ``model``.

    The planning model is always ``model`` and the execution model is looked up
    in ``MODEL_EXECUTION_MAP``. If there is no mapping, the planning model is
    also used for execution.
    """
    plan_llm = select_chat_model(model, temperature)
    exec_model = MODEL_EXECUTION_MAP.get(model, model)
    exec_llm = select_chat_model(exec_model, temperature)
    return plan_llm, exec_llm
