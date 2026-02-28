"""Utilities for managing chat models for the server.

This module encapsulates the logic for selecting chat models. The default
configuration uses the public OpenAI API with ``gpt-4o-mini``. Configuration
can be provided via environment variables.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

try:  # pragma: no cover - optional dependency
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenAIConfig:
    """Configuration for a custom OpenAI-compatible endpoint."""

    url: str
    model: str
    api_key: str
    context_len: int


@lru_cache(maxsize=1)
def _load_custom_openai_config() -> Optional[OpenAIConfig]:
    """Return the custom OpenAI configuration from environment variables.

    Required environment variables:
    - ASSIST_MODEL_URL: OpenAI-compatible API endpoint
    - ASSIST_MODEL_NAME: Model identifier
    - ASSIST_API_KEY: API key for authentication

    Optional environment variables:
    - ASSIST_CONTEXT_LEN: Context window size (default: 32768)

    Returns ``None`` if any required variable is missing.
    """
    url = os.getenv("ASSIST_MODEL_URL")
    model = os.getenv("ASSIST_MODEL_NAME")
    api_key = os.getenv("ASSIST_API_KEY")

    if not url or not model or not api_key:
        return None

    context_len = int(os.getenv("ASSIST_CONTEXT_LEN", "32768"))

    return OpenAIConfig(
        url=url,
        model=model,
        api_key=api_key,
        context_len=context_len,
    )


def _build_openai_chat_model(
        model: str, *, temperature: float, base_url: Optional[str] = None, api_key: Optional[str] = None
) -> BaseChatModel:
    """Create a ``ChatOpenAI`` instance with the provided parameters."""

    if ChatOpenAI is None:  # pragma: no cover - environment dependent
        raise RuntimeError("ChatOpenAI is not available")

    kwargs: dict[str, object] = {"model": model,
                                 "temperature": temperature,
                                 "max_retries": 0}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    return ChatOpenAI(**kwargs)


def select_chat_model(model: str, temperature: float) -> BaseChatModel:
    """Return a chat model honoring the optional custom OpenAI configuration."""

    config = _load_custom_openai_config()
    if config:
        logger.info("Using custom OpenAI-compatible endpoint at %s", config.url)
        llm = _build_openai_chat_model(
            config.model,
            temperature=temperature,
            base_url=config.url,
            api_key=config.api_key,
        )
        if not hasattr(llm, 'profile') or llm.profile is None:
            llm.profile = {}
        llm.profile["max_input_tokens"] = config.context_len
        return llm

    logger.info("Using OpenAI API configuration")
    return _build_openai_chat_model(model, temperature=temperature)
