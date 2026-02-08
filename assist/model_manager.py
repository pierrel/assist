"""Utilities for managing chat models for the server.

This module encapsulates the logic for selecting chat models. The default
configuration uses the public OpenAI API with ``gpt-4o-mini``. A local
``llm-config.yml`` file can override this behaviour by providing a custom
OpenAI-compatible endpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import requests
import yaml
from langchain_core.language_models.chat_models import BaseChatModel

from assist.config_manager import get_config, config_path

try:  # pragma: no cover - optional dependency
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from langchain_ollama import ChatOllama
except Exception:  # pragma: no cover
    ChatOllama = None  # type: ignore

DEFAULT_MODEL = "gpt-4o-mini"

# Mapping of model names to their character context limits. OpenAI models
# expose a 128k token context window, which comfortably exceeds most tool
# outputs. The limits here are expressed in characters for simplicity.
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    'models/mistral.gguf': 128_000,
}

DEFAULT_CONTEXT_LIMIT = 32_768


@dataclass(frozen=True)
class OpenAIConfig:
    """Configuration for a custom OpenAI-compatible endpoint."""

    url: str
    model: str
    api_key: str
    context_len: int
    test_url_path: Optional[str] = None


def _test_local_llm_availability(config: OpenAIConfig) -> bool:
    """Test if the local LLM is available by making a request to the test URL."""
    if not config.test_url_path:
        return False
    
    test_url = config.url.rstrip('/') + config.test_url_path
    try:
        response = requests.get(test_url, timeout=5)
        return bool(response.status_code == 200)
    except (requests.RequestException, Exception):
        return False


@lru_cache(maxsize=1)
def _load_custom_openai_config() -> Optional[OpenAIConfig]:
    """Return the custom OpenAI configuration."""
    raw = get_config()
    missing = [key for key in ("url", "model", "api_key") if not raw.get(key)]
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise RuntimeError(
            f"Config is missing required keys: {missing_keys}"
        )

    return OpenAIConfig(
        url=str(raw["url"]),
        model=str(raw["model"]),
        api_key=str(raw["api_key"]),
        test_url_path=str(raw["test_url_path"]),
        context_len=int(raw["context_len"])
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
        if _test_local_llm_availability(config):
            print(f"Using local LLM configuration from ")
            model= _build_openai_chat_model(
                config.model,
                temperature=temperature,
                base_url=config.url,
                api_key=config.api_key,
            )
            model.profile["max_input_tokens"] = config.context_len
            return model
        else:
            print(f"Local LLM from {config_path()} is not available, falling back to OpenAI API")

    print("Using OpenAI API configuration")
    if model.startswith("gpt-"):
        return _build_openai_chat_model(model, temperature=temperature)

    if ChatOllama is not None:
        return ChatOllama(model=model, temperature=temperature)

    raise RuntimeError(
        "ChatOllama is not available and no OpenAI-compatible configuration was found"
    )


def get_model_pair(temperature: float) -> Tuple[BaseChatModel, BaseChatModel]:
    """Return the planning and execution LLMs for the server."""

    config = _load_custom_openai_config()
    if config:
        if _test_local_llm_availability(config):
            print(f"Using local LLM configuration from {config_path()}")
            llm = _build_openai_chat_model(
                config.model,
                temperature=temperature,
                base_url=config.url,
                api_key=config.api_key,
            )
            return llm, llm
        else:
            print(f"Local LLM from {config_path()} is not available, falling back to OpenAI API")

    print("Using OpenAI API configuration")
    default_llm = _build_openai_chat_model(DEFAULT_MODEL, temperature=temperature)
    return default_llm, default_llm


def get_context_limit(llm: BaseChatModel) -> int:
    """Return the character context limit for ``llm``.

    If ``llm.model`` is unknown, ``DEFAULT_CONTEXT_LIMIT`` is used.
    """

    model_name = getattr(llm, "model", "")
    return MODEL_CONTEXT_LIMITS.get(model_name, DEFAULT_CONTEXT_LIMIT)
