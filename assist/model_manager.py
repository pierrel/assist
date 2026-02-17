"""Utilities for managing chat models for the server.

This module encapsulates the logic for selecting chat models. The default
configuration uses the public OpenAI API with ``gpt-4o-mini``. Configuration
can be provided via environment variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import requests
from langchain_core.language_models.chat_models import BaseChatModel

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
    """Return the custom OpenAI configuration from environment variables.

    Required environment variables:
    - ASSIST_MODEL_URL: OpenAI-compatible API endpoint
    - ASSIST_MODEL_NAME: Model identifier
    - ASSIST_API_KEY: API key for authentication

    Optional environment variables:
    - ASSIST_CONTEXT_LEN: Context window size (default: 32768)
    - ASSIST_TEST_URL_PATH: Path to test endpoint availability (e.g., /models)
    """
    url = os.getenv("ASSIST_MODEL_URL")
    model = os.getenv("ASSIST_MODEL_NAME")
    api_key = os.getenv("ASSIST_API_KEY")

    # Check for required variables
    missing = []
    if not url:
        missing.append("ASSIST_MODEL_URL")
    if not model:
        missing.append("ASSIST_MODEL_NAME")
    if not api_key:
        missing.append("ASSIST_API_KEY")

    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise RuntimeError(
            f"Missing required environment variables: {missing_keys}"
        )

    # Optional variables with defaults
    context_len = int(os.getenv("ASSIST_CONTEXT_LEN", "32768"))
    test_url_path = os.getenv("ASSIST_TEST_URL_PATH")

    return OpenAIConfig(
        url=url,
        model=model,
        api_key=api_key,
        test_url_path=test_url_path,
        context_len=context_len
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
            # Initialize profile if it doesn't exist
            if not hasattr(model, 'profile') or model.profile is None:
                model.profile = {}
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
            # Initialize profile if it doesn't exist
            if not hasattr(llm, 'profile') or llm.profile is None:
                llm.profile = {}
            llm.profile["max_input_tokens"] = config.context_len
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
