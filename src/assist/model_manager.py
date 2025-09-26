"""Utilities for managing chat models for the server.

This module encapsulates the logic for selecting chat models and mapping
planning models to their corresponding execution models.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import yaml

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

# Mapping of model names to their character context limits
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    # OpenAI models expose a 128k token context window, which comfortably
    # exceeds most tool outputs. The limits here are expressed in characters
    # rather than tokens for simplicity.
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
}

DEFAULT_CONTEXT_LIMIT = 32_768


def _load_custom_openai_config() -> Optional[Dict[str, Any]]:
    """Load custom OpenAI configuration from llm-config.yml if present.
    
    Returns:
        Dict with 'url', 'model', and 'api_key' keys if config file exists,
        otherwise None.
    """
    config_path = Path("llm-config.yml")
    if not config_path.exists():
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = {'url', 'model', 'api_key'}
        if not isinstance(config, dict) or not required_fields.issubset(config.keys()):
            return None
            
        return config
    except Exception:
        return None


def select_chat_model(model: str, temperature: float) -> BaseChatModel:
    """Return the appropriate chat model for ``model``.

    If a custom OpenAI configuration is available in llm-config.yml, it will be used
    for ChatOpenAI models. Otherwise, if the model string indicates a ChatGPT model 
    (``gpt-*``) a ``ChatOpenAI`` instance is returned, otherwise a ``ChatOllama`` 
    instance is used.
    """
    # Check for custom configuration first
    custom_config = _load_custom_openai_config()
    if custom_config:
        if ChatOpenAI is None:  # pragma: no cover - environment dependent
            raise RuntimeError("ChatOpenAI is not available")
        return ChatOpenAI(
            model=custom_config['model'],
            temperature=temperature,
            api_key=custom_config['api_key'],
            base_url=custom_config['url']
        )
    
    # Fall back to default behavior
    if model.startswith("gpt-"):
        if ChatOpenAI is None:  # pragma: no cover - environment dependent
            raise RuntimeError("ChatOpenAI is not available")
        return ChatOpenAI(model=model, temperature=temperature)
    return ChatOllama(model=model, temperature=temperature)


def get_model_pair(model: str, temperature: float) -> Tuple[BaseChatModel, BaseChatModel]:
    """Return a pair of (planning_llm, execution_llm) for ``model``.

    If a custom OpenAI configuration is available in llm-config.yml, both planning 
    and execution models will use the custom configuration. Otherwise, the planning 
    model is always ``model`` and the execution model is looked up in 
    ``MODEL_EXECUTION_MAP``. If there is no mapping, the planning model is also used 
    for execution.
    """
    # Check for custom configuration first  
    custom_config = _load_custom_openai_config()
    if custom_config:
        # When using custom config, use the same model for both planning and execution
        plan_llm = select_chat_model(model, temperature)
        exec_llm = select_chat_model(model, temperature)
        return plan_llm, exec_llm
    
    # Fall back to default behavior
    plan_llm = select_chat_model(model, temperature)
    exec_model = MODEL_EXECUTION_MAP.get(model, model)
    exec_llm = select_chat_model(exec_model, temperature)
    return plan_llm, exec_llm


def get_context_limit(llm: BaseChatModel) -> int:
    """Return the character context limit for ``llm``.

    If ``llm.model`` is unknown, ``DEFAULT_CONTEXT_LIMIT`` is used.
    """
    model_name = getattr(llm, "model", "")
    return MODEL_CONTEXT_LIMITS.get(model_name, DEFAULT_CONTEXT_LIMIT)
