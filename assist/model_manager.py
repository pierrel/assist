"""Chat-model selection with auto-discovery against the local LLM endpoint.

The serving endpoint (vLLM, OpenAI-compatible) already exposes the model
id and ``max_model_len`` via ``GET /v1/models``.  Rather than duplicating
that information across ``.dev.env``, ``.deploy.env``, and the systemd
unit, we probe the endpoint at first use, cache the result, and refresh
the cache when the model name no longer matches (i.e. the operator
swapped the model on the server).

Required env: ``ASSIST_MODEL_URL``.  Optional: ``ASSIST_API_KEY`` (falls
back through ``OPENAI_API_KEY`` to the constant ``"EMPTY"``, which vLLM
accepts and real OpenAI rejects — correct in both directions).

The probe is lazy.  ``select_chat_model`` does not network at module
import or at construction time; it only fires when an HTTP GET to
``{ASSIST_MODEL_URL}/models`` is needed to populate the cache.  This
preserves the existing "web server and vLLM start independently"
behavior — see ``ThreadManager.model``'s lazy property in
``assist/thread.py``.

Cache invalidation: ``_ModelNotFoundCacheBuster`` (a
``BaseCallbackHandler``) is wired onto every ``ChatOpenAI`` instance.
On ``openai.NotFoundError`` with ``code="model_not_found"`` it clears
the cache so the next request re-probes and picks up the new model.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Optional

import httpx
import openai
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


_PROBE_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class OpenAIConfig:
    """Configuration for the local OpenAI-compatible endpoint."""

    url: str
    model: str
    api_key: str
    context_len: int


_cached_config: Optional[OpenAIConfig] = None
_cache_lock = threading.Lock()


def _resolve_api_key() -> str:
    """ASSIST_API_KEY → OPENAI_API_KEY → ``"EMPTY"``.

    The local vLLM endpoint accepts any non-empty string; real OpenAI
    rejects ``"EMPTY"`` (correct — it should fail loudly if the user
    pointed this at the public API without a real key).
    """
    return (
        os.getenv("ASSIST_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or "EMPTY"
    )


def _probe_endpoint(url: str, api_key: str) -> OpenAIConfig:
    """Use the first entry from ``GET {url}/models`` — we serve one model
    at a time, so there's no selection logic.

    Raises ``RuntimeError`` if the endpoint is unreachable or returns no
    models.  Falls back to 32768 (and logs a warning) if the entry is
    missing ``max_model_len`` — vLLM always populates it, but real
    OpenAI's ``/v1/models`` does not.
    """
    probe_url = f"{url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = httpx.get(
            probe_url, headers=headers, timeout=_PROBE_TIMEOUT_S
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise RuntimeError(
            f"Could not reach {probe_url}: {e}.  "
            f"Check that the LLM server is running."
        ) from e

    payload = response.json()
    models = payload.get("data") or []
    if not models:
        raise RuntimeError(
            f"{probe_url} returned no models.  "
            f"Check that the LLM server has a model loaded."
        )
    entry = models[0]
    model_id = entry.get("id")
    if not model_id:
        raise RuntimeError(
            f"{probe_url} returned a model entry without an id: {entry!r}"
        )
    context_len = entry.get("max_model_len")
    if context_len is None:
        logger.warning(
            "Model %s did not report max_model_len; falling back to 32768",
            model_id,
        )
        context_len = 32768

    config = OpenAIConfig(
        url=url,
        model=model_id,
        api_key=api_key,
        context_len=int(context_len),
    )
    logger.info(
        "Discovered model %s (context=%d) at %s",
        config.model,
        config.context_len,
        config.url,
    )
    return config


def _get_config() -> Optional[OpenAIConfig]:
    """Return the cached config, probing on first access.

    Returns ``None`` only if ``ASSIST_MODEL_URL`` is unset — callers
    should treat that as a hard error.
    """
    global _cached_config
    url = os.getenv("ASSIST_MODEL_URL")
    if not url:
        return None
    with _cache_lock:
        if _cached_config is not None and _cached_config.url == url:
            return _cached_config
        api_key = _resolve_api_key()
        _cached_config = _probe_endpoint(url, api_key)
        return _cached_config


def invalidate_config_cache() -> None:
    """Bust the cache.  Next ``_get_config()`` call re-probes."""
    global _cached_config
    with _cache_lock:
        _cached_config = None


class _ModelNotFoundCacheBuster(BaseCallbackHandler):
    """Invalidate the config cache on ``model_not_found`` errors.

    Wired onto every ``ChatOpenAI`` instance via the ``callbacks=``
    kwarg.  Other ``NotFoundError`` shapes (and unrelated exceptions)
    pass through untouched — we only react to the specific signal that
    "the model on the server has changed."
    """

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        if (
            isinstance(error, openai.NotFoundError)
            and getattr(error, "code", None) == "model_not_found"
        ):
            logger.warning(
                "model_not_found from upstream; invalidating config cache"
            )
            invalidate_config_cache()


def _build_openai_chat_model(
    model: str,
    *,
    temperature: float,
    base_url: str,
    api_key: str,
) -> BaseChatModel:
    """Create a ``ChatOpenAI`` instance with the cache-buster callback."""

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_retries=0,
        callbacks=[_ModelNotFoundCacheBuster()],
        base_url=base_url,
        api_key=api_key,
    )


def select_chat_model(temperature: float) -> BaseChatModel:
    """Return a chat model bound to the auto-discovered local endpoint.

    Raises ``RuntimeError`` if ``ASSIST_MODEL_URL`` is unset.  This is
    a deliberate change from the previous behavior: there used to be a
    silent fallback to public OpenAI's ``gpt-4o-mini``, but the fallback
    branch was already broken (``model`` was undefined at that scope)
    and we don't actually want a quiet fallback to a remote API.
    """

    config = _get_config()
    if config is None:
        raise RuntimeError(
            "ASSIST_MODEL_URL is required.  Set it to the base URL of "
            "your OpenAI-compatible endpoint, e.g. "
            "http://localhost:8000/v1"
        )
    llm = _build_openai_chat_model(
        config.model,
        temperature=temperature,
        base_url=config.url,
        api_key=config.api_key,
    )
    llm.profile = {"max_input_tokens": config.context_len}
    return llm
