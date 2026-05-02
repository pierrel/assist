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

# HTTP request timeout for ChatOpenAI in seconds. Without this, a hung
# upstream (model stalled on a generation, network blip with no RST)
# can leave the agent loop blocked on a single ``invoke()`` indefinitely
# — which is how a single cron-launched eval run grew to 16 GB RSS
# over 21 hours.
#
# 180s was right for the prior vLLM + Qwen3-Coder-30B-A3B-AWQ MoE
# stack, where realistic generation finished in <60s.  The current
# llama.cpp + Qwen3.6-27B Q4_K_M dense stack has a much wider
# distribution: cheap tool-routing calls land in 5-20s, but a single
# heavy synthesis call (research-agent + nested fact-check on a
# >50k-token prompt) can need several minutes of pure prefill before
# generation even starts.  Empirically, the reasoning-impact eval
# saw both thinking_on and thinking_off time out at 180s on the
# finance-synthesis case (see docs/2026-05-01-reasoning-impact-eval.org).
# 600s gives meaningful headroom for those calls without going so
# wide that a genuine hang is invisible.
_REQUEST_TIMEOUT_S = 600.0


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


def _server_root(url: str) -> str:
    """Strip a trailing ``/v1`` (and any trailing slash) off the URL.

    ``ASSIST_MODEL_URL`` points at the OpenAI-compatible base
    (``http://host:port/v1``), but llama.cpp's ``/props`` endpoint lives
    at the server root (``http://host:port/props``).  Both shapes — with
    and without the ``/v1`` suffix — are accepted by callers, so this
    normalises before composing the root-level probe URL.
    """
    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base


def _probe_props_n_ctx(url: str, api_key: str) -> Optional[int]:
    """Try llama.cpp's ``/props`` for the runtime per-slot context length.

    Returns ``None`` when the endpoint is unreachable, returns non-200,
    or omits ``default_generation_settings.n_ctx``.  ``None`` is the
    "no signal — caller should fall back" answer; we deliberately do
    NOT raise, because a missing ``/props`` is the expected shape on
    vLLM and on real OpenAI.

    The value is the per-slot ``n_ctx`` (which is what a single
    conversation can actually use — llama.cpp divides ``-c`` across
    ``--parallel`` slots).
    """
    probe_url = f"{_server_root(url)}/props"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = httpx.get(
            probe_url, headers=headers, timeout=_PROBE_TIMEOUT_S
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return None

    try:
        payload = response.json()
    except Exception:
        return None

    settings = payload.get("default_generation_settings") or {}
    n_ctx = settings.get("n_ctx")
    if isinstance(n_ctx, int) and n_ctx > 0:
        return n_ctx
    return None


def _probe_endpoint(url: str, api_key: str) -> OpenAIConfig:
    """Discover the served model and its runtime context length.

    Probe order:
      1. ``GET {url}/models`` — required.  Yields the model id and, on
         vLLM, ``max_model_len`` (the runtime context).
      2. ``GET {server_root}/props`` — used as fallback when (1) lacks
         ``max_model_len``.  llama.cpp surfaces the runtime context here
         as ``default_generation_settings.n_ctx``.  We deliberately do
         NOT use ``meta.n_ctx_train`` from /v1/models — that is the
         model's *trained* length, which can exceed the configured
         runtime ``-c`` value and would lead to context-overflow.

    Falls back to 32768 (with a warning) only when neither probe
    surfaces a value — i.e. neither vLLM's ``max_model_len`` nor
    llama.cpp's ``/props`` is reachable.
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
        # llama.cpp path: /v1/models doesn't carry the runtime context;
        # try /props before resorting to the hardcoded default.
        context_len = _probe_props_n_ctx(url, api_key)

    if context_len is None:
        logger.warning(
            "Model %s did not report max_model_len and /props had no "
            "n_ctx; falling back to 32768",
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
    enable_thinking: bool | None = None,
) -> BaseChatModel:
    """Create a ``ChatOpenAI`` instance with the cache-buster callback.

    ``enable_thinking`` (Qwen3 family + llama.cpp): when explicitly
    False, the chat-template kwarg ``enable_thinking=false`` is passed
    via ``extra_body``, which prefills the assistant message with an
    empty ``<think></think>`` block — the canonical way to disable
    Qwen3 chain-of-thought.  When None or True, no ``extra_body`` is
    set: True is Qwen3's own default, and None means "leave the upstream
    behavior alone" so non-Qwen3 backends don't see a kwarg they may
    not understand.
    """

    extra_body = None
    if enable_thinking is False:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    kwargs = {
        "model": model,
        "temperature": temperature,
        "max_retries": 0,
        "timeout": _REQUEST_TIMEOUT_S,
        "callbacks": [_ModelNotFoundCacheBuster()],
        "base_url": base_url,
        "api_key": api_key,
    }
    if extra_body is not None:
        kwargs["extra_body"] = extra_body

    return ChatOpenAI(**kwargs)


def select_chat_model(
    temperature: float,
    *,
    enable_thinking: bool | None = None,
) -> BaseChatModel:
    """Return a chat model bound to the auto-discovered local endpoint.

    Raises ``RuntimeError`` if ``ASSIST_MODEL_URL`` is unset.  This is
    a deliberate change from the previous behavior: there used to be a
    silent fallback to public OpenAI's ``gpt-4o-mini``, but the fallback
    branch was already broken (``model`` was undefined at that scope)
    and we don't actually want a quiet fallback to a remote API.

    ``enable_thinking`` (default ``None``): see
    ``_build_openai_chat_model``.  Used by the reasoning-impact eval
    in ``edd/eval/test_reasoning_impact.py`` to A/B Qwen3 thinking
    mode against latency.  Production callers leave it unset.
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
        enable_thinking=enable_thinking,
    )
    llm.profile = {"max_input_tokens": config.context_len}
    return llm
