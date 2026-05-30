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
import socket
import threading
from dataclasses import dataclass
from typing import Any, Optional

import httpx
import openai
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from assist.env import env_float

logger = logging.getLogger(__name__)


_PROBE_TIMEOUT_S = 10.0


# TCP keepalive socket options for the ChatOpenAI httpx client.
# Defense-in-depth against dead-peer scenarios where a process is
# actively blocked on `recv()` against a connection that the peer's
# kernel has stopped acknowledging (e.g. peer-host kernel panic with
# no FIN/RST sent, NAT/firewall state expiry).  Without these, the
# kernel's safety net is the default `tcp_keepalive_time` (7200s = 2h
# on Linux).  With these, the kernel surfaces a dead peer as a clean
# ECONNRESET after:
#
#     TCP_KEEPIDLE + TCP_KEEPCNT * TCP_KEEPINTVL
#   = 30          + 3            * 10
#   = 60 seconds.
#
# What this does NOT bound: a connection that has already received FIN
# from the peer (socket in CLOSE_WAIT) but where the application hasn't
# called recv() since.  Keepalive only probes ESTABLISHED connections.
# That class is bounded by the openai SDK's per-call timeout + the
# upstream retry layer.
#
# Constants are Linux-specific (TCP_KEEPIDLE etc. are TCP_* level);
# `SO_KEEPALIVE` itself is portable but the per-option tuning is not.
# Production target is Linux, so this is the right scope.
_TCP_KEEPALIVE_SOCKET_OPTIONS: list[tuple[int, int, int]] = [
    (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
    (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30),
    (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10),
    (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3),
]


# Per-phase httpx Timeout for ChatOpenAI's underlying client.  Without
# this — i.e. with the prior single-scalar ``timeout=600`` — a stuck
# TCP handshake against an unreachable endpoint burned the full 600s
# of httpcore connect retries before raising.  Splitting the four
# phases lets connect fail fast (10s) while still giving long
# generations the headroom they empirically need (read=600s, matches
# the per-thread queue ``hold_timeout_s``).
#
# Read note on ``read``: this is httpx's *idle-byte* timeout, not a
# total-response cap.  With a streaming endpoint a slow generation
# that emits tokens continuously will not trip it.  ChatOpenAI's
# default ``streaming=False`` means ``read`` is effectively the cap
# on time-from-request-write to first response byte; for genuinely
# long synthesis calls operators can bump ``ASSIST_LLM_READ_TIMEOUT_S``.
#
# Each value is read on every ``_build_request_timeout`` call.  In
# practice ``ThreadManager.model`` is a lazy property cached for the
# process lifetime, so in production the env is consulted once at
# first-message time and a systemd restart is required to pick up
# new values.  The on-call read still simplifies tests and avoids
# stale module-level constants.


def _build_request_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=env_float("ASSIST_LLM_CONNECT_TIMEOUT_S", 10.0),
        read=env_float("ASSIST_LLM_READ_TIMEOUT_S", 600.0),
        write=env_float("ASSIST_LLM_WRITE_TIMEOUT_S", 60.0),
        pool=env_float("ASSIST_LLM_POOL_TIMEOUT_S", 10.0),
    )


class _HttpClient(httpx.Client):
    """``httpx.Client`` subclass that closes itself on GC.

    Mirrors langchain_openai's own ``_SyncHttpxClientWrapper`` (which
    subclasses ``openai.DefaultHttpxClient`` for the same reason): a
    plain ``httpx.Client`` has no ``__del__`` finalizer, so a ChatOpenAI
    instance going out of scope leaks its connection pool until process
    exit.  Production hits this once per process (cached via
    ``ThreadManager.model``), but eval sweeps construct N clients across
    test classes — the cumulative leak adds up.
    """

    def __del__(self) -> None:
        if self.is_closed:
            return
        try:
            self.close()
        except Exception:
            pass


class _AsyncHttpClient(httpx.AsyncClient):
    """``httpx.AsyncClient`` subclass with a sync ``__del__``.

    Same reason as :class:`_HttpClient`.  ``AsyncClient.close`` is
    async; calling it from ``__del__`` (a sync context) would raise.
    ``AsyncClient.aclose`` is also async.  The sync escape hatch is
    ``_close_rs``-style transport close, but that's private — instead
    we just let the OS reap the underlying sockets on process exit.
    The pool's idle connections are bounded by httpx's defaults (5),
    so the leak is small and the test-sweep concern doesn't apply
    here in the same magnitude.
    """

    def __del__(self) -> None:
        # No safe sync close path; rely on OS-level reaping.  See class
        # docstring for the trade-off.
        return


def _build_http_client() -> httpx.Client:
    """Construct the sync httpx.Client used by every ChatOpenAI instance.

    Wires in TCP keepalive (see ``_TCP_KEEPALIVE_SOCKET_OPTIONS``) so a
    dead peer surfaces in ~60s rather than waiting on the kernel
    default ``tcp_keepalive_time`` (2h on Linux).  The per-phase
    :class:`httpx.Timeout` from :func:`_build_request_timeout` is set
    as the client default and is *also* set via ChatOpenAI's ``timeout``
    kwarg — in normal openai-SDK usage the per-call ``client.with_options(
    timeout=...)`` always overrides the client default, so the client
    default is the fallback for any direct httpx use.
    """
    return _HttpClient(
        timeout=_build_request_timeout(),
        transport=httpx.HTTPTransport(
            socket_options=_TCP_KEEPALIVE_SOCKET_OPTIONS,
        ),
    )


def _build_http_async_client() -> httpx.AsyncClient:
    """Construct the async httpx.AsyncClient used by every ChatOpenAI.

    The async client is what deepagents' subagent dispatch
    (``await subagent.ainvoke(...)``) uses.  Without setting
    ``http_async_client`` on ChatOpenAI, the openai SDK falls back to
    its default async client which has no socket options, leaving
    subagent LLM calls completely unprotected from the half-closed-peer
    wedge that motivated this PR.  Setting both keeps sync and async
    paths in lockstep.
    """
    return _AsyncHttpClient(
        timeout=_build_request_timeout(),
        transport=httpx.AsyncHTTPTransport(
            socket_options=_TCP_KEEPALIVE_SOCKET_OPTIONS,
        ),
    )


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
        # max_retries=0 disables the OpenAI Python SDK's built-in
        # retry layer.  The single retry layer for transient errors
        # is ModelRetryMiddleware in assist/agent.py — keeping retries
        # in one place gives uniform logging and predictable bounds
        # on per-call wall-clock.
        "max_retries": 0,
        "timeout": _build_request_timeout(),
        # Both sync AND async clients carry TCP keepalive — see
        # _build_http_client / _build_http_async_client.  Async covers
        # the deepagents subagent dispatch path (await subagent.ainvoke).
        "http_client": _build_http_client(),
        "http_async_client": _build_http_async_client(),
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
