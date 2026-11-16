"""Private observations for live research evals; no product tools are replaced."""

from __future__ import annotations

import contextvars
import json
import os
import time
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, message_to_dict, messages_to_dict


_thread_id = contextvars.ContextVar("research_eval_thread_id", default=None)


def append_private(path: Path, event: str, **fields) -> None:
    record = {"at": time.time(), "event": event, "thread_id": _thread_id.get(), **fields}
    encoded = json.dumps(record, ensure_ascii=False, default=str)
    for name in ("TAVILY_API_KEY", "ASSIST_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            encoded = encoded.replace(secret, "[REDACTED]")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, (encoded + "\n").encode())
    finally:
        os.close(fd)


class _Trace(BaseCallbackHandler):
    """Retain the actual model/tool graph, including nested task inputs and returns."""

    def __init__(self, artifact: Path):
        self.artifact = artifact
        self.tools: dict[str, str | None] = {}

    def on_chat_model_start(self, serialized, messages, **kwargs):
        append_private(self.artifact / "events.jsonl", "model_start")
        append_private(self.artifact / "trace.jsonl", "model_start",
                       run_id=str(kwargs.get("run_id")),
                       parent_run_id=str(kwargs.get("parent_run_id")),
                       batches=[messages_to_dict(batch) for batch in messages])

    def on_llm_end(self, response, **kwargs):
        append_private(self.artifact / "events.jsonl", "model_end")
        append_private(self.artifact / "trace.jsonl", "model_end",
                       run_id=str(kwargs.get("run_id")),
                       parent_run_id=str(kwargs.get("parent_run_id")),
                       generations=[[message_to_dict(generation.message)
                                     if hasattr(generation, "message") else
                                     {"type": type(generation).__name__,
                                      "text": generation.text}
                                     for generation in batch]
                                    for batch in response.generations])

    def on_llm_error(self, error, **kwargs):
        append_private(self.artifact / "events.jsonl", "model_error",
                       error=type(error).__name__)
        append_private(self.artifact / "trace.jsonl", "model_error",
                       run_id=str(kwargs.get("run_id")),
                       parent_run_id=str(kwargs.get("parent_run_id")),
                       error_type=type(error).__name__, error_text=str(error))

    def on_tool_start(self, serialized, input_str, *, run_id=None,
                      parent_run_id=None, inputs=None, **kwargs):
        name = serialized.get("name")
        self.tools[str(run_id)] = name
        subagent = None
        if name == "task":
            if isinstance(inputs, Mapping):
                subagent = inputs.get("subagent_type")
            else:
                try:
                    subagent = json.loads(input_str).get("subagent_type")
                except (TypeError, ValueError, AttributeError):
                    pass
        append_private(self.artifact / "events.jsonl", "tool_start",
                       name=name, subagent=subagent,
                       run_id=str(run_id), parent_run_id=str(parent_run_id))
        append_private(self.artifact / "trace.jsonl", "tool_start",
                       name=name, run_id=str(run_id),
                       parent_run_id=str(parent_run_id), inputs=inputs,
                       input_str=input_str)

    def on_tool_end(self, output, *, run_id=None, parent_run_id=None, **kwargs):
        name = self.tools.pop(str(run_id), None)
        append_private(self.artifact / "events.jsonl", "tool_end",
                       name=name, run_id=str(run_id),
                       parent_run_id=str(parent_run_id))
        append_private(self.artifact / "trace.jsonl", "tool_end",
                       name=name, run_id=str(run_id),
                       parent_run_id=str(parent_run_id),
                       output_type=type(output).__name__,
                       output=(message_to_dict(output) if isinstance(output, BaseMessage)
                               else str(output)))

    def on_tool_error(self, error, *, run_id=None, **kwargs):
        name = self.tools.pop(str(run_id), None)
        append_private(self.artifact / "events.jsonl", "tool_error",
                       name=name, run_id=str(run_id),
                       error=type(error).__name__)
        append_private(self.artifact / "trace.jsonl", "tool_error",
                       name=name, run_id=str(run_id),
                       parent_run_id=str(kwargs.get("parent_run_id")),
                       error_type=type(error).__name__, error_text=str(error))


def install_model_and_tool_trace(artifact: Path) -> None:
    """Attach the existing full-message callback to every private web Thread."""
    from assist.thread import Thread
    from assist.thread_queue import THREAD_QUEUE
    from langchain_openai import ChatOpenAI

    callback = _Trace(artifact)
    original_init = Thread.__init__
    original_run = Thread._run
    original_stream = ChatOpenAI._stream
    original_generate = ChatOpenAI._generate
    original_acquire = THREAD_QUEUE.acquire

    @contextmanager
    def acquire(*args, **kwargs):
        started = time.monotonic()
        with original_acquire(*args, **kwargs) as handle:
            acquired = time.monotonic()
            append_private(artifact / "lifecycle.jsonl", "model_slot_acquired",
                           waiting_ms=round((acquired - started) * 1000, 2))
            try:
                yield handle
            finally:
                append_private(artifact / "lifecycle.jsonl", "model_slot_released",
                               holding_ms=round((time.monotonic() - acquired) * 1000, 2))

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.runconfig["callbacks"] = [callback, *self.runconfig.get("callbacks", ())]

    def run(self, graph_input):
        token = _thread_id.set(self.thread_id)
        try:
            return original_run(self, graph_input)
        finally:
            _thread_id.reset(token)

    def provider_request(model, messages, kwargs, request_id):
        append_private(
            artifact / "provider-calls.jsonl", "provider_request",
            request_id=request_id,
            model=model.model_name, streaming=model.streaming,
            messages=messages_to_dict(messages), tools=kwargs.get("tools"),
            tool_choice=kwargs.get("tool_choice"),
            response_format=kwargs.get("response_format"),
            temperature=model.temperature, top_p=model.top_p,
            extra_body=kwargs.get("extra_body", model.extra_body),
        )

    def stream(model, *args, **kwargs):
        messages = args[0] if args else kwargs["messages"]
        request_id = uuid.uuid4().hex
        provider_request(model, messages, kwargs, request_id)
        started = time.monotonic()
        completed = False
        try:
            yield from original_stream(model, *args, **kwargs)
            completed = True
        finally:
            append_private(artifact / "provider-calls.jsonl", "provider_finished",
                           request_id=request_id, completed=completed,
                           elapsed_ms=round((time.monotonic() - started) * 1000, 2))

    def generate(model, messages, stop=None, run_manager=None, **kwargs):
        if model.streaming:
            return original_generate(model, messages, stop=stop, run_manager=run_manager,
                                     **kwargs)
        request_id = uuid.uuid4().hex
        provider_request(model, messages, kwargs, request_id)
        started = time.monotonic()
        completed = False
        try:
            result = original_generate(model, messages, stop=stop,
                                       run_manager=run_manager, **kwargs)
            completed = True
            return result
        finally:
            append_private(artifact / "provider-calls.jsonl", "provider_finished",
                           request_id=request_id, completed=completed,
                           elapsed_ms=round((time.monotonic() - started) * 1000, 2))

    Thread.__init__ = init
    Thread._run = run
    ChatOpenAI._stream = stream
    ChatOpenAI._generate = generate
    THREAD_QUEUE.acquire = acquire


def install_http_trace(artifact: Path, search_url: str) -> None:
    """Retain configured search HTTP responses before tool normalization."""
    import requests

    original_get = requests.get
    original_post = requests.post
    search = urlsplit(search_url)

    def observed(method, original, url, *args, **kwargs):
        try:
            target = urlsplit(str(url))
            target_port = target.port
        except ValueError:
            return original(url, *args, **kwargs)
        search_call = (target.scheme, target.hostname, target_port, target.path) == (
            search.scheme, search.hostname, search.port, search.path.rstrip("/") + "/search")
        tavily_call = target.hostname == "api.tavily.com" and target.path == "/search"
        if not search_call and not tavily_call:
            return original(url, *args, **kwargs)
        safe_request = {"url": f"{target.scheme}://{target.hostname}{':' + str(target_port) if target_port else ''}{target.path}",
                        "method": method}
        if search_call:
            params = kwargs.get("params") or {}
            safe_request.update(backend="searxng", query=params.get("q"),
                                requested_engine=params.get("engines"),
                                category=params.get("categories"))
        elif tavily_call:
            body = kwargs.get("json") or {}
            safe_request.update(backend="tavily", query=body.get("query"))
        started = time.monotonic()
        try:
            response = original(url, *args, **kwargs)
            result = {"status": response.status_code,
                      "elapsed_ms": round((time.monotonic() - started) * 1000, 2)}
            if search_call or tavily_call:
                result["raw_response"] = response.text
                try:
                    result["structured_response"] = response.json()
                except ValueError:
                    result["structured_response"] = None
            append_private(artifact / "http.jsonl", "http_result", **safe_request,
                           **result)
            return response
        except Exception as error:
            append_private(artifact / "http.jsonl", "http_error", **safe_request,
                           elapsed_ms=round((time.monotonic() - started) * 1000, 2),
                           error_type=type(error).__name__, error=str(error))
            raise

    requests.get = lambda url, *args, **kwargs: observed(
        "GET", original_get, url, *args, **kwargs)
    requests.post = lambda url, *args, **kwargs: observed(
        "POST", original_post, url, *args, **kwargs)


def install_existing_egress_guard(proof_path: Path) -> None:
    """Permit private sandboxes only on the inspected existing proxy/network."""
    from assist.egress import runtime_state
    from assist.egress.client_map import configured_directory
    from assist.egress.store import APPROVALS_SUBDIR
    from assist.sandbox_manager import (
        BROWSER_NETWORK, EGRESS_NETWORK, EGRESS_PROXY_NAME, SandboxManager,
        _egress_proxy_config_hash, _load_egress_allowlist, _network_identity,
    )

    proof = json.loads(proof_path.read_text())
    expected = proof["reuse_validation"]
    approvals_root = os.environ.get("ASSIST_EGRESS_APPROVALS_DIR")
    approvals_mount = (os.path.join(approvals_root, APPROVALS_SUBDIR)
                       if approvals_root else None)
    if (approvals_root != proof["production_env_paths"]["ASSIST_EGRESS_APPROVALS_DIR"]
            or approvals_mount != expected["approvals_mount"]
            or configured_directory() != expected["client_map_dir"]
            or runtime_state.runtime_directory() != expected["runtime_dir"]):
        raise RuntimeError("eval egress configuration differs from the live proxy")

    def verify(_cls, client):
        ordinary = client.networks.get(EGRESS_NETWORK)
        browser = client.networks.get(BROWSER_NETWORK)
        ordinary_id, ordinary_cidr = _network_identity(ordinary, isolated=False)
        browser_id, browser_cidr = _network_identity(browser, isolated=True)
        if (ordinary_id != proof["networks"][EGRESS_NETWORK]["id"]
                or browser_id != proof["networks"][BROWSER_NETWORK]["id"]
                or ordinary_cidr != proof["networks"][EGRESS_NETWORK]["subnet"]
                or browser_cidr != proof["networks"][BROWSER_NETWORK]["subnet"]):
            raise RuntimeError("eval egress network identity changed")
        proxy = client.containers.get(EGRESS_PROXY_NAME)
        proxy.reload()
        network_ref = f"{ordinary_id}:{ordinary_cidr}|{browser_id}:{browser_cidr}"
        fingerprint = _egress_proxy_config_hash(
            ",".join(_load_egress_allowlist()), approvals_mount, network_ref,
            expected["client_map_dir"])
        if (proxy.id != expected["proxy_id"] or proxy.status != "running"
                or proxy.labels.get("assist.egress-allowlist-hash") != fingerprint
                or fingerprint != expected["expected_hash"]
                or any(runtime_state.pending_creation(kind, name) for kind, name in (
                    ("proxy", EGRESS_PROXY_NAME), ("network", EGRESS_NETWORK),
                    ("network", BROWSER_NETWORK)))
                or runtime_state.retirement("proxy", proxy.id) is not None
                or runtime_state.retirement("network", ordinary_id) is not None
                or runtime_state.retirement("network", browser_id) is not None):
            raise RuntimeError("eval shared egress proxy is not the inspected live generation")
        return EGRESS_PROXY_NAME

    SandboxManager._ensure_egress_proxy_running = classmethod(verify)
    # Browser reconciliation calls the bounded entry directly, so keep that
    # path read-only too while sharing the same inspected proxy generation.
    SandboxManager._ensure_egress_proxy_bounded = classmethod(
        lambda cls: verify(cls, cls._get_docker_client()))
    ordinary_backend = SandboxManager.get_sandbox_backend.__func__

    def require_docker(cls, *args, **kwargs):
        backend = ordinary_backend(cls, *args, **kwargs)
        if backend is None:
            raise RuntimeError("live research eval requires the ordinary Docker sandbox")
        return backend

    SandboxManager.get_sandbox_backend = classmethod(require_docker)
