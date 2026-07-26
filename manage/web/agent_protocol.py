"""Minimal LangGraph SDK-compatible ASGI routes for agent runs.

The router deliberately accepts only the calls made by Assist's five
Deep Agents-shaped subagent tools. It is an in-process adapter over an injected
service, not a second execution path or a remote configuration API.
"""
from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from typing import Annotated, Any, Literal, Protocol

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from assist.run_service import RunNotFound


ASSISTANT_IDS = frozenset({
    "context-agent", "research-agent", "critique-agent", "delegate-agent",
})
MAX_BODY_BYTES = 66_000
MAX_MESSAGE_CHARS = 64_000
_RESOURCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class AgentProtocolService(Protocol):
    """Blocking service operations required by the HTTP adapter."""

    def create_thread(self, thread_id: str | None, metadata: dict | None) -> Any: ...
    def get_thread(self, thread_id: str) -> Any: ...
    def create_run(
        self,
        thread_id: str,
        assistant_id: str,
        text: str,
        *,
        multitask_strategy: str | None = None,
        metadata: dict | None = None,
    ) -> Any: ...
    def get_run(self, thread_id: str, run_id: str) -> Any: ...
    def cancel_run(self, thread_id: str, run_id: str) -> Any: ...


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _CreateThread(_StrictModel):
    thread_id: str | None = None
    metadata: dict[str, str] | None = None
    if_exists: Literal["do_nothing"] | None = None


class _Message(_StrictModel):
    role: Literal["user"]
    content: Annotated[str, Field(min_length=1, max_length=MAX_MESSAGE_CHARS)]


class _RunInput(_StrictModel):
    messages: Annotated[list[_Message], Field(min_length=1, max_length=1)]


class _CreateRun(_StrictModel):
    assistant_id: str
    input: _RunInput
    multitask_strategy: Literal["interrupt"] | None = None
    metadata: dict[str, str] | None = None
    # langgraph-sdk 0.3.3 always serializes these defaults. This implementation
    # has no streaming surface, so admit only the exact inert values it sends.
    stream_mode: Literal["values"] = "values"
    stream_subgraphs: Literal[False] = False
    stream_resumable: Literal[False] = False


async def _validated_body(request: Request, model: type[_StrictModel]) -> _StrictModel:
    """Read and validate JSON without allowing an unbounded request body."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_BODY_BYTES:
                raise HTTPException(status_code=413, detail="Request body too large")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length") from exc

    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="Request body too large")
        body.extend(chunk)
    try:
        return model.model_validate_json(bytes(body))
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if is_dataclass(value):
        return asdict(value)
    raise TypeError("agent protocol service results must be mappings or dataclasses")


def _validated_id(value: str) -> str:
    """Keep client identifiers bounded and inert before they reach a store."""
    if value in {".", ".."} or not _RESOURCE_ID.fullmatch(value):
        raise HTTPException(status_code=422, detail="Invalid resource id")
    return value


def _thread_response(value: Any) -> dict[str, Any]:
    source = _mapping(value)
    return {
        "thread_id": source["thread_id"],
        "created_at": source.get("created_at"),
        "updated_at": source.get("updated_at"),
        "metadata": source.get("metadata") or {},
        "status": source.get("status", "idle"),
        "values": source.get("values") or {},
        "interrupts": source.get("interrupts") or {},
    }


def _run_response(value: Any) -> dict[str, Any]:
    source = _mapping(value)
    result = {
        "run_id": source.get("run_id") or source.get("id"),
        "thread_id": source["thread_id"],
        "assistant_id": source["assistant_id"],
        "created_at": source.get("created_at"),
        "updated_at": source.get("updated_at"),
        "status": source["status"],
        "metadata": {},
        "multitask_strategy": source["multitask_strategy"],
    }
    if source.get("error") is not None:
        result["error"] = source["error"]
    return result


def create_agent_protocol_router(
    service: AgentProtocolService,
) -> APIRouter:
    """Return the private installed-SDK route subset backed by ``service``."""
    router = APIRouter()

    async def call(operation, *args, **kwargs):
        try:
            return await run_in_threadpool(operation, *args, **kwargs)
        except (FileNotFoundError, RunNotFound) as exc:
            raise HTTPException(status_code=404, detail="Resource not found") from exc

    @router.post("/threads")
    async def create_thread(request: Request) -> dict[str, Any]:
        body = await _validated_body(request, _CreateThread)
        assert isinstance(body, _CreateThread)
        if (body.thread_id is None) != (body.if_exists is None):
            raise HTTPException(
                status_code=422,
                detail="thread_id and if_exists=do_nothing must be provided together")
        if body.thread_id is not None:
            _validated_id(body.thread_id)
        return _thread_response(await call(
            service.create_thread, body.thread_id, body.metadata))

    @router.get("/threads/{thread_id}")
    async def get_thread(thread_id: str) -> dict[str, Any]:
        return _thread_response(
            await call(service.get_thread, _validated_id(thread_id)))

    @router.post("/threads/{thread_id}/runs")
    async def create_run(thread_id: str, request: Request) -> dict[str, Any]:
        thread_id = _validated_id(thread_id)
        body = await _validated_body(request, _CreateRun)
        assert isinstance(body, _CreateRun)
        if body.assistant_id not in ASSISTANT_IDS:
            raise HTTPException(status_code=404, detail="Unknown assistant")
        if body.metadata is not None and set(body.metadata) != {
                "parent_thread_id", "parent_run_id", "dispatch_key"}:
            raise HTTPException(status_code=422, detail="Invalid task metadata")
        message = body.input.messages[0]
        run = await call(
            service.create_run,
            thread_id,
            body.assistant_id,
            message.content,
            multitask_strategy=body.multitask_strategy,
            metadata=body.metadata,
        )
        return _run_response(run)

    @router.get("/threads/{thread_id}/runs/{run_id}")
    async def get_run(thread_id: str, run_id: str) -> dict[str, Any]:
        return _run_response(
            await call(
                service.get_run, _validated_id(thread_id), _validated_id(run_id)))

    @router.post("/threads/{thread_id}/runs/{run_id}/cancel")
    async def cancel_run(thread_id: str, run_id: str) -> None:
        await call(
            service.cancel_run, _validated_id(thread_id), _validated_id(run_id))

    return router
