"""Bounded WebSocket transport for the voice-call client.

This module owns framing, transport authentication, backpressure, and teardown.
Call state and policy will live in ``session.py``. This module exposes one
synchronous runner seam for that later layer without importing the agent,
speech, or flow layers.
"""
from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import json
import logging
import math
import os
import secrets
import threading
import time
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from manage.web.app import app


logger = logging.getLogger(__name__)

MAX_WS_MESSAGE_BYTES = 4096
PCM_FRAME_BYTES = 640
FIRST_RING_SECONDS = 5.0
IDLE_SECONDS = 5.0
MAX_CONNECTION_SECONDS = 30 * 60.0
MAX_CONTROLS_PER_SECOND = 20
INBOUND_CAPACITY = 50
OUTBOUND_CAPACITY = 100
TERMINAL_GRACE_SECONDS = 2.0

_PHONE_FIELDS = {
    "ring": {"call_id", "caller"},
    "answered": {"call_id"},
    "call_end": {"cause"},
    "dtmf": {"digit"},
    "stats": {"rate_hz", "underruns"},
}
_SERVER_FIELDS = {
    "answer": set(),
    "hangup": set(),
    "flush_uplink": set(),
}


class WireProtocolError(ValueError):
    """A known wire message violates the protocol."""


class BufferClosed(Exception):
    """A call-local buffer was closed during an operation."""


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise WireProtocolError(f"duplicate field: {key}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise WireProtocolError(f"non-finite number: {value}")


def _bounded_string(value: Any, field: str, limit: int) -> str:
    if type(value) is not str or len(value) > limit:
        raise WireProtocolError(f"invalid {field}")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise WireProtocolError(f"invalid {field}") from exc
    return value


def decode_phone_control(text: str) -> dict[str, Any] | None:
    """Decode a phone control, returning ``None`` for an unknown type."""
    try:
        encoded_size = len(text.encode("utf-8"))
    except UnicodeError as exc:
        raise WireProtocolError("invalid UTF-8 control") from exc
    if encoded_size > MAX_WS_MESSAGE_BYTES:
        raise WireProtocolError("control frame too large")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise WireProtocolError("invalid JSON control") from exc
    if type(value) is not dict or type(value.get("type")) is not str:
        raise WireProtocolError("control must be an object with a string type")

    kind = value["type"]
    fields = _PHONE_FIELDS.get(kind)
    if fields is None:
        return None
    if set(value) != fields | {"type"}:
        raise WireProtocolError(f"invalid fields for {kind}")

    if kind == "ring":
        if not _bounded_string(value["call_id"], "call_id", 128):
            raise WireProtocolError("invalid call_id")
        _bounded_string(value["caller"], "caller", 32)
    elif kind == "answered":
        if not _bounded_string(value["call_id"], "call_id", 128):
            raise WireProtocolError("invalid call_id")
    elif kind == "call_end":
        _bounded_string(value["cause"], "cause", 64)
    elif kind == "dtmf":
        digit = _bounded_string(value["digit"], "digit", 1)
        if len(digit) != 1:
            raise WireProtocolError("invalid digit")
    elif kind == "stats":
        rate = value["rate_hz"]
        underruns = value["underruns"]
        if (
            type(rate) not in (int, float)
            or not math.isfinite(rate)
            or rate <= 0
            or type(underruns) is not int
            or underruns < 0
        ):
            raise WireProtocolError("invalid stats")
    return value


def encode_server_control(control: dict[str, Any]) -> str:
    """Encode one server control after validating its exact shape."""
    if type(control) is not dict or type(control.get("type")) is not str:
        raise WireProtocolError("server control must have a string type")
    fields = _SERVER_FIELDS.get(control["type"])
    if fields is None or set(control) != fields | {"type"}:
        raise WireProtocolError("invalid server control")
    return json.dumps(control, separators=(",", ":"))


class _CloseableBuffer:
    def __init__(self, capacity: int):
        self._capacity = capacity
        self._items: deque[Any] = deque()
        self._closed = False
        self._condition = threading.Condition()

    def put(self, item: Any) -> None:
        with self._condition:
            while len(self._items) >= self._capacity and not self._closed:
                self._condition.wait()
            if self._closed:
                raise BufferClosed
            self._items.append(item)
            self._condition.notify_all()

    def get(self) -> Any:
        with self._condition:
            while not self._items and not self._closed:
                self._condition.wait()
            if not self._items:
                raise BufferClosed
            item = self._items.popleft()
            self._condition.notify_all()
            return item

    def close(self, *, discard: bool = True) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            if discard:
                self._items.clear()
            self._condition.notify_all()


class InboundBuffer(_CloseableBuffer):
    """Bounded phone-to-session buffer; close discards call-local backlog."""

    def __init__(self) -> None:
        super().__init__(INBOUND_CAPACITY)


class OutboundBuffer(_CloseableBuffer):
    """Bounded session-to-phone buffer with graceful or abrupt teardown."""

    def __init__(self) -> None:
        super().__init__(OUTBOUND_CAPACITY)

    def abort(self) -> None:
        with self._condition:
            self._closed = True
            self._items.clear()
            self._condition.notify_all()

    def close_with_final(self, final: dict[str, Any]) -> None:
        encode_server_control(final)
        with self._condition:
            if self._closed:
                return
            self._items.clear()
            self._items.append(final)
            self._closed = True
            self._condition.notify_all()

    def finish(self) -> None:
        self.close(discard=False)


@dataclass(frozen=True)
class CallBuffers:
    inbound: InboundBuffer
    outbound: OutboundBuffer


CallRunner = Callable[[dict[str, Any], CallBuffers], None]
_CALL_RUNNER: CallRunner | None = None
_active_call = False


def configure_call_runner(runner: CallRunner | None) -> None:
    """Set the process-wide synchronous session runner during app startup."""
    global _CALL_RUNNER
    _CALL_RUNNER = runner


def _claim_call() -> bool:
    global _active_call
    if _active_call:
        return False
    _active_call = True
    return True


def _release_call() -> None:
    global _active_call
    _active_call = False


class _ControlRate:
    def __init__(self) -> None:
        self._times: deque[float] = deque()

    def record(self, now: float) -> None:
        while self._times and now - self._times[0] >= 1.0:
            self._times.popleft()
        if len(self._times) >= MAX_CONTROLS_PER_SECOND:
            raise WireProtocolError("control rate exceeded")
        self._times.append(now)


async def _receive_message(
    websocket: WebSocket, timeout: float
) -> dict[str, Any]:
    try:
        return await asyncio.wait_for(websocket.receive(), timeout=timeout)
    except TimeoutError:
        raise WireProtocolError("wire timeout") from None


async def _receive_ring(
    websocket: WebSocket, rate: _ControlRate
) -> dict[str, Any]:
    deadline = time.monotonic() + FIRST_RING_SECONDS
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WireProtocolError("ring timeout")
        message = await _receive_message(websocket, remaining)
        if message["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(message.get("code", 1000))
        text = message.get("text")
        if text is None:
            data = message.get("bytes")
            if data is not None and len(data) != PCM_FRAME_BYTES:
                raise WireProtocolError("invalid PCM frame size")
            continue
        rate.record(time.monotonic())
        control = decode_phone_control(text)
        if control is not None and control["type"] == "ring":
            return control


async def _receive_loop(
    websocket: WebSocket, inbound: InboundBuffer, rate: _ControlRate
) -> None:
    last_valid = time.monotonic()
    while True:
        remaining = IDLE_SECONDS - (time.monotonic() - last_valid)
        if remaining <= 0:
            raise WireProtocolError("idle timeout")
        message = await _receive_message(websocket, remaining)
        if message["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(message.get("code", 1000))
        text = message.get("text")
        if text is not None:
            rate.record(time.monotonic())
            control = decode_phone_control(text)
            if control is None:
                last_valid = time.monotonic()
                continue
            item: Any = control
        else:
            item = message.get("bytes")
            if item is None:
                raise WireProtocolError("invalid WebSocket message")
            if len(item) != PCM_FRAME_BYTES:
                raise WireProtocolError("invalid PCM frame size")
        await run_in_threadpool(inbound.put, item)
        last_valid = time.monotonic()


async def _send_loop(websocket: WebSocket, outbound: OutboundBuffer) -> None:
    while True:
        try:
            item = await run_in_threadpool(outbound.get)
        except BufferClosed:
            return
        if type(item) is bytes:
            if len(item) != PCM_FRAME_BYTES:
                raise WireProtocolError("invalid PCM frame size")
            await websocket.send_bytes(item)
        else:
            await websocket.send_text(encode_server_control(item))


async def _run_call(
    websocket: WebSocket,
    ring: dict[str, Any],
    buffers: CallBuffers,
    rate: _ControlRate,
) -> None:
    runner = _CALL_RUNNER
    assert runner is not None

    receiver = asyncio.create_task(
        _receive_loop(websocket, buffers.inbound, rate)
    )
    sender = asyncio.create_task(_send_loop(websocket, buffers.outbound))
    runner_task = asyncio.create_task(run_in_threadpool(runner, ring, buffers))
    graceful = False
    failure: BaseException | None = None
    try:
        done, _ = await asyncio.wait(
            {receiver, sender, runner_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in done:
            if not task.cancelled() and task.exception() is not None:
                failure = task.exception()
                break
        graceful = failure is None and (
            runner_task in done or sender in done
        )
    finally:
        await run_in_threadpool(buffers.inbound.close)
        if graceful:
            await run_in_threadpool(buffers.outbound.finish)
            receiver.cancel()
            done, _ = await asyncio.wait(
                {sender}, timeout=TERMINAL_GRACE_SECONDS
            )
            if not done:
                await run_in_threadpool(buffers.outbound.abort)
        else:
            await run_in_threadpool(buffers.outbound.abort)

        for task in (receiver, sender):
            if not task.done():
                task.cancel()
        shuttle_results = await asyncio.gather(
            receiver, sender, return_exceptions=True
        )
        # Closing both buffers is the runner's termination contract.  Keep the
        # process-wide call slot until it has actually honored that contract.
        runner_result = await asyncio.shield(
            asyncio.gather(runner_task, return_exceptions=True)
        )
        if failure is None:
            for result in (*shuttle_results, *runner_result):
                if isinstance(result, BaseException) and not isinstance(
                    result,
                    (BufferClosed, WebSocketDisconnect,
                     asyncio.CancelledError),
                ):
                    failure = result
                    break

    if failure is not None:
        raise failure


async def _close_websocket(websocket: WebSocket, code: int) -> None:
    """Bound a final close handshake after call-local workers are stopped."""
    try:
        await asyncio.wait_for(
            websocket.close(code=code), TERMINAL_GRACE_SECONDS
        )
    except (TimeoutError, RuntimeError, WebSocketDisconnect):
        pass


async def _close_buffers(buffers: CallBuffers) -> None:
    """Wake synchronous workers without taking their locks on the loop."""
    await run_in_threadpool(buffers.inbound.close)
    await run_in_threadpool(buffers.outbound.abort)


@app.websocket("/call")
async def call(websocket: WebSocket) -> None:
    secret = os.getenv("ASSIST_VOICE_SECRET")
    authorization = websocket.headers.get("authorization", "")
    expected = f"Bearer {secret}" if secret else ""
    if not expected or not secrets.compare_digest(
        authorization.encode("utf-8"), expected.encode("utf-8")
    ):
        await _close_websocket(websocket, 1008)
        return
    if not _claim_call():
        await _close_websocket(websocket, 1013)
        return

    buffers = CallBuffers(InboundBuffer(), OutboundBuffer())
    close_code: int | None = 1000
    try:
        await websocket.accept()
        async with asyncio.timeout(MAX_CONNECTION_SECONDS):
            rate = _ControlRate()
            ring = await _receive_ring(websocket, rate)
            if _CALL_RUNNER is None:
                close_code = 1011
            else:
                await _run_call(websocket, ring, buffers, rate)
    except WebSocketDisconnect:
        close_code = None
    except WireProtocolError:
        close_code = 1008
    except TimeoutError:
        close_code = 1008
    except Exception:
        logger.exception("voice call transport failed")
        close_code = 1011
    finally:
        await _close_buffers(buffers)
        # _run_call joins its synchronous runner before returning. Release the
        # slot before the terminal close tells the client it can reconnect.
        _release_call()
        if close_code is not None:
            await _close_websocket(websocket, close_code)
