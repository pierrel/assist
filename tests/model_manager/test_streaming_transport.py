"""Exercise ChatOpenAI's real HTTP streaming and complete-message aggregation."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from assist.model_manager import _build_openai_chat_model


_READ_TIMEOUT = 0.3
_CHUNK_PAUSE = 0.1


@contextmanager
def _completion_server(chunks: list[dict], *, stall_after_first: bool = False):
    requests = []
    first_chunk_sent = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append(json.loads(body))
            if not requests[-1]["stream"]:
                time.sleep(_READ_TIMEOUT * 2)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                try:
                    self.wfile.write(json.dumps({
                        "id": "complete", "object": "chat.completion", "created": 0,
                        "model": "fixture", "choices": [{"index": 0,
                            "message": {"role": "assistant", "content": "done"},
                            "finish_reason": "stop"}],
                    }).encode())
                except BrokenPipeError:
                    pass
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for index, chunk in enumerate(chunks):
                try:
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                except BrokenPipeError:
                    return
                if index == 0:
                    first_chunk_sent.set()
                time.sleep(_READ_TIMEOUT * 2 if stall_after_first and index == 0
                           else _CHUNK_PAUSE)
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except BrokenPipeError:
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests, first_chunk_sent
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=1)


def _chunk(delta=None, *, finish_reason=None, usage=None):
    return {"id": "streamed", "object": "chat.completion.chunk", "created": 0,
            "model": "fixture", "choices": ([] if usage else [{"index": 0,
                "delta": delta or {}, "finish_reason": finish_reason}]), "usage": usage}


def _model(url):
    return _build_openai_chat_model(
        "fixture", temperature=0, base_url=url, api_key="EMPTY",
        request_timeout=httpx.Timeout(_READ_TIMEOUT),
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_progressive_stream_returns_complete_message_beyond_read_timeout(asynchronous):
    chunks = [
        _chunk({"role": "assistant", "content": "<think>plan</think>"}),
        _chunk({"content": "Hello"}),
        _chunk({"content": ", world."}),
        _chunk(finish_reason="stop"),
        _chunk(usage={"prompt_tokens": 2, "completion_tokens": 7,
                      "total_tokens": 9}),
    ]
    with _completion_server(chunks) as (url, requests, _):
        model = _model(url)
        start = time.monotonic()
        result = (asyncio.run(model.ainvoke("Hi")) if asynchronous
                  else model.invoke("Hi"))
    assert time.monotonic() - start > _READ_TIMEOUT
    assert requests[0]["stream"] is True
    assert requests[0]["stream_options"] == {"include_usage": True}
    assert result.content == "<think>plan</think>Hello, world."
    assert {key: result.usage_metadata[key] for key in
            ("input_tokens", "output_tokens", "total_tokens")} == {
                "input_tokens": 2, "output_tokens": 7, "total_tokens": 9}


def test_stalled_stream_still_times_out():
    with _completion_server([_chunk({"role": "assistant", "content": "partial"}),
                             _chunk({"content": "never"})], stall_after_first=True) as (url, requests, _):
        with pytest.raises(httpx.ReadTimeout):
            _model(url).invoke("Hi")
    assert requests[0]["stream"] is True


def test_tool_call_chunks_aggregate_into_one_complete_call():
    chunks = [
        _chunk({"role": "assistant", "tool_calls": [{"index": 0, "id": "call_1",
            "type": "function", "function": {"name": "calculate", "arguments": '{"x":'}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}),
        _chunk(finish_reason="tool_calls"),
        _chunk(usage={"prompt_tokens": 8, "completion_tokens": 4,
                      "total_tokens": 12}),
    ]
    with _completion_server(chunks) as (url, requests, _):
        result = _model(url).invoke("Calculate")
    assert requests[0]["stream"] is True
    assert result.content == ""
    assert result.tool_calls == [{"name": "calculate", "args": {"x": 1},
                                  "id": "call_1", "type": "tool_call"}]
    assert result.usage_metadata["total_tokens"] == 12


def test_async_stream_can_be_cancelled_after_first_chunk():
    chunks = [_chunk({"role": "assistant", "content": "partial"}),
              _chunk({"content": "never"})]
    with _completion_server(chunks, stall_after_first=True) as (url, requests, first_chunk_sent):
        async def cancel_after_first_chunk():
            task = asyncio.create_task(_model(url).ainvoke("Hi"))
            assert await asyncio.to_thread(first_chunk_sent.wait, 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(cancel_after_first_chunk())
    assert requests[0]["stream"] is True
