"""Unit tests for ReadUrlToFileMiddleware — the bounded-by-construction seam.

Drives wrap_tool_call with a fake read_url ToolMessage + a stub backend and asserts:
the model-visible result is small (preview + path + grep instruction) regardless of
page size, and the stub backend received the FULL content.  Un-mocked at the seam
that matters (no LLM).
"""
from types import SimpleNamespace

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from assist.middleware.read_url_to_file import (
    ReadUrlToFileMiddleware, _OFFLOAD_FLOOR_CHARS, _PREVIEW_CHARS)


class _StubBackend:
    def __init__(self, existing=False):
        self.written = {}
        self._existing = existing

    def write(self, path, content):
        self.written[path] = content
        # mimic StateBackend.write's return shape (has .error / .path)
        return SimpleNamespace(error=("exists" if self._existing else None), path=None)


class _Req:
    def __init__(self, name="read_url", url="https://example.com/x"):
        self.tool_call = {"name": name, "args": {"url": url}}


def _msg(content, tcid="call_1"):
    return ToolMessage(content=content, tool_call_id=tcid)


def _run(mw, req, result):
    return mw.wrap_tool_call(req, lambda r: result)


def test_large_read_url_offloaded_to_file_and_bounded():
    backend = _StubBackend()
    mw = ReadUrlToFileMiddleware(backend)
    full = "The capacity is 88,204 seats. " + ("filler text. " * 3000)   # >> floor
    out = _run(mw, _Req(), _msg(full))
    # the file got the FULL content:
    assert len(backend.written) == 1
    path, written = next(iter(backend.written.items()))
    assert path.startswith("/large_tool_results/")
    assert "88,204" in written and len(written) == len(full)
    # the model-visible message is BOUNDED (preview + path + grep instruction):
    assert isinstance(out, ToolMessage)
    assert len(out.content) < _PREVIEW_CHARS + 500
    assert path in out.content and "grep(" in out.content


def test_small_read_url_returned_inline():
    backend = _StubBackend()
    mw = ReadUrlToFileMiddleware(backend)
    small = "x" * (_OFFLOAD_FLOOR_CHARS - 1)
    out = _run(mw, _Req(), _msg(small))
    assert out.content == small and not backend.written   # untouched, no file


def test_error_string_returned_inline():
    backend = _StubBackend()
    mw = ReadUrlToFileMiddleware(backend)
    err = "Error fetching URL: timeout"
    out = _run(mw, _Req(), _msg(err))
    assert out.content == err and not backend.written


def test_non_read_url_tool_untouched():
    backend = _StubBackend()
    mw = ReadUrlToFileMiddleware(backend)
    big = "y" * 50000
    out = _run(mw, _Req(name="search_internet"), _msg(big))
    assert out.content == big and not backend.written


def test_write_error_falls_back_to_original():
    backend = _StubBackend(existing=True)   # write returns an error
    mw = ReadUrlToFileMiddleware(backend)
    full = "z" * 50000
    out = _run(mw, _Req(), _msg(full))
    assert out.content == full   # fell back to the original, not a broken preview


def test_command_result_passes_through():
    backend = _StubBackend()
    mw = ReadUrlToFileMiddleware(backend)
    cmd = Command(update={"messages": []})
    out = _run(mw, _Req(), cmd)
    assert out is cmd and not backend.written
