"""Unit tests for ToolResultToFileMiddleware — the bounded-by-construction seam.

Drives wrap_tool_call with a fake tool ToolMessage + a stub backend and asserts:
the model-visible result is small (preview + path + grep instruction) regardless of
size, and the stub backend received the FULL content.  Un-mocked at the seam that
matters (no LLM).
"""
from types import SimpleNamespace

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from assist.middleware.tool_result_to_file import (
    ToolResultToFileMiddleware, _DEFAULT_FLOOR_CHARS, _PREVIEW_CHARS, UNTRUSTED_OFFLOAD_MARK)


class _StubBackend:
    def __init__(self, existing=False):
        self.written = {}
        self._existing = existing

    def write(self, path, content):
        self.written[path] = content
        # mimic StateBackend.write's return shape (has .error / .path)
        return SimpleNamespace(error=("exists" if self._existing else None), path=None)


class _Req:
    def __init__(self, name="read_url", **args):
        self.tool_call = {"name": name, "args": args or {"url": "https://example.com/x"}}


def _msg(content, tcid="call_1"):
    return ToolMessage(content=content, tool_call_id=tcid)


def _mw(backend, tools=frozenset({"read_url"}), floor=4000, style="head"):
    return ToolResultToFileMiddleware(backend, tools=tools, floor_chars=floor, preview_style=style)


def _run(mw, req, result):
    return mw.wrap_tool_call(req, lambda r: result)


def test_large_result_offloaded_to_file_and_bounded():
    backend = _StubBackend()
    full = "The capacity is 88,204 seats. " + ("filler text. " * 3000)   # >> floor
    out = _run(_mw(backend), _Req(), _msg(full))
    # the file got the FULL content:
    assert len(backend.written) == 1
    path, written = next(iter(backend.written.items()))
    assert path.startswith("/large_tool_results/")
    assert "88,204" in written and len(written) == len(full)
    # the model-visible message is BOUNDED (preview + path + grep instruction):
    assert isinstance(out, ToolMessage)
    assert len(out.content) < _PREVIEW_CHARS + 500
    assert path in out.content and "grep(" in out.content


def test_untrusted_offload_marks_the_path():
    # untrusted=True writes to /large_tool_results/untrusted-<id> so UrlProvenanceMiddleware
    # can recognize a later read/grep of it and never provenance its URLs. The model still
    # gets the exact (untrusted-) path to grep.
    backend = _StubBackend()
    mw = ToolResultToFileMiddleware(backend, tools=frozenset({"read_url"}),
                                    floor_chars=4000, untrusted=True)
    out = _run(mw, _Req(), _msg("x" * 5000))
    path = next(iter(backend.written))
    assert path.startswith(f"/{UNTRUSTED_OFFLOAD_MARK}"), path
    assert path in out.content and "grep(" in out.content


def test_trusted_offload_has_no_untrusted_marker():
    # Default (untrusted=False) keeps the plain /large_tool_results/<id> path — no mixing:
    # a trusted large offload is NOT flagged as untrusted.
    backend = _StubBackend()
    out = _run(_mw(backend), _Req(), _msg("y" * 5000))
    path = next(iter(backend.written))
    assert path.startswith("/large_tool_results/") and "untrusted-" not in path, path


def test_small_result_returned_inline():
    backend = _StubBackend()
    small = "x" * (_DEFAULT_FLOOR_CHARS - 1)
    out = _run(_mw(backend), _Req(), _msg(small))
    assert out.content == small and not backend.written   # untouched, no file


def test_short_error_stays_inline_under_floor():
    # A read_url error string is short → under the floor → inline (no error-prefix
    # special-case needed now; the floor is the tool-agnostic gate).
    backend = _StubBackend()
    err = "Error fetching URL: timeout"
    out = _run(_mw(backend), _Req(), _msg(err))
    assert out.content == err and not backend.written


def test_tool_not_in_allowlist_untouched():
    backend = _StubBackend()
    big = "y" * 50000
    out = _run(_mw(backend, tools=frozenset({"read_url"})), _Req(name="search_internet"), _msg(big))
    assert out.content == big and not backend.written   # positive allowlist: not listed → skip


def test_write_error_falls_back_to_original():
    backend = _StubBackend(existing=True)   # write returns an error
    full = "z" * 50000
    out = _run(_mw(backend), _Req(), _msg(full))
    assert out.content == full   # fell back to the original, not a broken preview


def test_command_result_passes_through():
    backend = _StubBackend()
    cmd = Command(update={"messages": []})
    out = _run(_mw(backend), _Req(), cmd)
    assert out is cmd and not backend.written


def test_execute_tool_offloaded_with_head_tail_preview():
    # execute on its own floor (8000) + head_tail preview: the salient line is at the
    # TAIL of a log, so the preview must include the tail.
    backend = _StubBackend()
    tail_value = "BUILD FAILED: error code ZX-99741"
    log = ("compiling module... ok\n" * 2000) + tail_value   # ~44k chars, value at the end
    out = _run(_mw(backend, tools=frozenset({"execute"}), floor=8000, style="head_tail"),
               _Req(name="execute", command="make build"), _msg(log))
    # full log to file:
    path, written = next(iter(backend.written.items()))
    assert tail_value in written
    # the head_tail preview shows the TAIL (where the value is), and points to the file:
    assert tail_value in out.content and path in out.content and "grep(" in out.content
    assert "make build" in out.content   # the command is surfaced as the source


def test_offload_root_tmp_writes_real_fs_path():
    # offload_root="/tmp" (the sandbox-backed execute instance) lands the file on the
    # real /tmp bind-mount — one path the model can shell-`cat` AND grep — while keeping
    # the untrusted- marker so UrlProvenanceMiddleware still recognizes a read of it.
    backend = _StubBackend()
    mw = ToolResultToFileMiddleware(backend, tools=frozenset({"execute"}),
                                    floor_chars=8000, untrusted=True, offload_root="/tmp")
    out = _run(mw, _Req(name="execute", command="make build"), _msg("x" * 9000))
    path = next(iter(backend.written))
    assert path == f"/tmp/{UNTRUSTED_OFFLOAD_MARK}call_1", path
    assert path in out.content and "grep(" in out.content
