"""Unit tests for WriteCollisionMiddleware.

These tests run without an LLM — they construct synthetic
`ToolCallRequest` / `ToolMessage` pairs and verify that the middleware
rewrites collision errors and passes everything else through unchanged.

Companion to the LLM-driven regression in `test_memory.py` (memory-trap
trip).  See docs/2026-04-27-write-file-recoverable-plan.org §Eval plan.
"""
import asyncio
from unittest import TestCase
from unittest.mock import MagicMock

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from assist.middleware.write_collision import (
    WriteCollisionMiddleware,
    _extract_path,
    _rewrite_message,
)


# Verbatim from deepagents/backends/{state,store,filesystem}.py.
_STATE_BACKEND_ERROR = (
    "Cannot write to /AGENTS.md because it already exists. "
    "Read and then make an edit, or write to a new path."
)

# Verbatim from deepagents/backends/sandbox.py:73 — note the `repr()`
# produces single-quoted paths.
_SANDBOX_BACKEND_ERROR = "Error: File already exists: '/workspace/report.md'"


def _make_request(tool_name: str, tool_call_id: str = "call_1") -> MagicMock:
    """Build a minimal `ToolCallRequest`-like object the middleware can read."""
    request = MagicMock()
    request.tool = MagicMock()
    request.tool.name = tool_name
    request.tool_call = {"name": tool_name, "id": tool_call_id, "args": {}}
    return request


def _make_tool_message(content: str, tool_call_id: str = "call_1",
                      name: str = "write_file", status: str = "success") -> ToolMessage:
    """Build a ToolMessage.  `status` must be 'success' or 'error' per
    langchain-core ≥0.3 — None is rejected.  Default is 'success'; tests
    that assert on error-shaped results pass status='error' explicitly.
    """
    return ToolMessage(
        content=content,
        tool_call_id=tool_call_id,
        name=name,
        status=status,
    )


class TestExtractPath(TestCase):
    """The path-extraction helper drives the rewrite — test it independently."""

    def test_state_backend_shape(self):
        self.assertEqual(_extract_path(_STATE_BACKEND_ERROR), "/AGENTS.md")

    def test_sandbox_backend_shape_strips_quotes(self):
        # repr('/workspace/report.md') -> "'/workspace/report.md'"
        # The middleware must strip the surrounding quotes.
        self.assertEqual(
            _extract_path(_SANDBOX_BACKEND_ERROR),
            "/workspace/report.md",
        )

    def test_double_quoted_path(self):
        # repr() can produce double quotes if the path contains a single quote.
        self.assertEqual(
            _extract_path('Error: File already exists: "/foo/bar.md"'),
            "/foo/bar.md",
        )

    def test_path_with_whitespace_state_shape(self):
        # `\S+` would have truncated at the space; the non-greedy + literal
        # terminator captures the full path.
        self.assertEqual(
            _extract_path(
                "Cannot write to /notes/My Notes.md because it already exists. "
                "Read and then make an edit, or write to a new path."
            ),
            "/notes/My Notes.md",
        )

    def test_path_with_whitespace_sandbox_shape(self):
        # repr('/notes/My Notes.md') -> "'/notes/My Notes.md'".
        self.assertEqual(
            _extract_path("Error: File already exists: '/notes/My Notes.md'"),
            "/notes/My Notes.md",
        )

    def test_unrelated_error_returns_none(self):
        self.assertIsNone(_extract_path("Some unrelated error message"))

    def test_anchored_at_start(self):
        # A collision-shaped fragment in the middle of an unrelated message
        # should not match — the regex is anchored with `^`.
        self.assertIsNone(
            _extract_path("Tool finished. Cannot write to X because it already exists.")
        )


class TestRewriteMessage(TestCase):
    """The rewritten error has specific small-LLM-friendly properties."""

    def test_repeats_path_for_emphasis(self):
        msg = _rewrite_message("/AGENTS.md")
        # Path appears at least 4× by design (small-LLM review).  Heavy
        # repetition is the lever; exact count is implementation detail.
        self.assertGreaterEqual(msg.count("/AGENTS.md"), 4)

    def test_names_edit_file_not_write_file(self):
        msg = _rewrite_message("/x.md")
        self.assertIn("edit_file", msg)
        self.assertIn("read_file", msg)
        # The rewritten message must not nudge the model toward write_file again.
        self.assertNotIn("write to a new path", msg)
        self.assertNotIn("different filename", msg)

    def test_includes_example_call(self):
        msg = _rewrite_message("/x.md")
        self.assertIn('edit_file(path="/x.md"', msg)


class TestWriteCollisionMiddleware(TestCase):
    """End-to-end: the middleware should rewrite collisions and pass-through everything else."""

    def setUp(self):
        self.mw = WriteCollisionMiddleware()

    def _wrap_with(self, tool_name: str, handler_returns):
        """Invoke wrap_tool_call with a stub handler returning `handler_returns`."""
        request = _make_request(tool_name)
        handler = MagicMock(return_value=handler_returns)
        return self.mw.wrap_tool_call(request, handler), handler

    def test_rewrites_state_backend_collision(self):
        original = _make_tool_message(_STATE_BACKEND_ERROR, status="error")
        result, _ = self._wrap_with("write_file", original)
        self.assertIsInstance(result, ToolMessage)
        self.assertNotEqual(result.content, _STATE_BACKEND_ERROR)
        self.assertIn("edit_file", result.content)
        self.assertIn("/AGENTS.md", result.content)

    def test_rewrites_sandbox_backend_collision(self):
        original = _make_tool_message(_SANDBOX_BACKEND_ERROR, status="error")
        result, _ = self._wrap_with("write_file", original)
        self.assertIsInstance(result, ToolMessage)
        self.assertIn("/workspace/report.md", result.content)
        # Path must be unquoted in the rewritten message — small-LLM review.
        self.assertNotIn("'/workspace/report.md'", result.content)

    def test_rewritten_message_keeps_status_error(self):
        # Loop-detection's Pattern C requires status=error on at least one
        # event in the window.  The rewrite must preserve it.
        original = _make_tool_message(_STATE_BACKEND_ERROR, status="error")
        result, _ = self._wrap_with("write_file", original)
        self.assertEqual(result.status, "error")

    def test_rewritten_message_preserves_tool_call_id(self):
        original = _make_tool_message(
            _STATE_BACKEND_ERROR, tool_call_id="abc123", status="error"
        )
        result, _ = self._wrap_with("write_file", original)
        self.assertEqual(result.tool_call_id, "abc123")

    def test_passes_through_successful_write(self):
        original = _make_tool_message("Updated file /AGENTS.md")
        result, _ = self._wrap_with("write_file", original)
        self.assertIs(result, original)  # no copy made

    def test_passes_through_unrelated_error(self):
        original = _make_tool_message("Error: permission denied", status="error")
        result, _ = self._wrap_with("write_file", original)
        self.assertIs(result, original)

    def test_passes_through_other_tools(self):
        # An edit_file error that happens to mention "already exists" must
        # not be rewritten — only write_file is in scope.
        original = _make_tool_message(
            _STATE_BACKEND_ERROR, name="edit_file", status="error"
        )
        result, handler = self._wrap_with("edit_file", original)
        self.assertIs(result, original)
        handler.assert_called_once()

    def test_passes_through_command_results(self):
        # If a write_file handler returns a Command (rare but possible if a
        # future backend emits state updates), pass it through unchanged.
        cmd = Command(update={"messages": []})
        result, _ = self._wrap_with("write_file", cmd)
        self.assertIs(result, cmd)

    def test_passes_through_non_string_content(self):
        # Some content fields are list-of-parts; we only act on plain str.
        original = ToolMessage(
            content=[{"type": "text", "text": _STATE_BACKEND_ERROR}],
            tool_call_id="call_1",
            name="write_file",
            status="error",
        )
        result, _ = self._wrap_with("write_file", original)
        self.assertIs(result, original)

    def test_async_path_rewrites_collision(self):
        # awrap_tool_call must mirror wrap_tool_call.  Confirm with a real
        # async handler stub.
        original = _make_tool_message(_STATE_BACKEND_ERROR, status="error")

        async def handler(_request):
            return original

        request = _make_request("write_file")
        result = asyncio.run(self.mw.awrap_tool_call(request, handler))
        self.assertIsInstance(result, ToolMessage)
        self.assertIn("edit_file", result.content)
        self.assertIn("/AGENTS.md", result.content)
