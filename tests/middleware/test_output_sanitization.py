"""Tests for OutputSanitizationMiddleware.

The middleware strips ANSI CSI sequences and stray control chars from
ToolMessage content via the wrap_tool_call after-path, so the sanitized
version is what lands in state["messages"].
"""
from unittest.mock import Mock

from langchain_core.messages import AIMessage, ToolMessage

from assist.middleware.output_sanitization import (
    OutputSanitizationMiddleware,
    _sanitize,
    _CSI_RE,
    _CONTROL_RE,
)


# ---------------------------------------------------------------------------
# Pure-regex tests
# ---------------------------------------------------------------------------

def test_csi_regex_strips_sgr_color():
    """Standard SGR color sequence — most common case."""
    assert _CSI_RE.sub("", "\x1b[31mred\x1b[0m") == "red"


def test_csi_regex_strips_clear_screen():
    """`\\x1b[2J` — clear screen.  Old narrow regex missed this."""
    assert _CSI_RE.sub("", "\x1b[2Jhello") == "hello"


def test_csi_regex_strips_cursor_moves():
    """Cursor move sequences — A/B/C/D/G/H/F."""
    s = "before\x1b[2A\x1b[5Gafter"
    assert _CSI_RE.sub("", s) == "beforeafter"


def test_csi_regex_strips_24bit_color_with_colon():
    """24-bit color forms use `:` as a parameter byte.  Old `[0-9;]*` regex missed these."""
    s = "\x1b[38:2:255:128:0morange\x1b[0m"
    assert _CSI_RE.sub("", s) == "orange"


def test_csi_regex_strips_private_mode():
    """DEC private mode set/reset (h/l with `?` prefix)."""
    s = "\x1b[?25l hidden \x1b[?25h"
    assert _CSI_RE.sub("", s) == " hidden "


def test_csi_regex_preserves_normal_text():
    """No false positives on text that contains `[` or numbers."""
    s = "list[0] = items[1]; cost: $10"
    assert _CSI_RE.sub("", s) == s


def test_control_regex_strips_null_and_bel():
    """NUL and BEL — break JSON serialization on some endpoints."""
    assert _CONTROL_RE.sub("", "hi\x00there\x07!") == "hithere!"


def test_control_regex_keeps_whitespace():
    """\\n \\r \\t are valid JSON whitespace; must be preserved."""
    s = "line1\nline2\rline3\tcol"
    assert _CONTROL_RE.sub("", s) == s


def test_sanitize_combines_both():
    """_sanitize applies CSI strip then control strip in order."""
    s = "\x1b[31mERROR\x1b[0m\x00 details"
    assert _sanitize(s) == "ERROR details"


def test_sanitize_no_change_returns_same_text():
    """Idempotent on clean input."""
    s = "plain text with no escapes"
    assert _sanitize(s) == s


# ---------------------------------------------------------------------------
# Middleware behavior tests
# ---------------------------------------------------------------------------

def _make_result_with_messages(messages):
    """Construct a minimal mock that mimics the wrap_tool_call handler return.

    The middleware uses `result.messages` and `result.model_copy(update=...)`.
    """
    result = Mock()
    result.messages = messages
    # model_copy returns a new mock with the messages updated
    def model_copy(update=None):
        new = Mock()
        new.messages = update.get("messages", messages) if update else messages
        return new
    result.model_copy = model_copy
    return result


def test_middleware_strips_ansi_from_tool_message():
    mw = OutputSanitizationMiddleware()
    msg = ToolMessage(content="\x1b[31mred error\x1b[0m output", tool_call_id="c1")
    result = _make_result_with_messages([msg])
    handler = Mock(return_value=result)
    request = Mock()

    out = mw.wrap_tool_call(request, handler)

    assert out.messages[0].content == "red error output"
    handler.assert_called_once_with(request)


def test_middleware_leaves_clean_tool_message_unmodified():
    """No sanitization needed → don't allocate a new message."""
    mw = OutputSanitizationMiddleware()
    msg = ToolMessage(content="clean output\nwith newlines\tand tabs", tool_call_id="c1")
    result = _make_result_with_messages([msg])
    handler = Mock(return_value=result)

    out = mw.wrap_tool_call(Mock(), handler)

    # Returned the same result object (no model_copy fired because nothing changed)
    assert out is result


def test_middleware_only_touches_tool_messages():
    """AIMessage content with ANSI is left alone — that's BadRequestRetry's job
    (defense in depth)."""
    mw = OutputSanitizationMiddleware()
    ai = AIMessage(content="\x1b[31m[thinking]\x1b[0m")  # would be unusual but possible
    tool = ToolMessage(content="\x1b[31merror\x1b[0m", tool_call_id="c1")
    result = _make_result_with_messages([ai, tool])
    handler = Mock(return_value=result)

    out = mw.wrap_tool_call(Mock(), handler)

    # AI message unchanged (ANSI still in content)
    assert out.messages[0].content == "\x1b[31m[thinking]\x1b[0m"
    # Tool message sanitized
    assert out.messages[1].content == "error"


def test_middleware_handles_no_messages_attribute():
    """Some handler returns may not have a messages list — don't crash."""
    mw = OutputSanitizationMiddleware()
    result = Mock(spec=[])  # no .messages attribute
    handler = Mock(return_value=result)

    out = mw.wrap_tool_call(Mock(), handler)

    assert out is result


def test_middleware_handles_non_string_content():
    """Some ToolMessages have list-of-blocks content; skip those (don't crash)."""
    mw = OutputSanitizationMiddleware()
    msg = ToolMessage(content=[{"type": "text", "text": "hi"}], tool_call_id="c1")
    result = _make_result_with_messages([msg])
    handler = Mock(return_value=result)

    out = mw.wrap_tool_call(Mock(), handler)

    # Returned as-is (we only mutate string content)
    assert out is result


def test_middleware_swallows_sanitizer_exceptions():
    """A bug in _sanitize must NEVER block the tool-result path."""
    mw = OutputSanitizationMiddleware()

    class WeirdContent:
        """Pretends to be a string but raises on regex sub."""
        def __str__(self):
            raise RuntimeError("nope")

    msg = ToolMessage(content="hello", tool_call_id="c1")
    msg_broken = Mock(spec=ToolMessage)
    msg_broken.content = "hi"
    # Patch isinstance check — bypass with a real ToolMessage that has
    # a content attribute the regex chokes on.  Easier: just sanity-
    # check the exception swallow path by patching _sanitize.
    import assist.middleware.output_sanitization as mod
    original_sanitize = mod._sanitize
    mod._sanitize = lambda _: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        result = _make_result_with_messages([msg])
        handler = Mock(return_value=result)
        out = mw.wrap_tool_call(Mock(), handler)
        # Returned the original result (untouched) because sanitizer raised
        assert out is result
    finally:
        mod._sanitize = original_sanitize
