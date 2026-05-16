"""Tests for OutputSanitizationMiddleware.

The middleware strips ANSI CSI sequences and stray control chars from
ToolMessage content in the wrap_tool_call after-path so the sanitized
version is what lands in state["messages"].

Critical: the langchain `wrap_tool_call` handler returns either a bare
`ToolMessage` or a `langgraph.types.Command(update={"messages": [...]})`.
NOT an object with `.messages`.  A prior version of this test used Mock
objects with a fabricated `.messages` attribute and the middleware
silently no-op'd in production — the type contract matters.
"""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from assist.middleware.output_sanitization import (
    OutputSanitizationMiddleware,
    _sanitize,
    _sanitize_tool_message,
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
# _sanitize_tool_message helper tests
# ---------------------------------------------------------------------------

def test_sanitize_tool_message_strips_ansi():
    msg = ToolMessage(content="\x1b[31merror\x1b[0m output", tool_call_id="c1")
    out = _sanitize_tool_message(msg)
    assert out is not msg
    assert out.content == "error output"
    assert out.tool_call_id == "c1"


def test_sanitize_tool_message_no_change_returns_same_instance():
    msg = ToolMessage(content="clean output", tool_call_id="c1")
    out = _sanitize_tool_message(msg)
    assert out is msg


def test_sanitize_tool_message_skips_list_content():
    """ToolMessage with list-of-blocks content is passed through untouched."""
    msg = ToolMessage(
        content=[{"type": "text", "text": "hi"}],
        tool_call_id="c1",
    )
    out = _sanitize_tool_message(msg)
    assert out is msg


# ---------------------------------------------------------------------------
# Middleware behavior tests — using REAL types per wrap_tool_call contract
# ---------------------------------------------------------------------------

def _identity_handler(return_value):
    """A handler stub: ignores the request, returns the given value."""
    return lambda request: return_value


def test_middleware_sanitizes_bare_tool_message_return():
    """Most common: handler returns a bare ToolMessage."""
    mw = OutputSanitizationMiddleware()
    dirty = ToolMessage(content="\x1b[31mred\x1b[0m", tool_call_id="c1")
    out = mw.wrap_tool_call(request=None, handler=_identity_handler(dirty))
    assert isinstance(out, ToolMessage)
    assert out.content == "red"
    assert out.tool_call_id == "c1"


def test_middleware_passes_through_clean_tool_message():
    mw = OutputSanitizationMiddleware()
    clean = ToolMessage(content="hello", tool_call_id="c1")
    out = mw.wrap_tool_call(request=None, handler=_identity_handler(clean))
    assert out is clean  # no copy made


def test_middleware_sanitizes_command_with_tool_messages():
    """Less common but real: handler returns Command(update={'messages': [...]})."""
    mw = OutputSanitizationMiddleware()
    dirty = ToolMessage(content="\x1b[31merror\x1b[0m", tool_call_id="c1")
    cmd = Command(update={"messages": [dirty], "files": {"a.txt": "x"}})
    out = mw.wrap_tool_call(request=None, handler=_identity_handler(cmd))
    assert isinstance(out, Command)
    assert out.update["messages"][0].content == "error"
    assert out.update["messages"][0].tool_call_id == "c1"
    # Other update keys preserved
    assert out.update["files"] == {"a.txt": "x"}


def test_middleware_command_with_clean_messages_passes_through():
    mw = OutputSanitizationMiddleware()
    clean = ToolMessage(content="hello", tool_call_id="c1")
    cmd = Command(update={"messages": [clean]})
    out = mw.wrap_tool_call(request=None, handler=_identity_handler(cmd))
    assert out is cmd  # no copy when nothing changed


def test_middleware_command_leaves_ai_messages_alone():
    """AIMessage with ANSI is left alone — that's BadRequestRetry's job
    (defense-in-depth path; this middleware only touches ToolMessage)."""
    mw = OutputSanitizationMiddleware()
    ai = AIMessage(content="\x1b[31m[thinking]\x1b[0m")
    tool = ToolMessage(content="\x1b[31mTOOL ERR\x1b[0m", tool_call_id="c1")
    cmd = Command(update={"messages": [ai, tool]})
    out = mw.wrap_tool_call(request=None, handler=_identity_handler(cmd))
    # AI unchanged (still has ANSI)
    assert out.update["messages"][0].content == "\x1b[31m[thinking]\x1b[0m"
    # Tool sanitized
    assert out.update["messages"][1].content == "TOOL ERR"


def test_middleware_handles_unknown_return_type_gracefully():
    """If handler returns something that isn't ToolMessage or Command,
    pass it through untouched (don't crash)."""
    mw = OutputSanitizationMiddleware()
    weird = "not a message"
    out = mw.wrap_tool_call(request=None, handler=_identity_handler(weird))
    assert out == weird


def test_middleware_swallows_sanitizer_exceptions():
    """A bug in _sanitize must NEVER block the tool-result path."""
    mw = OutputSanitizationMiddleware()
    # Patch the module-level _sanitize_tool_message to raise, simulating
    # a bug in the sanitization path.
    import assist.middleware.output_sanitization as mod
    original = mod._sanitize_tool_message

    def broken(_):
        raise RuntimeError("boom")
    mod._sanitize_tool_message = broken
    try:
        dirty = ToolMessage(content="\x1b[31mred\x1b[0m", tool_call_id="c1")
        out = mw.wrap_tool_call(request=None, handler=_identity_handler(dirty))
        # Returned the original (untouched) because sanitizer raised
        assert out is dirty
    finally:
        mod._sanitize_tool_message = original


# ---------------------------------------------------------------------------
# Wiring tests — verify the middleware is actually in each agent's chain
# ---------------------------------------------------------------------------

def test_middleware_is_wired_into_create_agent_path():
    """Construct the agent module's middleware list and confirm
    OutputSanitizationMiddleware appears in it.  Guards against a
    future refactor that drops the wiring."""
    import inspect
    from assist import agent as agent_module
    # Read the source to confirm OutputSanitizationMiddleware is
    # referenced in each agent factory.  Cheap structural test; avoids
    # spinning up a real ChatModel.
    src = inspect.getsource(agent_module)
    # Three factory functions
    factories = ["create_agent", "create_context_agent", "create_research_agent"]
    for factory in factories:
        fn = getattr(agent_module, factory)
        fsrc = inspect.getsource(fn)
        assert "OutputSanitizationMiddleware" in fsrc, (
            f"OutputSanitizationMiddleware missing from {factory} — "
            f"future refactor regression?"
        )
