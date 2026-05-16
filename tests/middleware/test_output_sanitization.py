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


def test_sanitize_tool_message_passes_clean_list_content_through():
    """ToolMessage with list-of-blocks content that has nothing to sanitize
    returns the same instance (no allocation when no change needed)."""
    msg = ToolMessage(
        content=[{"type": "text", "text": "hi"}],
        tool_call_id="c1",
    )
    out = _sanitize_tool_message(msg)
    assert out is msg


def test_sanitize_tool_message_handles_unknown_content_shape():
    """Bytes / None / other unknown content types pass through untouched."""
    msg = ToolMessage(content=b"raw bytes", tool_call_id="c1")
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


def test_command_preserves_goto_on_rebuild():
    """When sanitization triggers a Command rebuild, goto must survive.
    Otherwise we silently break tools that combine state updates with
    parent-graph handoffs."""
    from langgraph.types import Send
    mw = OutputSanitizationMiddleware()
    dirty = ToolMessage(content="\x1b[31merror\x1b[0m", tool_call_id="c1")
    cmd = Command(
        update={"messages": [dirty], "other": 42},
        goto=[Send("next_node", {"x": 1})],
    )
    out = mw.wrap_tool_call(request=None, handler=_identity_handler(cmd))
    assert isinstance(out, Command)
    # Sanitization happened
    assert out.update["messages"][0].content == "error"
    # Control-flow + other update fields preserved
    assert out.goto == cmd.goto
    assert out.update["other"] == 42


def test_command_preserves_graph_and_resume_on_rebuild():
    mw = OutputSanitizationMiddleware()
    dirty = ToolMessage(content="\x1b[31merror\x1b[0m", tool_call_id="c1")
    cmd = Command(
        update={"messages": [dirty]},
        graph="PARENT",
        resume={"foo": "bar"},
    )
    out = mw.wrap_tool_call(request=None, handler=_identity_handler(cmd))
    assert out.graph == "PARENT"
    assert out.resume == {"foo": "bar"}
    assert out.update["messages"][0].content == "error"


def test_list_content_sanitizes_text_blocks():
    """ToolMessage.content can be list[str | dict].  ANSI in a text
    block must be stripped — Copilot review #3 caught that the old
    early-return on non-str content was a hole."""
    mw = OutputSanitizationMiddleware()
    msg = ToolMessage(
        content=[
            {"type": "text", "text": "\x1b[31mERROR\x1b[0m line 1"},
            {"type": "text", "text": "clean line 2"},
            {"type": "image", "url": "..."},  # non-text block passes through
        ],
        tool_call_id="c1",
    )
    out = mw.wrap_tool_call(request=None, handler=_identity_handler(msg))
    assert out is not msg
    assert out.content[0] == {"type": "text", "text": "ERROR line 1"}
    assert out.content[1] == {"type": "text", "text": "clean line 2"}
    assert out.content[2] == {"type": "image", "url": "..."}


def test_list_content_clean_passes_through():
    """No-op when nothing in the list needs sanitizing."""
    mw = OutputSanitizationMiddleware()
    msg = ToolMessage(
        content=[{"type": "text", "text": "all clean"}, {"type": "image", "url": "x"}],
        tool_call_id="c1",
    )
    out = mw.wrap_tool_call(request=None, handler=_identity_handler(msg))
    assert out is msg


def test_list_content_bare_strings_in_list():
    """ToolMessage.content can also be list[str] (without dict wrappers)."""
    mw = OutputSanitizationMiddleware()
    msg = ToolMessage(
        content=["\x1b[31mred\x1b[0m", "clean"],
        tool_call_id="c1",
    )
    out = mw.wrap_tool_call(request=None, handler=_identity_handler(msg))
    assert out.content == ["red", "clean"]


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

async def _async_identity_handler(return_value):
    """An async handler stub for awrap_tool_call tests."""
    async def handler(request):
        return return_value
    return handler


def test_awrap_sanitizes_bare_tool_message():
    """Async path must also sanitize.  Research/context subagents call
    via the async path (ReferencesCleanupRunnable.ainvoke)."""
    import asyncio
    mw = OutputSanitizationMiddleware()
    dirty = ToolMessage(content="\x1b[31mred\x1b[0m", tool_call_id="c1")

    async def run():
        handler = await _async_identity_handler(dirty)
        return await mw.awrap_tool_call(request=None, handler=handler)

    out = asyncio.run(run())
    assert isinstance(out, ToolMessage)
    assert out.content == "red"


def test_awrap_sanitizes_command():
    import asyncio
    mw = OutputSanitizationMiddleware()
    dirty = ToolMessage(content="\x1b[2Jerror", tool_call_id="c1")
    cmd = Command(update={"messages": [dirty]})

    async def run():
        handler = await _async_identity_handler(cmd)
        return await mw.awrap_tool_call(request=None, handler=handler)

    out = asyncio.run(run())
    assert isinstance(out, Command)
    assert out.update["messages"][0].content == "error"


def test_awrap_passes_through_clean():
    import asyncio
    mw = OutputSanitizationMiddleware()
    clean = ToolMessage(content="clean", tool_call_id="c1")

    async def run():
        handler = await _async_identity_handler(clean)
        return await mw.awrap_tool_call(request=None, handler=handler)

    out = asyncio.run(run())
    assert out is clean


def test_middleware_is_wired_into_create_agent_path():
    """Confirm OutputSanitizationMiddleware is *instantiated* in each
    agent factory.  Guards against a future refactor that drops the
    wiring.  Checks for `OutputSanitizationMiddleware(` (with paren) so
    a comment-only mention (e.g., the `mw = [..., OutputSanitization...]`
    docstring) doesn't falsely satisfy the test."""
    import inspect
    from assist import agent as agent_module
    factories = ["create_agent", "create_context_agent", "create_research_agent"]
    for factory in factories:
        fn = getattr(agent_module, factory)
        fsrc = inspect.getsource(fn)
        # Strip comment lines so a comment mentioning the class name
        # doesn't satisfy the check — only real `OutputSanitizationMiddleware(`
        # call sites count.
        code_lines = [
            line for line in fsrc.split("\n")
            if not line.lstrip().startswith("#")
        ]
        code = "\n".join(code_lines)
        assert "OutputSanitizationMiddleware(" in code, (
            f"OutputSanitizationMiddleware() not instantiated in {factory} — "
            f"future refactor regression?"
        )
