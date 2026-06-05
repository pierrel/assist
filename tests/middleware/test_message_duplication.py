"""Regression tests for the middleware message-duplication bug.

All six fixed hooks shared one anti-pattern: when a middleware modified a
message it returned the WHOLE ``messages`` list.  The channel reducer
(``add_messages``) dedupes by ``.id``, but the checkpointer deserializes
``HumanMessage`` / ``ToolMessage`` with ``id=None``.  Because ``after_model``
receives a *separate* deserialized copy of the state, returning the full list
re-appends every id-less message as a duplicate — which is what put a duplicate
trailing ``HumanMessage`` in prod thread 20260605060257-e3f8753c (prompt
rendered twice, looked unanswered).  See
docs/2026-06-05-middleware-message-duplication.org.

Each test reproduces the deserialization gap faithfully: the hook is fed one
set of message objects (``incoming``) and its return is reduced against a
*separate, identical* set (``channel``).  A test that reused the same objects
would pass even against the buggy full-list return, so the separation is the
crux — these tests assert it (left-operand human/tool are ``id=None``).
"""
from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

from assist.middleware.loop_detection import LoopDetectionMiddleware
from assist.middleware.subagent_type_inference import SubagentTypeInferenceMiddleware
from assist.middleware.tool_name_sanitization import ToolNameSanitizationMiddleware
from assist.middleware.json_validation_middleware import JsonValidationMiddleware


def _ai(tc_id, name, args, *, id, content=""):
    """AIMessage with a model-style id and one tool call (mirrors prod)."""
    msg = AIMessage(content=content, id=id)
    msg.tool_calls = [{"id": tc_id, "name": name, "args": args}]
    return msg


def _tool(tc_id, content, status="success"):
    """ToolMessage with id=None, as the checkpointer deserializes it."""
    return ToolMessage(content=content, tool_call_id=tc_id, status=status)


def _human(content):
    return HumanMessage(content=content)  # id=None


def _content_counts(messages):
    """Count messages by (type, content) so duplicates are visible."""
    counts = {}
    for m in messages:
        key = (type(m).__name__, str(m.content))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _assert_deserialization_gap(channel):
    """Pin the precondition the bug needs: id-less human/tool, id'd last AI."""
    for m in channel:
        if isinstance(m, (HumanMessage, ToolMessage)):
            assert m.id is None, f"test setup: {type(m).__name__} should be id-less"
    assert isinstance(channel[-1], AIMessage) and channel[-1].id, (
        "test setup: last message must be an AIMessage carrying an id"
    )


# ---------------------------------------------------------------------------
# Variant A — after_model hooks that modify only the last AIMessage
# ---------------------------------------------------------------------------

class TestAfterModelNoDuplication:
    def test_loop_detection(self):
        # Pattern A: same tool + same error twice, last call extends it.
        def build():
            return [
                _human("do the thing"),
                _ai("c1", "write_file", {"file_path": "/a"}, id="ai-1"),
                _tool("c1", "Error: cannot write /a"),
                _ai("c2", "write_file", {"file_path": "/a"}, id="ai-2"),
                _tool("c2", "Error: cannot write /a"),
                _ai("c3", "write_file", {"file_path": "/a"}, id="ai-3"),
            ]
        channel, incoming = build(), build()
        _assert_deserialization_gap(channel)

        mw = LoopDetectionMiddleware(error_repeat_threshold=2)
        update = mw.after_model({"messages": incoming}, Mock())
        assert update is not None, "loop should have been detected"

        merged = add_messages(channel, update["messages"])
        counts = _content_counts(merged)
        assert counts[("HumanMessage", "do the thing")] == 1
        # The last AI is replaced in place: same count as the channel, the
        # final message has no tool calls (loop terminated).
        assert len(merged) == len(channel)
        assert merged[-1].tool_calls == []

    def test_tool_name_sanitization(self):
        def build():
            return [
                _human("hello"),
                _tool("prev", "earlier result"),
                _ai("bad", "[]", {}, id="ai-1"),  # invalid tool name -> stripped
            ]
        channel, incoming = build(), build()
        _assert_deserialization_gap(channel)

        mw = ToolNameSanitizationMiddleware()
        update = mw.after_model({"messages": incoming}, Mock())
        assert update is not None

        merged = add_messages(channel, update["messages"])
        counts = _content_counts(merged)
        assert counts[("HumanMessage", "hello")] == 1
        assert counts[("ToolMessage", "earlier result")] == 1
        assert len(merged) == len(channel)
        assert merged[-1].tool_calls == []

    def test_subagent_type_inference(self):
        def build():
            return [
                _human("research X"),
                _tool("prev", "earlier result"),
                # A `task` call whose subagent_type is wrong/missing so the
                # middleware patches it -> returns the modified last message.
                _ai("t1", "task",
                    {"description": "go", "subagent_type": "reasearch-agent"},
                    id="ai-1"),
            ]
        channel, incoming = build(), build()
        _assert_deserialization_gap(channel)

        mw = SubagentTypeInferenceMiddleware(valid_subagent_types={"research-agent"})
        update = mw.after_model({"messages": incoming}, Mock())
        assert update is not None, "subagent type should have been inferred/patched"

        merged = add_messages(channel, update["messages"])
        counts = _content_counts(merged)
        assert counts[("HumanMessage", "research X")] == 1
        assert counts[("ToolMessage", "earlier result")] == 1
        assert len(merged) == len(channel)

    def test_json_validation(self):
        # after_model only flags a tool call carrying a raw `function`
        # payload with invalid-JSON `arguments` (see _validate_tool_call);
        # craft exactly that so the fixed return path is exercised.
        def build():
            ai = AIMessage(content="", id="ai-1")
            ai.tool_calls = [{
                "id": "c1", "name": "write_file", "args": {"file_path": "/a"},
                "function": {"name": "write_file",
                             "arguments": '{"file_path": "/a", invalid}'},
            }]
            return [_human("hi"), _tool("prev", "earlier result"), ai]
        channel, incoming = build(), build()
        _assert_deserialization_gap(channel)

        mw = JsonValidationMiddleware(strict=False)
        update = mw.after_model({"messages": incoming}, Mock())
        assert update is not None, "invalid-JSON arguments should have been flagged"

        merged = add_messages(channel, update["messages"])
        counts = _content_counts(merged)
        assert counts[("HumanMessage", "hi")] == 1
        assert counts[("ToolMessage", "earlier result")] == 1
        assert len(merged) == len(channel)
        # The fix returns a COPY (no in-place mutation of the channel object).
        assert incoming[-1] is not merged[-1]


# ---------------------------------------------------------------------------
# Variant B — before_model history scrubs (must REMOVE, not just dedupe)
# ---------------------------------------------------------------------------

class TestBeforeModelScrub:
    def test_tool_name_sanitization_removes_bad_messages(self):
        # History carries an AIMessage with an invalid-named tool call plus
        # its result; the scrub must remove BOTH and not duplicate anything.
        def build():
            ai = AIMessage(content="", id="ai-1")
            ai.tool_calls = [
                {"id": "bad", "name": "[]", "args": {}},
                {"id": "ok", "name": "ls", "args": {"path": "/"}},
            ]
            return [
                _human("hello"),
                ai,
                _tool("bad", "bad result"),
                _tool("ok", "good result"),
            ]
        channel, incoming = build(), build()
        for m in channel:
            if isinstance(m, (HumanMessage, ToolMessage)):
                assert m.id is None

        mw = ToolNameSanitizationMiddleware()
        update = mw.before_model({"messages": incoming}, Mock())
        assert update is not None, "scrub should have fired"

        merged = add_messages(channel, update["messages"])
        counts = _content_counts(merged)
        # Bad tool result is actually gone (the old code left it in place).
        assert ("ToolMessage", "bad result") not in counts
        # Good messages survive exactly once (no duplication).
        assert counts[("HumanMessage", "hello")] == 1
        assert counts[("ToolMessage", "good result")] == 1
        # The surviving AI keeps only the valid call.
        ai_msgs = [m for m in merged if isinstance(m, AIMessage)]
        assert len(ai_msgs) == 1
        assert [tc["name"] for tc in ai_msgs[0].tool_calls] == ["ls"]

    def test_json_validation_scrub_no_duplication(self):
        # A control character in past content triggers before_model
        # sanitization, which returns the rewritten history.
        def build():
            return [
                _human("hello"),
                _tool("c1", "result with \x07 bell control char"),
                AIMessage(content="prior answer", id="ai-1"),
            ]
        channel, incoming = build(), build()
        for m in channel:
            if isinstance(m, (HumanMessage, ToolMessage)):
                assert m.id is None

        mw = JsonValidationMiddleware(strict=False)
        update = mw.before_model({"messages": incoming}, Mock())
        # The \x07 control char deterministically trips content sanitization,
        # so the scrub must fire — a skip here would let the suite pass if the
        # scrub silently stopped working.
        assert update is not None, "control char should have tripped sanitization"

        merged = add_messages(channel, update["messages"])
        counts = _content_counts(merged)
        assert counts[("HumanMessage", "hello")] == 1
        # Exactly the same number of messages — content sanitized, none added.
        assert len(merged) == len(channel)
