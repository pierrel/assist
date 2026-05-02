"""Tests for EmptyResponseRecoveryMiddleware."""
import logging
from unittest.mock import Mock

import pytest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain.agents.middleware.types import ModelRequest, ModelResponse

from assist.middleware.empty_response_recovery import (
    EmptyResponseRecoveryMiddleware,
    _AUGMENT_INSTRUCTION,
    _compose_fallback,
    _content_to_text,
    _is_empty_terminal,
    _strip_think,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ai(content: str = "", tool_calls=None) -> AIMessage:
    msg = AIMessage(content=content)
    if tool_calls:
        msg.tool_calls = tool_calls
    return msg


def _tool_call(tc_id: str, name: str, args: dict) -> dict:
    return {"id": tc_id, "name": name, "args": args}


def _tool_msg(tc_id: str, content: str, status: str = "success") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=tc_id, status=status)


def _request(messages):
    """Minimal ModelRequest stub for middleware tests.

    ``ModelRequest.__init__`` allows ``state``/``runtime`` to be ``None``
    and we don't exercise either field; ``messages`` is the only thing
    the recovery middleware reads.
    """
    return ModelRequest(
        model=Mock(),
        messages=list(messages),
        system_message=None,
        tools=[],
        state=None,
        runtime=None,
    )


def _model_response(msg: AIMessage) -> ModelResponse:
    return ModelResponse(result=[msg])


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------

class TestStripThink:
    def test_removes_simple_block(self):
        assert _strip_think("<think>foo</think>bar") == "bar"

    def test_removes_multiline_block(self):
        text = "<think>line1\nline2\n  line3</think>final answer"
        assert _strip_think(text) == "final answer"

    def test_removes_multiple_blocks(self):
        text = "<think>a</think>x<think>b</think>y"
        assert _strip_think(text) == "xy"

    def test_handles_no_block(self):
        assert _strip_think("plain text") == "plain text"

    def test_handles_unclosed_block_does_not_strip_rest(self):
        # An unclosed <think> would be a regex disaster if non-greedy
        # turned greedy.  We refuse to match it; the trailing content
        # stays so the caller can decide.
        text = "<think>truncated trace ... real content"
        assert _strip_think(text) == text


class TestContentToText:
    def test_str_passthrough(self):
        assert _content_to_text("hello") == "hello"

    def test_none_is_empty(self):
        assert _content_to_text(None) == ""

    def test_list_of_text_parts(self):
        content = [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]
        assert _content_to_text(content) == "hello world"

    def test_list_with_non_text_parts_ignored(self):
        content = [{"type": "image", "url": "x"}, {"type": "text", "text": "ok"}]
        assert _content_to_text(content) == "ok"

    def test_list_with_string_parts(self):
        content = ["a", "b", "c"]
        assert _content_to_text(content) == "abc"


class TestIsEmptyTerminal:
    def test_empty_string_no_tools(self):
        assert _is_empty_terminal(_ai(""))

    def test_only_whitespace_no_tools(self):
        assert _is_empty_terminal(_ai("   \n\t  "))

    def test_only_think_block_no_tools(self):
        assert _is_empty_terminal(_ai("<think>I am done.</think>"))

    def test_think_with_trailing_whitespace_only(self):
        assert _is_empty_terminal(_ai("<think>...</think>\n  \n"))

    def test_short_content_no_tools_is_not_empty(self):
        # Legitimate-short guard: "Done." is a valid response.
        assert not _is_empty_terminal(_ai("Done."))

    def test_think_then_real_content_is_not_empty(self):
        assert not _is_empty_terminal(_ai("<think>plan</think>Here is the answer."))

    def test_empty_with_tool_calls_is_not_terminal(self):
        msg = _ai("", tool_calls=[_tool_call("c1", "ls", {})])
        assert not _is_empty_terminal(msg)

    def test_list_content_with_text_part(self):
        msg = AIMessage(content=[{"type": "text", "text": "  "}])
        assert _is_empty_terminal(msg)

    def test_list_content_with_real_text(self):
        msg = AIMessage(content=[{"type": "text", "text": "real"}])
        assert not _is_empty_terminal(msg)


class TestComposeFallback:
    def test_with_artifact_cites_path(self):
        msgs = [
            HumanMessage(content="write a report"),
            _ai("", tool_calls=[_tool_call("c1", "write_file",
                                          {"file_path": "/workspace/report.md"})]),
            _tool_msg("c1", "ok"),
        ]
        out = _compose_fallback(msgs)
        assert "/workspace/report.md" in out
        # Two-sentence cap.
        assert out.count(". ") <= 2

    def test_without_artifact_asks_user(self):
        msgs = [HumanMessage(content="hi")]
        out = _compose_fallback(msgs)
        # Honest about the failure rather than inventing summary.
        assert "wasn't able" in out.lower() or "could not" in out.lower()
        # Does NOT name a specific file path that doesn't exist.
        assert "/workspace/" not in out

    def test_artifact_pulled_only_from_current_turn(self):
        # Successful write in a PRIOR turn must not surface in the
        # current turn's fallback — mirrors the loop_detection
        # cross-turn boundary fix.
        msgs = [
            HumanMessage(content="prior turn"),
            _ai("", tool_calls=[_tool_call("c1", "write_file",
                                          {"file_path": "/old.md"})]),
            _tool_msg("c1", "ok"),
            _ai("done", tool_calls=[]),  # prior turn terminated
            HumanMessage(content="new turn"),
            _ai("", tool_calls=[_tool_call("c2", "ls", {})]),
            _tool_msg("c2", "ok"),
        ]
        out = _compose_fallback(msgs)
        assert "/old.md" not in out


# ---------------------------------------------------------------------------
# Middleware-level tests (with mock handler)
# ---------------------------------------------------------------------------

class TestEmptyResponseRecoveryMiddleware:
    def test_passes_through_non_empty_response(self):
        mw = EmptyResponseRecoveryMiddleware()
        ai = _ai("Done.")
        handler = Mock(return_value=_model_response(ai))
        req = _request([HumanMessage(content="hi")])

        result = mw.wrap_model_call(req, handler)

        assert handler.call_count == 1
        # Same response returned unchanged
        assert result.result[0] is ai

    def test_passes_through_response_with_tool_calls(self):
        mw = EmptyResponseRecoveryMiddleware()
        ai = _ai("", tool_calls=[_tool_call("c1", "ls", {})])
        handler = Mock(return_value=_model_response(ai))
        req = _request([HumanMessage(content="ls")])

        result = mw.wrap_model_call(req, handler)

        assert handler.call_count == 1
        assert result.result[0] is ai

    def test_retries_once_on_empty_terminal(self):
        mw = EmptyResponseRecoveryMiddleware()
        empty = _ai("")
        good = _ai("Here is the summary.")
        handler = Mock(side_effect=[_model_response(empty), _model_response(good)])
        req = _request([HumanMessage(content="hi")])

        result = mw.wrap_model_call(req, handler)

        assert handler.call_count == 2
        assert result.result[0] is good

    def test_retry_request_includes_augmenting_human_message(self):
        mw = EmptyResponseRecoveryMiddleware()
        empty = _ai("")
        good = _ai("OK")
        handler = Mock(side_effect=[_model_response(empty), _model_response(good)])
        original_msgs = [HumanMessage(content="user prompt")]
        req = _request(original_msgs)

        mw.wrap_model_call(req, handler)

        # Second call's request must include the augmenting HumanMessage.
        retry_request = handler.call_args_list[1].args[0]
        last_msg = retry_request.messages[-1]
        assert isinstance(last_msg, HumanMessage)
        assert last_msg.content == _AUGMENT_INSTRUCTION
        # Locality contract: the original request.messages were not
        # mutated in place; the augmenter exists only on the override.
        assert len(req.messages) == len(original_msgs)
        assert req.messages[-1] is original_msgs[-1]

    def test_falls_back_when_retry_also_empty(self):
        mw = EmptyResponseRecoveryMiddleware()
        empty1 = _ai("")
        empty2 = _ai("")
        handler = Mock(side_effect=[_model_response(empty1), _model_response(empty2)])
        req = _request([HumanMessage(content="hi")])

        result = mw.wrap_model_call(req, handler)

        # Handler called once + max_retries (default 1) = 2 times.
        assert handler.call_count == 2
        # Returned AIMessage is the synthesised fallback — non-empty.
        synth = result.result[0]
        assert isinstance(synth, AIMessage)
        assert synth.content
        # And it's not just whitespace.
        assert synth.content.strip()

    def test_fallback_cites_last_successful_artifact(self):
        mw = EmptyResponseRecoveryMiddleware()
        empty = _ai("")
        handler = Mock(return_value=_model_response(empty))
        req = _request([
            HumanMessage(content="write a report"),
            _ai("", tool_calls=[_tool_call("c1", "write_file",
                                          {"file_path": "/workspace/x.md"})]),
            _tool_msg("c1", "ok"),
        ])

        result = mw.wrap_model_call(req, handler)

        synth = result.result[0]
        assert "/workspace/x.md" in synth.content

    def test_think_only_response_triggers_recovery(self):
        mw = EmptyResponseRecoveryMiddleware()
        think_only = _ai("<think>I'm done thinking</think>")
        good = _ai("Here you go.")
        handler = Mock(side_effect=[_model_response(think_only), _model_response(good)])
        req = _request([HumanMessage(content="hi")])

        result = mw.wrap_model_call(req, handler)

        assert handler.call_count == 2
        assert result.result[0] is good

    def test_logs_intervention(self, caplog):
        mw = EmptyResponseRecoveryMiddleware()
        empty = _ai("")
        good = _ai("OK")
        handler = Mock(side_effect=[_model_response(empty), _model_response(good)])
        req = _request([HumanMessage(content="hi")])

        with caplog.at_level(
            logging.WARNING, logger="assist.middleware.empty_response_recovery"
        ):
            mw.wrap_model_call(req, handler)

        assert any(
            "intervention #1" in rec.message
            for rec in caplog.records
        )

    def test_logs_fallback(self, caplog):
        mw = EmptyResponseRecoveryMiddleware()
        empty = _ai("")
        handler = Mock(return_value=_model_response(empty))
        req = _request([HumanMessage(content="hi")])

        with caplog.at_level(
            logging.ERROR, logger="assist.middleware.empty_response_recovery"
        ):
            mw.wrap_model_call(req, handler)

        assert any(
            "fallback #1" in rec.message
            for rec in caplog.records
        )

    def test_max_retries_zero_skips_retry_goes_straight_to_fallback(self):
        mw = EmptyResponseRecoveryMiddleware(max_retries=0)
        empty = _ai("")
        handler = Mock(return_value=_model_response(empty))
        req = _request([HumanMessage(content="hi")])

        result = mw.wrap_model_call(req, handler)

        assert handler.call_count == 1
        # Synthesised fallback returned.
        assert isinstance(result.result[0], AIMessage)
        assert result.result[0].content.strip()

    def test_handler_returns_aimessage_directly(self):
        # The contract permits returning an AIMessage instead of
        # ModelResponse.  Recovery must handle both shapes.
        mw = EmptyResponseRecoveryMiddleware()
        empty = _ai("")
        good = _ai("Direct AIMessage return.")
        handler = Mock(side_effect=[empty, good])
        req = _request([HumanMessage(content="hi")])

        result = mw.wrap_model_call(req, handler)

        assert handler.call_count == 2
        assert result is good

    def test_retry_success_does_not_leak_augmenter_into_result(self):
        # Locality contract: even on retry success, the returned
        # ModelResponse.result must NOT contain the augmenting
        # HumanMessage — only the model's reply messages.
        mw = EmptyResponseRecoveryMiddleware()
        empty = _ai("")
        good = _ai("OK")
        handler = Mock(side_effect=[_model_response(empty), _model_response(good)])
        req = _request([HumanMessage(content="user prompt")])

        result = mw.wrap_model_call(req, handler)

        for m in result.result:
            assert not (
                isinstance(m, HumanMessage)
                and m.content == _AUGMENT_INSTRUCTION
            ), "Augmenter must not appear in returned result"

    def test_retry_exception_propagates(self):
        # If the retry handler raises, the exception must propagate to
        # the outer middleware (BadRequestRetryMiddleware composes by
        # catching this).  Recovery does not swallow.
        mw = EmptyResponseRecoveryMiddleware()
        empty = _ai("")
        handler = Mock(side_effect=[
            _model_response(empty),
            RuntimeError("simulated upstream failure"),
        ])
        req = _request([HumanMessage(content="hi")])

        with pytest.raises(RuntimeError, match="simulated upstream failure"):
            mw.wrap_model_call(req, handler)

        assert handler.call_count == 2

    def test_fallback_preserves_structured_response(self):
        mw = EmptyResponseRecoveryMiddleware()
        empty = _ai("")
        # Original response carried a structured_response payload —
        # the synthesised fallback must propagate it.
        original = ModelResponse(result=[empty], structured_response={"k": "v"})
        handler = Mock(return_value=original)
        req = _request([HumanMessage(content="hi")])

        result = mw.wrap_model_call(req, handler)

        assert result.structured_response == {"k": "v"}
