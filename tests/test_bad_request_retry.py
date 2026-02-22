"""Tests for BadRequestRetryMiddleware.

Deterministic unit tests using mocked model calls — no running LLM required.
"""
import httpx
import pytest
from unittest.mock import Mock, MagicMock, patch
from openai import BadRequestError

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain.agents.middleware.types import ModelRequest, ModelResponse

from assist.middleware.bad_request_retry import BadRequestRetryMiddleware


def _make_bad_request_error(msg="Expecting ':' delimiter"):
    """Create a realistic BadRequestError like vLLM returns."""
    req = httpx.Request("POST", "http://localhost/v1/chat/completions")
    resp = httpx.Response(400, json={"error": {"message": msg}}, request=req)
    return BadRequestError(msg, response=resp, body={"error": {"message": msg}})


def _make_model_request(messages=None):
    """Create a minimal ModelRequest for testing."""
    if messages is None:
        messages = [HumanMessage(content="hello")]
    model = Mock()
    return ModelRequest(
        model=model,
        messages=messages,
    )


def _make_model_response(content="ok"):
    """Create a minimal successful ModelResponse."""
    return ModelResponse(result=[AIMessage(content=content)])


class TestBadRequestRetryMiddleware:
    """Tests for the BadRequestRetryMiddleware."""

    def test_passes_through_on_success(self):
        """No error — handler result is returned directly."""
        mw = BadRequestRetryMiddleware(max_retries=3)
        request = _make_model_request()
        expected = _make_model_response("all good")

        handler = Mock(return_value=expected)
        result = mw.wrap_model_call(request, handler)

        assert result == expected
        assert handler.call_count == 1

    def test_retries_on_bad_request_then_succeeds(self):
        """First call fails, second succeeds after sanitization."""
        mw = BadRequestRetryMiddleware(max_retries=3)
        request = _make_model_request()

        handler = Mock(side_effect=[
            _make_bad_request_error(),
            _make_model_response("recovered"),
        ])

        result = mw.wrap_model_call(request, handler)

        assert isinstance(result, ModelResponse)
        assert result.result[0].content == "recovered"
        assert handler.call_count == 2

    def test_retries_up_to_max_then_returns_error_message(self):
        """All retries fail — returns AIMessage with error instead of raising."""
        mw = BadRequestRetryMiddleware(max_retries=2)
        request = _make_model_request()

        handler = Mock(side_effect=_make_bad_request_error())

        result = mw.wrap_model_call(request, handler)

        # Should return AIMessage, not raise
        assert isinstance(result, AIMessage)
        assert "rejected the request" in result.content
        assert "3 attempts" in result.content  # 1 initial + 2 retries
        # 1 initial + 2 retries = 3 calls
        assert handler.call_count == 3

    def test_non_bad_request_errors_propagate(self):
        """Other exception types are NOT caught — they propagate immediately."""
        mw = BadRequestRetryMiddleware(max_retries=3)
        request = _make_model_request()

        handler = Mock(side_effect=RuntimeError("something else"))

        with pytest.raises(RuntimeError, match="something else"):
            mw.wrap_model_call(request, handler)

        assert handler.call_count == 1

    def test_sanitizes_control_chars_in_messages(self):
        """Messages with control characters should be sanitized on retry."""
        mw = BadRequestRetryMiddleware(max_retries=3)

        # Message with null bytes and control chars
        bad_msg = HumanMessage(content="hello\x00world\x01\x02")
        request = _make_model_request([bad_msg])

        # First call fails, second succeeds
        handler = Mock(side_effect=[
            _make_bad_request_error(),
            _make_model_response("ok"),
        ])

        result = mw.wrap_model_call(request, handler)

        assert isinstance(result, ModelResponse)
        # The second call should have received sanitized messages
        second_call_request = handler.call_args_list[1][0][0]
        sanitized_content = second_call_request.messages[0].content
        assert "\x00" not in sanitized_content
        assert "\x01" not in sanitized_content
        assert "helloworld" in sanitized_content

    def test_sanitizes_tool_call_args(self):
        """Tool call arguments with bad escapes should be sanitized."""
        mw = BadRequestRetryMiddleware(max_retries=3)

        ai_msg = AIMessage(content="", tool_calls=[{
            "id": "call_123",
            "name": "write_file",
            "args": {"content": "line1\\\nline2"},  # trailing backslash
        }])
        request = _make_model_request([ai_msg])

        handler = Mock(side_effect=[
            _make_bad_request_error(),
            _make_model_response("ok"),
        ])

        result = mw.wrap_model_call(request, handler)

        assert isinstance(result, ModelResponse)
        assert handler.call_count == 2

    def test_aggressive_truncation_on_later_attempts(self):
        """On attempt 2+, large content should be truncated."""
        mw = BadRequestRetryMiddleware(max_retries=3)

        big_content = "x" * 50_000
        big_msg = ToolMessage(
            content=big_content,
            tool_call_id="call_123",
        )
        request = _make_model_request([
            HumanMessage(content="hi"),
            big_msg,
        ])

        # Fail twice (to trigger aggressive mode), succeed on third
        handler = Mock(side_effect=[
            _make_bad_request_error(),
            _make_bad_request_error(),
            _make_model_response("ok"),
        ])

        result = mw.wrap_model_call(request, handler)

        assert isinstance(result, ModelResponse)
        # Third call should have truncated content
        third_request = handler.call_args_list[2][0][0]
        tool_msg = third_request.messages[1]
        assert len(str(tool_msg.content)) < 50_000
        assert "[Content truncated" in str(tool_msg.content)

    def test_tracks_retry_count(self):
        """Internal retry counter should increment."""
        mw = BadRequestRetryMiddleware(max_retries=2)
        request = _make_model_request()

        handler = Mock(side_effect=[
            _make_bad_request_error(),
            _make_model_response("ok"),
        ])

        assert mw._retry_count == 0
        mw.wrap_model_call(request, handler)
        assert mw._retry_count == 1

    def test_max_retries_zero_means_no_retries(self):
        """With max_retries=0, should fail immediately with AIMessage."""
        mw = BadRequestRetryMiddleware(max_retries=0)
        request = _make_model_request()

        handler = Mock(side_effect=_make_bad_request_error())

        result = mw.wrap_model_call(request, handler)

        assert isinstance(result, AIMessage)
        assert "rejected the request" in result.content
        assert handler.call_count == 1


class TestSanitizationHelpers:
    """Tests for the sanitization utility methods."""

    def test_strip_control_chars_preserves_whitespace(self):
        mw = BadRequestRetryMiddleware()
        result = mw._strip_control_chars("hello\nworld\ttab\rreturn")
        assert result == "hello\nworld\ttab\rreturn"

    def test_strip_control_chars_removes_nulls(self):
        mw = BadRequestRetryMiddleware()
        result = mw._strip_control_chars("hel\x00lo\x01wo\x7frld")
        assert result == "helloworld"

    def test_fix_json_escapes(self):
        mw = BadRequestRetryMiddleware()
        # Invalid: \  (backslash space) — should become \\
        result = mw._fix_json_escapes('hello\\ world')
        assert result == 'hello\\\\ world'

    def test_fix_json_escapes_preserves_valid(self):
        mw = BadRequestRetryMiddleware()
        # Valid escapes should not be modified
        result = mw._fix_json_escapes('hello\\nworld\\t\\r\\\\"')
        assert result == 'hello\\nworld\\t\\r\\\\"'

    def test_sanitize_list_content(self):
        mw = BadRequestRetryMiddleware()
        content = [
            {"type": "text", "text": "hello\x00world"},
            {"type": "image_url", "url": "http://example.com"},
        ]
        result = mw._sanitize_message_content(content)
        assert result[0]["text"] == "helloworld"
        assert result[1] == content[1]  # non-text parts unchanged

    def test_truncate_large_content(self):
        mw = BadRequestRetryMiddleware()
        big = "x" * 30_000
        result = mw._truncate_large_content(big, max_chars=20_000)
        assert len(result) < 30_000
        assert "[Content truncated" in result
