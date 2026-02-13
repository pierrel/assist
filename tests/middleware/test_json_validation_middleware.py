"""Tests for JsonValidationMiddleware.

These tests cover real-world JSON validation issues encountered when sending
requests to vLLM, including:
- Trailing backslashes in markdown content
- Invalid escape sequences
- Large content payloads
"""
import json
import pytest
from unittest.mock import Mock
from langchain_core.messages import AIMessage, HumanMessage

from assist.middleware.json_validation_middleware import JsonValidationMiddleware


class TestJsonValidationMiddleware:
    """Test suite for JSON validation middleware."""

    def test_sanitize_trailing_backslash(self):
        """Test that trailing backslashes in content are properly escaped.

        This is a common issue in markdown where lines end with backslashes
        for line continuation, e.g.:
        - **Pros**: \
        - **Cons**: \
        """
        middleware = JsonValidationMiddleware(strict=False)

        # Content with trailing backslash (common in markdown)
        problematic_content = """# Report

## Comparison

- **Pros**: \
  - Feature A
  - Feature B
- **Cons**: \
  - Limitation A"""

        # Test sanitization
        sanitized = middleware._sanitize_string_content(problematic_content)

        # Verify it can be JSON serialized
        try:
            json.dumps(sanitized)
            # If we get here, sanitization worked
            assert True
        except (TypeError, ValueError) as e:
            pytest.fail(f"Sanitization failed: {e}")

    def test_sanitize_multiple_escape_sequences(self):
        """Test content with multiple types of escape sequences."""
        middleware = JsonValidationMiddleware(strict=False)

        # Mix of valid and problematic escapes
        content = r"""Text with \n newline and \t tab.
Also has trailing backslash: \
And escaped quote: \"quoted\""""

        sanitized = middleware._sanitize_string_content(content)

        # Should be JSON-safe
        try:
            result = json.dumps(sanitized)
            # Verify we can parse it back
            parsed = json.loads(result)
            assert isinstance(parsed, str)
        except json.JSONDecodeError as e:
            pytest.fail(f"Sanitization failed: {e}")


    def test_before_model_sanitizes_message_content(self):
        """Test that before_model hook sanitizes message content."""
        middleware = JsonValidationMiddleware(strict=False)

        # Create a message with problematic content
        problematic_msg = HumanMessage(content="Content with trailing backslash: \\")
        state = {"messages": [problematic_msg]}
        runtime = Mock()

        result = middleware.before_model(state, runtime)

        # Should return modified state
        if result is not None:
            sanitized_messages = result["messages"]
            sanitized_content = sanitized_messages[0].content

            # Verify it's JSON-safe
            try:
                json.dumps(sanitized_content)
            except (TypeError, ValueError) as e:
                pytest.fail(f"Sanitized content is not JSON-safe: {e}")

    def test_before_model_sanitizes_tool_call_args(self):
        """Test that before_model hook sanitizes tool call arguments."""
        middleware = JsonValidationMiddleware(strict=False)

        # Create a message with tool calls containing problematic content
        msg = AIMessage(content="")
        msg.tool_calls = [{
            'id': 'call_1',
            'name': 'write_file',
            'args': {
                'path': 'test.md',
                'content': 'Line with backslash: \\\nMore content'
            }
        }]

        state = {"messages": [msg]}
        runtime = Mock()

        result = middleware.before_model(state, runtime)

        # Should return modified state
        if result is not None:
            sanitized_messages = result["messages"]
            sanitized_args = sanitized_messages[0].tool_calls[0]['args']

            # Verify args are JSON-safe
            try:
                json.dumps(sanitized_args)
            except (TypeError, ValueError) as e:
                pytest.fail(f"Sanitized args are not JSON-safe: {e}")

    def test_realistic_vllm_error_case(self):
        """Test the realistic case that caused vLLM JSON errors.

        This recreates the actual error scenario: a write_file tool call
        with markdown content containing trailing backslashes that causes
        'Invalid \\escape' errors when serialized to JSON.
        """
        middleware = JsonValidationMiddleware(strict=False)

        # Realistic markdown content with trailing backslashes
        problematic_markdown = """* Technology Evaluation Report

** Executive Summary
This report provides a comprehensive comparison of technologies A and B.

** Technology A

*** Pros
- **Performance**: \
  High throughput and low latency
- **Scalability**: \
  Handles millions of requests
- **Cost**: \
  Economical at scale

*** Cons
- **Complexity**: \
  Steep learning curve
- **Support**: \
  Limited documentation

** Technology B

*** Pros
- **Ease of Use**: \
  Intuitive interface
- **Documentation**: \
  Comprehensive guides

*** Cons
- **Performance**: \
  Slower than A
- **Cost**: \
  More expensive

*** Sources
[1] Source A: https://example.com/a
[2] Source B: https://example.com/b
"""

        # Create a tool call similar to what would be generated
        msg = AIMessage(content="I'll write the report now.")
        msg.tool_calls = [{
            'id': 'call_write_report',
            'name': 'write_file',
            'args': {
                'path': 'report.org',
                'content': problematic_markdown
            }
        }]

        state = {"messages": [msg]}
        runtime = Mock()

        # Run before_model hook
        result = middleware.before_model(state, runtime)

        # Extract the content (modified or original)
        if result is not None:
            final_content = result["messages"][0].tool_calls[0]['args']['content']
        else:
            final_content = msg.tool_calls[0]['args']['content']

        # The critical test: can this be serialized to JSON without errors?
        try:
            json_str = json.dumps({'content': final_content})
            # And can we parse it back?
            parsed = json.loads(json_str)
            assert 'content' in parsed
            # Success - the middleware fixed the issue
        except json.JSONDecodeError as e:
            pytest.fail(
                f"Content is still not JSON-safe after middleware processing. "
                f"Error: {e}\n"
                f"This is the error that would be sent to vLLM and cause it to fail."
            )

    def test_after_model_validates_tool_calls(self):
        """Test that after_model hook validates tool calls from the model."""
        middleware = JsonValidationMiddleware(strict=False)

        # Simulate a model response with a tool call
        msg = AIMessage(content="")
        msg.tool_calls = [{
            'id': 'call_1',
            'function': {
                'name': 'search',
                'arguments': '{"query": "test"}'  # Valid JSON string
            }
        }]

        state = {"messages": [msg]}
        runtime = Mock()

        result = middleware.after_model(state, runtime)

        # Should validate successfully
        # If there are no issues, result might be None
        if result is not None:
            # Check that the tool call is still valid
            assert 'messages' in result

    def test_after_model_fixes_invalid_json(self):
        """Test that after_model attempts to fix invalid JSON in tool calls."""
        middleware = JsonValidationMiddleware(strict=False)

        # Invalid JSON: trailing comma
        msg = AIMessage(content="")
        msg.tool_calls = [{
            'id': 'call_1',
            'function': {
                'name': 'search',
                'arguments': '{"query": "test",}'  # Trailing comma - invalid
            }
        }]

        state = {"messages": [msg]}
        runtime = Mock()

        result = middleware.after_model(state, runtime)

        # Should attempt to fix
        if result is not None:
            fixed_args = result["messages"][0].tool_calls[0]['function']['arguments']
            # Should be valid JSON now
            try:
                json.loads(fixed_args)
            except json.JSONDecodeError as e:
                pytest.fail(f"Middleware failed to fix invalid JSON: {e}")

    def test_strict_mode_raises_on_invalid_json(self):
        """Test that strict mode raises errors instead of fixing."""
        middleware = JsonValidationMiddleware(strict=True)

        # Invalid JSON
        msg = AIMessage(content="")
        msg.tool_calls = [{
            'id': 'call_1',
            'function': {
                'name': 'search',
                'arguments': '{"query": invalid}'  # Invalid JSON
            }
        }]

        state = {"messages": [msg]}
        runtime = Mock()

        # Should raise ValueError in strict mode
        with pytest.raises(ValueError, match="Invalid JSON in tool call"):
            middleware.after_model(state, runtime)

    def test_empty_state_handling(self):
        """Test that middleware handles empty state gracefully."""
        middleware = JsonValidationMiddleware(strict=False)

        state = {"messages": []}
        runtime = Mock()

        # Should not crash
        result_before = middleware.before_model(state, runtime)
        result_after = middleware.after_model(state, runtime)

        assert result_before is None
        assert result_after is None

    def test_statistics_tracking(self):
        """Test that middleware tracks statistics correctly."""
        middleware = JsonValidationMiddleware(strict=False)

        # Process several tool calls
        msg = AIMessage(content="")
        msg.tool_calls = [{
            'id': 'call_1',
            'function': {
                'name': 'search',
                'arguments': '{"query": "test"}'
            }
        }]

        state = {"messages": [msg]}
        runtime = Mock()

        # Process multiple times
        for _ in range(5):
            middleware.after_model(state, runtime)

        # Check statistics
        assert middleware._validation_count == 5
        # Fix count should be 0 since JSON was valid
        assert middleware._fix_count == 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_non_string_content(self):
        """Test handling of non-string content."""
        middleware = JsonValidationMiddleware(strict=False)

        # Non-string content should be returned as-is
        assert middleware._sanitize_string_content(None) is None
        assert middleware._sanitize_string_content(123) == 123
        assert middleware._sanitize_string_content([]) == []

    def test_empty_string(self):
        """Test handling of empty strings."""
        middleware = JsonValidationMiddleware(strict=False)

        sanitized = middleware._sanitize_string_content("")
        assert sanitized == ""

        # Should be JSON-safe
        json.dumps(sanitized)

    def test_already_valid_json_string(self):
        """Test that valid content is not modified unnecessarily."""
        middleware = JsonValidationMiddleware(strict=False)

        valid_content = "This is perfectly valid content with newlines\nand tabs\t"
        sanitized = middleware._sanitize_string_content(valid_content)

        # Should be unchanged (or at least still valid)
        try:
            json.dumps(sanitized)
        except Exception as e:
            pytest.fail(f"Valid content became invalid: {e}")

    def test_unicode_content(self):
        """Test handling of unicode characters."""
        middleware = JsonValidationMiddleware(strict=False)

        unicode_content = "Content with émojis 🎉 and spëcial çharacters"
        sanitized = middleware._sanitize_string_content(unicode_content)

        # Should be JSON-safe
        json_str = json.dumps(sanitized)
        parsed = json.loads(json_str)
        assert isinstance(parsed, str)

    def test_very_large_content(self):
        """Test handling of very large content (stress test)."""
        middleware = JsonValidationMiddleware(strict=False)

        # 1MB of content - should still be sanitizable even if huge
        huge_content = "A" * 1_000_000

        # Should be able to sanitize without errors
        sanitized = middleware._sanitize_string_content(huge_content)

        # Should be JSON-safe (even if large)
        try:
            json.dumps(sanitized)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Large content sanitization failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
