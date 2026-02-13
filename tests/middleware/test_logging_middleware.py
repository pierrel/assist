"""Tests for ModelLoggingMiddleware."""
import pytest
from unittest.mock import Mock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from assist.middleware.model_logging_middleware import ModelLoggingMiddleware


class TestModelLoggingMiddleware:
    """Test suite for model logging middleware."""

    def test_count_tool_calls(self):
        """Test counting tool calls in a message."""
        middleware = ModelLoggingMiddleware()

        # Message with no tool calls
        msg = AIMessage(content="Hello")
        assert middleware._count_tool_calls(msg) == 0

        # Message with tool calls
        msg = AIMessage(content="")
        msg.tool_calls = [
            {'id': 'call_1', 'name': 'search', 'args': {'query': 'test'}},
            {'id': 'call_2', 'name': 'read_file', 'args': {'path': 'test.txt'}},
            {'id': 'call_3', 'name': 'write_file', 'args': {'path': 'out.txt'}}
        ]
        assert middleware._count_tool_calls(msg) == 3

    def test_count_tool_results(self):
        """Test counting tool result messages."""
        middleware = ModelLoggingMiddleware()

        # No tool results
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi")
        ]
        assert middleware._count_tool_results(messages) == 0

        # With tool results
        messages = [
            HumanMessage(content="Search for X"),
            AIMessage(content="", tool_calls=[{'id': 'call_1', 'name': 'search', 'args': {}}]),
            ToolMessage(content="Result 1", tool_call_id="call_1"),
            ToolMessage(content="Result 2", tool_call_id="call_2"),
            ToolMessage(content="Result 3", tool_call_id="call_3")
        ]
        assert middleware._count_tool_results(messages) == 3

    def test_get_tool_result_info(self):
        """Test getting tool result information."""
        middleware = ModelLoggingMiddleware()

        # No tool results
        messages = [HumanMessage(content="Hello")]
        info = middleware._get_tool_result_info(messages)
        assert info["count"] == 0
        assert info["approx_tokens"] == 0

        # With tool results
        messages = [
            ToolMessage(content="Short result", tool_call_id="call_1"),
            ToolMessage(content="A" * 1000, tool_call_id="call_2")
        ]
        info = middleware._get_tool_result_info(messages)
        assert info["count"] == 2
        assert info["approx_tokens"] > 0  # Should have some tokens

    def test_before_model_logs_tool_results(self):
        """Test that before_model logs tool result information."""
        middleware = ModelLoggingMiddleware(agent_name="test-agent")

        # State with tool results
        messages = [
            HumanMessage(content="Search"),
            AIMessage(content="", tool_calls=[{'id': 'call_1', 'name': 'search', 'args': {}}]),
            ToolMessage(content="Search result data", tool_call_id="call_1"),
            ToolMessage(content="More results", tool_call_id="call_2")
        ]
        state = {"messages": messages}
        runtime = Mock()

        # Should not raise and should log appropriately
        result = middleware.before_model(state, runtime)
        assert result is None  # before_model doesn't modify state

        # Check that counter incremented
        assert middleware._model_call_count == 1

    def test_after_model_tracks_concurrent_calls(self):
        """Test that after_model tracks concurrent tool calls."""
        middleware = ModelLoggingMiddleware(agent_name="test-agent")

        # Simulate model response with concurrent tool calls
        msg = AIMessage(content="")
        msg.tool_calls = [
            {'id': 'call_1', 'name': 'search', 'args': {'query': 'A'}},
            {'id': 'call_2', 'name': 'search', 'args': {'query': 'B'}},
            {'id': 'call_3', 'name': 'read_file', 'args': {'path': 'test.txt'}}
        ]

        state = {"messages": [msg]}
        runtime = Mock()

        middleware._model_call_count = 1  # Simulate before_model was called
        result = middleware.after_model(state, runtime)

        # Check statistics
        assert middleware._total_tool_calls == 3
        assert middleware._max_concurrent_calls == 3
        assert middleware._concurrent_calls_distribution[3] == 1

    def test_concurrent_call_distribution(self):
        """Test that concurrent call distribution is tracked correctly."""
        middleware = ModelLoggingMiddleware(agent_name="test-agent")

        # Process multiple responses with different concurrent call counts
        test_cases = [
            1,  # 1 concurrent call
            3,  # 3 concurrent calls
            2,  # 2 concurrent calls
            3,  # 3 concurrent calls (again)
            1,  # 1 concurrent call (again)
        ]

        for i, call_count in enumerate(test_cases):
            msg = AIMessage(content="")
            msg.tool_calls = [
                {'id': f'call_{j}', 'name': 'tool', 'args': {}}
                for j in range(call_count)
            ]

            state = {"messages": [msg]}
            runtime = Mock()

            middleware._model_call_count = i + 1
            middleware.after_model(state, runtime)

        # Check distribution
        assert middleware._concurrent_calls_distribution[1] == 2
        assert middleware._concurrent_calls_distribution[2] == 1
        assert middleware._concurrent_calls_distribution[3] == 2
        assert middleware._max_concurrent_calls == 3
        assert middleware._total_tool_calls == sum(test_cases)

    def test_format_tool_message(self):
        """Test formatting of tool result messages."""
        middleware = ModelLoggingMiddleware()

        # Tool message
        msg = ToolMessage(content="Search results", tool_call_id="call_1")
        msg.name = "search"
        formatted = middleware._format_message(msg)

        assert "[TOOL RESULT: search]" in formatted
        assert "Search results" in formatted

    def test_format_tool_message_large_content(self):
        """Test formatting of tool messages with large content."""
        middleware = ModelLoggingMiddleware()

        # Large tool result
        large_content = "A" * 500
        msg = ToolMessage(content=large_content, tool_call_id="call_1")
        msg.name = "read_file"
        formatted = middleware._format_message(msg)

        assert "[TOOL RESULT: read_file]" in formatted
        assert "tokens" in formatted
        # Should be truncated in preview
        assert len(formatted) < len(large_content)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
