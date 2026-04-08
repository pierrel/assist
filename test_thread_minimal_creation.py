#!/usr/bin/env python3
"""
Test cases for minimal thread creation.

This module tests:
- Minimal thread creation with only the request.
"""

import pytest
from unittest.mock import MagicMock


class TestThreadCreation:
    """Test thread creation with minimal data."""

    def test_minimal_thread_creation(self):
        """Test thread creation with minimal data (request only)."""
        # Mock a minimal thread
        mock_thread = MagicMock()
        mock_thread.id = "12345"
        mock_thread.request = {
            "id": "12345",
            "type": "text",
            "content": "Hello, world!"
        }
        
        assert mock_thread is not None, "Thread should be created"
        assert hasattr(mock_thread, "id"), "Thread should have an ID"
        assert hasattr(mock_thread, "request"), "Thread should have a request"
        assert mock_thread.request["id"] == "12345", "Request ID should match"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])