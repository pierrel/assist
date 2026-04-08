#!/usr/bin/env python3
"""
Tests for the thread creation logic in `manage/web.py`.

This module tests the creation of threads with minimal data and the scheduling of background tasks.
"""

import pytest
from unittest.mock import MagicMock, patch
from manage.web import new_thread_with_message
from assist.thread import ThreadManager


class TestThreadCreation:
    """Test the creation of threads with minimal data and background tasks."""

    def setup_method(self):
        """Set up a fresh ThreadManager for each test."""
        self.manager = ThreadManager()

    def test_create_thread_with_minimal_data(self):
        """Test creating a thread with only the request and a placeholder description."""
        request = "Test request"
        thread_id = new_thread_with_message(request)
        
        thread = self.manager.get(thread_id)
        assert thread is not None
        assert thread_id == thread.thread_id
        assert thread.get_messages() == [{"role": "user", "content": request}]

    def test_schedule_background_tasks(self):
        """Test scheduling background tasks for a thread."""
        request = "Test request"
        
        with patch('manage.web.BackgroundTaskQueue') as mock_queue:
            mock_task_queue = MagicMock()
            mock_queue.return_value = mock_task_queue
            
            thread_id = new_thread_with_message(request)
            
            # Verify that background tasks were scheduled
            mock_task_queue.schedule_task.assert_called()
            
            # Verify that the thread has the expected messages
            thread = self.manager.get(thread_id)
            assert thread is not None
            assert thread.get_messages() == [{"role": "user", "content": request}]

    def test_thread_creation_with_invalid_input(self):
        """Test creating a thread with invalid input."""
        with pytest.raises(ValueError) as excinfo:
            new_thread_with_message(None)
        assert "Request cannot be empty" in str(excinfo.value)

    def test_thread_creation_with_empty_request(self):
        """Test creating a thread with an empty request."""
        with pytest.raises(ValueError) as excinfo:
            new_thread_with_message("")
        assert "Request cannot be empty" in str(excinfo.value)

    def test_thread_creation_with_placeholder_description(self):
        """Test creating a thread with a placeholder description."""
        request = "Test request"
        thread_id = new_thread_with_message(request)
        
        thread = self.manager.get(thread_id)
        assert thread is not None
        assert thread.get_messages() == [{"role": "user", "content": request}]
        
        # Verify that a placeholder description is added
        assert any(
            msg.get('content', '').startswith('Processing request:') 
            for msg in thread.get_messages()
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])