#!/usr/bin/env python3
"""
Tests for the thread rendering logic in `manage/web.py`.

This module tests the rendering of threads with minimal data initially,
and dynamically updating the web view as background tasks complete.
"""

import pytest
from unittest.mock import MagicMock, patch
from manage.web import render_thread
from assist.thread import ThreadManager


class TestThreadRendering:
    """Test the rendering of threads with minimal data and dynamic updates."""

    def setup_method(self):
        """Set up a fresh ThreadManager for each test."""
        self.manager = ThreadManager()

    def test_render_thread_with_minimal_data(self):
        """Test rendering a thread with only the request initially."""
        request = "Test request"
        thread_id = "test_thread_id"
        
        # Create a thread with minimal data
        thread = self.manager.new()
        thread.message({"role": "user", "content": request})
        
        # Render the thread
        rendered = render_thread(thread_id)
        
        # Verify that only the request is rendered initially
        assert request in rendered
        assert "Processing request:" not in rendered

    def test_render_thread_with_task_progress(self):
        """Test rendering task progress for background tasks."""
        request = "Test request"
        thread_id = "test_thread_id"
        
        # Create a thread with minimal data
        thread = self.manager.new()
        thread.message({"role": "user", "content": request})
        
        # Mock the BackgroundTaskQueue to simulate task progress
        with patch('manage.web.BackgroundTaskQueue') as mock_queue:
            mock_task_queue = MagicMock()
            mock_queue.return_value = mock_task_queue
            
            # Simulate task progress
            mock_task_queue.get_task_status.return_value = "in_progress"
            mock_task_queue.get_progress.return_value = 0.5
            
            # Render the thread
            rendered = render_thread(thread_id)
            
            # Verify that task progress is rendered
            assert "Processing request:" in rendered
            assert "50%" in rendered

    def test_render_thread_with_completed_tasks(self):
        """Test rendering a thread after background tasks complete."""
        request = "Test request"
        thread_id = "test_thread_id"
        
        # Create a thread with minimal data
        thread = self.manager.new()
        thread.message({"role": "user", "content": request})
        
        # Mock the BackgroundTaskQueue to simulate completed tasks
        with patch('manage.web.BackgroundTaskQueue') as mock_queue:
            mock_task_queue = MagicMock()
            mock_queue.return_value = mock_queue
            
            # Simulate completed task
            mock_queue.get_task_status.return_value = "completed"
            mock_queue.get_task_result.return_value = "Description generated"
            
            # Render the thread
            rendered = render_thread(thread_id)
            
            # Verify that completed task results are rendered
            assert request in rendered
            assert "Description generated" in rendered

    def test_render_thread_with_failed_tasks(self):
        """Test rendering a thread after background tasks fail."""
        request = "Test request"
        thread_id = "test_thread_id"
        
        # Create a thread with minimal data
        thread = self.manager.new()
        thread.message({"role": "user", "content": request})
        
        # Mock the BackgroundTaskQueue to simulate failed tasks
        with patch('manage.web.BackgroundTaskQueue') as mock_queue:
            mock_task_queue = MagicMock()
            mock_queue.return_value = mock_queue
            
            # Simulate failed task
            mock_queue.get_task_status.return_value = "failed"
            mock_queue.get_task_error.return_value = "Task failed: Description generation error"
            
            # Render the thread
            rendered = render_thread(thread_id)
            
            # Verify that failed task errors are rendered
            assert request in rendered
            assert "Task failed: Description generation error" in rendered

    def test_render_thread_with_no_background_tasks(self):
        """Test rendering a thread with no background tasks."""
        request = "Test request"
        thread_id = "test_thread_id"
        
        # Create a thread with minimal data
        thread = self.manager.new()
        thread.message({"role": "user", "content": request})
        
        # Render the thread
        rendered = render_thread(thread_id)
        
        # Verify that only the request is rendered
        assert request in rendered
        assert "Processing request:" not in rendered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])