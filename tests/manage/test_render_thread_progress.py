#!/usr/bin/env python3
"""
Tests for rendering thread progress using ProgressTracker.
This file tests the integration of ProgressTracker with render_thread.
"""

import pytest
from manage.progress_tracker import ProgressTracker
from manage.web import render_thread


class TestRenderThreadProgress:
    """Tests for rendering thread progress using ProgressTracker."""

    def setup_method(self):
        """Setup test fixtures."""
        self.progress_tracker = ProgressTracker()
        
    def test_render_thread_with_task_progress(self, mocker):
        """Test that render_thread fetches task progress from ProgressTracker."""
        # Mock the progress_tracker.get_task_status method
        mock_get_status = mocker.patch.object(
            self.progress_tracker, 'get_task_status',
            return_value="in_progress"
        )
        
        # Mock the progress_tracker.get_progress method
        mock_get_progress = mocker.patch.object(
            self.progress_tracker, 'get_progress',
            return_value=50
        )
        
        # Mock the FastAPI request and response objects
        mock_request = mocker.MagicMock()
        mock_response = mocker.MagicMock()
        
        # Mock the thread data
        mock_thread_data = {
            "tid": "test_tid",
            "messages": [{"text": "test_text"}],
            "tasks": [{"task_id": "test_task_id", "type": "git_clone"}]
        }
        
        # Call the function under test
        html = render_thread(
            request=mock_request,
            tid="test_tid",
            progress_tracker=self.progress_tracker,
            thread_data=mock_thread_data
        )
        
        # Assert that progress_tracker.get_task_status was called
        mock_get_status.assert_called_once()
        args, kwargs = mock_get_status.call_args
        assert args[0] == "test_task_id"
        
        # Assert that progress_tracker.get_progress was called
        mock_get_progress.assert_called_once()
        args, kwargs = mock_get_progress.call_args
        assert args[0] == "test_task_id"
        
        # Assert that the rendered HTML contains progress information
        assert "50%" in html or "in_progress" in html
        
    def test_render_thread_with_completed_tasks(self, mocker):
        """Test that render_thread renders completed task results."""
        # Mock the progress_tracker.get_task_status method
        mock_get_status = mocker.patch.object(
            self.progress_tracker, 'get_task_status',
            return_value="completed"
        )
        
        # Mock the progress_tracker.get_progress method
        mock_get_progress = mocker.patch.object(
            self.progress_tracker, 'get_progress',
            return_value=100
        )
        
        # Mock the FastAPI request and response objects
        mock_request = mocker.MagicMock()
        mock_response = mocker.MagicMock()
        
        # Mock the thread data with completed task results
        mock_thread_data = {
            "tid": "test_tid",
            "messages": [{"text": "test_text"}],
            "tasks": [{"task_id": "test_task_id", "type": "git_clone", "result": "success"}]
        }
        
        # Call the function under test
        html = render_thread(
            request=mock_request,
            tid="test_tid",
            progress_tracker=self.progress_tracker,
            thread_data=mock_thread_data
        )
        
        # Assert that progress_tracker.get_task_status was called
        mock_get_status.assert_called_once()
        args, kwargs = mock_get_status.call_args
        assert args[0] == "test_task_id"
        
        # Assert that progress_tracker.get_progress was called
        mock_get_progress.assert_called_once()
        args, kwargs = mock_get_progress.call_args
        assert args[0] == "test_task_id"
        
        # Assert that the rendered HTML contains completion information
        assert "completed" in html or "success" in html
        
    def test_render_thread_with_failed_tasks(self, mocker):
        """Test that render_thread renders failed task errors."""
        # Mock the progress_tracker.get_task_status method
        mock_get_status = mocker.patch.object(
            self.progress_tracker, 'get_task_status',
            return_value="failed"
        )
        
        # Mock the progress_tracker.get_progress method
        mock_get_progress = mocker.patch.object(
            self.progress_tracker, 'get_progress',
            return_value=0
        )
        
        # Mock the FastAPI request and response objects
        mock_request = mocker.MagicMock()
        mock_response = mocker.MagicMock()
        
        # Mock the thread data with failed task errors
        mock_thread_data = {
            "tid": "test_tid",
            "messages": [{"text": "test_text"}],
            "tasks": [{"task_id": "test_task_id", "type": "git_clone", "error": "Git not installed"}]
        }
        
        # Call the function under test
        html = render_thread(
            request=mock_request,
            tid="test_tid",
            progress_tracker=self.progress_tracker,
            thread_data=mock_thread_data
        )
        
        # Assert that progress_tracker.get_task_status was called
        mock_get_status.assert_called_once()
        args, kwargs = mock_get_status.call_args
        assert args[0] == "test_task_id"
        
        # Assert that progress_tracker.get_progress was called
        mock_get_progress.assert_called_once()
        args, kwargs = mock_get_progress.call_args
        assert args[0] == "test_task_id"
        
        # Assert that the rendered HTML contains error information
        assert "failed" in html or "Git not installed" in html


if __name__ == "__main__":
    pytest.main([__file__, "-v"])