#!/usr/bin/env python3
"""
Test cases for ensuring the web view remains responsive during background task failures.
This includes:
- UI updates during failures
- Progress indicators
- Error handling
"""

import pytest
from unittest.mock import patch, MagicMock
from manage.background_tasks import BackgroundTaskQueue
from manage.progress_tracker import ProgressTracker
from manage.web import TASK_QUEUE, PROGRESS_TRACKER, render_thread


class TestWebViewResponsiveness:
    """Test cases for ensuring the web view remains responsive during background task failures."""

    def test_ui_updates_during_failure(self, client):
        """Test that the web view updates to show error messages during background task failures."""
        # Simulate a background task failure
        task_id = "test_task_id"
        
        # Mock progress tracking to simulate a failed task
        with patch.object(PROGRESS_TRACKER, "get_task_status", return_value="failed"):
            with patch.object(PROGRESS_TRACKER, "get_task_error", return_value="Task failed"):
                with patch.object(PROGRESS_TRACKER, "get_progress", return_value=1.0):
                    # Trigger a web request to render the thread
                    response = client.get(f"/thread/{task_id}")
                    
                    # Verify UI responsiveness
                    assert response.status_code == 200
                    assert "error-msg" in response.text
                    assert "Task failed" in response.text
                    
    def test_progress_indicators(self, client):
        """Test that progress indicators update dynamically during background tasks."""
        # Simulate a background task with partial progress
        task_id = "test_task_id"
        
        # Mock progress tracking to simulate a task in progress
        with patch.object(PROGRESS_TRACKER, "get_task_status", return_value="in_progress"):
            with patch.object(PROGRESS_TRACKER, "get_progress", return_value=0.5):
                # Trigger a web request to render the thread
                response = client.get(f"/thread/{task_id}")
                
                # Verify progress indicators are present
                assert response.status_code == 200
                assert "progress-bar" in response.text or "progress-indicator" in response.text
                
    def test_ui_responsiveness_during_failure(self, client):
        """Test that the web view remains responsive during background task failures."""
        # Simulate a background task failure
        task_id = "test_task_id"
        
        # Mock progress tracking to simulate a failed task
        with patch.object(PROGRESS_TRACKER, "get_task_status", return_value="failed"):
            with patch.object(PROGRESS_TRACKER, "get_task_error", return_value="Task failed"):
                with patch.object(PROGRESS_TRACKER, "get_progress", return_value=1.0):
                    # Trigger a web request to render the thread
                    response = client.get(f"/thread/{task_id}")
                    
                    # Verify UI responsiveness
                    assert response.status_code == 200
                    assert "error-msg" in response.text
                    assert "Task failed" in response.text
                    
    def test_successful_task_ui_update(self, client):
        """Test that the web view updates correctly for a successfully completed task."""
        # Simulate a background task that completed successfully
        task_id = "test_task_id"
        
        # Mock progress tracking to simulate a completed task
        with patch.object(PROGRESS_TRACKER, "get_task_status", return_value="completed"):
            with patch.object(PROGRESS_TRACKER, "get_task_result", return_value="Task completed successfully"):
                with patch.object(PROGRESS_TRACKER, "get_progress", return_value=1.0):
                    # Trigger a web request to render the thread
                    response = client.get(f"/thread/{task_id}")
                    
                    # Verify UI responsiveness
                    assert response.status_code == 200
                    assert "success-msg" in response.text
                    assert "Task completed successfully" in response.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])