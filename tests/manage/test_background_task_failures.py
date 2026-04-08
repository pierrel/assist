#!/usr/bin/env python3
"""
Test cases for simulating background task failures.
This includes:
- Git clone failures
- Docker setup failures
- General task failures
"""

import pytest
from unittest.mock import patch, MagicMock
from manage.background_tasks import BackgroundTaskQueue
from manage.progress_tracker import ProgressTracker
from manage.web import TASK_QUEUE, PROGRESS_TRACKER, create_thread_with_message
from tests.conftest import client


class TestBackgroundTaskFailures:
    """Test cases for simulating background task failures."""

    def test_git_clone_failure(self, client):
        """Test that a Git clone failure is handled gracefully."""
        # Override DomainManager to raise a GitCommandError
        with patch("manage.web.DOMAIN_MANAGER") as mock_domain_manager:
            mock_domain_manager.clone_repo.side_effect = Exception("Failed to clone repository")
            
            # Trigger a background task that requires Git clone
            response = client.post(
                "/threads/with-message",
                json={"message": "test"}
            )
            
            # Verify response status
            assert response.status_code == 200
            
            # Verify progress tracking and UI feedback
            # (Assuming task_id is captured or mocked for testing)
            # For now, verify that the error is logged and UI shows feedback
            assert "error-msg" in response.text
            
    def test_docker_setup_failure(self, client):
        """Test that a Docker setup failure is handled gracefully."""
        # Mock SandboxManager to raise a DockerError
        with patch("manage.web.SandboxManager.get_sandbox_backend") as mock_get_sandbox:
            mock_get_sandbox.side_effect = Exception("Failed to setup Docker")
            
            # Trigger a background task that requires Docker
            # (Assuming a task like `/thread/{tid}/capture` is triggered)
            response = client.post(
                "/thread/123/capture",
                json={"thread_id": "123"}
            )
            
            # Verify response status
            assert response.status_code == 200
            
            # Verify progress tracking and UI feedback
            assert "error-msg" in response.text
            
    def test_general_task_failure(self):
        """Test that a general task failure is handled gracefully."""
        # Override task_func in BackgroundTaskQueue to raise an exception
        def failing_task_func(*args, **kwargs):
            raise Exception("Task failed")
        
        # Schedule a task that will fail
        task_id = TASK_QUEUE.schedule_task(failing_task_func, arg1="test")
        
        # Simulate progress update before failure
        TASK_QUEUE.update_progress(task_id, 0.5)
        
        # Verify progress tracking
        assert PROGRESS_TRACKER.get_progress(task_id) == 0.5
        assert PROGRESS_TRACKER.get_task_status(task_id) == "in_progress"
        
        # Simulate failure
        TASK_QUEUE.update_progress(task_id, 1.0, error="Task failed")
        
        # Verify progress tracking after failure
        assert PROGRESS_TRACKER.get_task_status(task_id) == "failed"
        assert PROGRESS_TRACKER.get_task_error(task_id) == "Task failed"
        
    def test_partial_progress_update(self):
        """Test that partial progress updates are tracked correctly."""
        def partial_progress_task(*args, **kwargs):
            # Simulate a task that updates progress but fails
            TASK_QUEUE.update_progress(task_id, 0.5)
            raise Exception("Task failed midway")
        
        # Schedule a task
        task_id = TASK_QUEUE.schedule_task(partial_progress_task, arg1="test")
        
        # Verify progress tracking during partial progress
        assert PROGRESS_TRACKER.get_progress(task_id) == 0.5
        assert PROGRESS_TRACKER.get_task_status(task_id) == "in_progress"
        
        # Simulate failure after partial progress
        TASK_QUEUE.update_progress(task_id, 1.0, error="Task failed midway")
        
        # Verify progress tracking after failure
        assert PROGRESS_TRACKER.get_task_status(task_id) == "failed"
        assert PROGRESS_TRACKER.get_task_error(task_id) == "Task failed midway"
        
    def test_completion_failure_states(self):
        """Test tasks that complete successfully and tasks that fail."""
        # Test successful completion
        def successful_task(*args, **kwargs):
            return "Task completed successfully"
        
        task_id_success = TASK_QUEUE.schedule_task(successful_task, arg1="test")
        TASK_QUEUE.update_progress(task_id_success, 1.0)
        
        assert PROGRESS_TRACKER.get_task_status(task_id_success) == "completed"
        assert PROGRESS_TRACKER.get_task_result(task_id_success) == "Task completed successfully"
        
        # Test failure
        def failing_task(*args, **kwargs):
            raise Exception("Task failed")
        
        task_id_failure = TASK_QUEUE.schedule_task(failing_task, arg1="test")
        TASK_QUEUE.update_progress(task_id_failure, 1.0, error="Task failed")
        
        assert PROGRESS_TRACKER.get_task_status(task_id_failure) == "failed"
        assert PROGRESS_TRACKER.get_task_error(task_id_failure) == "Task failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])