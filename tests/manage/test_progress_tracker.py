#!/usr/bin/env python3
"""
Tests for the ProgressTracker module.

This module tests the tracking of progress for background tasks in real-time.
"""

import pytest
from unittest.mock import MagicMock, patch
from manage.progress_tracker import ProgressTracker


class TestProgressTracker:
    """Test the ProgressTracker class."""

    def setup_method(self):
        """Set up a fresh ProgressTracker for each test."""
        self.tracker = ProgressTracker()

    def test_track_task(self):
        """Test tracking a task and updating progress."""
        task_id = "test_task_id"
        
        def mock_task():
            return "task_result"
        
        # Track the task
        self.tracker.track_task(task_id, mock_task)
        
        # Verify initial status
        assert self.tracker.get_task_status(task_id) == "pending"
        
        # Simulate progress updates
        self.tracker.update_progress(task_id, 0.3)
        assert self.tracker.get_task_status(task_id) == "in_progress"
        assert self.tracker.get_progress(task_id) == 0.3
        
        # Simulate task completion
        self.tracker.update_progress(task_id, 1.0)
        assert self.tracker.get_task_status(task_id) == "completed"
        assert self.tracker.get_task_result(task_id) == "task_result"

    def test_task_failure(self):
        """Test tracking a failed task."""
        task_id = "test_task_id"
        
        def mock_task():
            raise ValueError("Task failed")
        
        # Track the task
        self.tracker.track_task(task_id, mock_task)
        
        # Simulate task failure
        with patch.object(self.tracker, '_execute_task') as mock_execute:
            mock_execute.side_effect = ValueError("Task failed")
            
            try:
                self.tracker._execute_task(task_id)
            except ValueError:
                pass
        
        # Verify task status
        assert self.tracker.get_task_status(task_id) == "failed"
        assert self.tracker.get_task_error(task_id) == "Task failed"

    def test_multiple_tasks(self):
        """Test tracking multiple tasks."""
        task_ids = ["task_1", "task_2", "task_3"]
        
        for task_id in task_ids:
            def mock_task():
                return f"{task_id}_result"
            
            self.tracker.track_task(task_id, mock_task)
            
            # Simulate progress
            self.tracker.update_progress(task_id, 0.5)
            assert self.tracker.get_task_status(task_id) == "in_progress"
            assert self.tracker.get_progress(task_id) == 0.5
            
            # Simulate completion
            self.tracker.update_progress(task_id, 1.0)
            assert self.tracker.get_task_status(task_id) == "completed"
            assert self.tracker.get_task_result(task_id) == f"{task_id}_result"

    def test_cancel_task(self):
        """Test cancelling a task."""
        task_id = "test_task_id"
        
        def mock_task():
            return "task_result"
        
        self.tracker.track_task(task_id, mock_task)
        
        # Cancel the task
        self.tracker.cancel_task(task_id)
        
        # Verify task status
        assert self.tracker.get_task_status(task_id) == "cancelled"

    def test_get_task_results(self):
        """Test retrieving task results."""
        task_id = "test_task_id"
        
        def mock_task():
            return "task_result"
        
        self.tracker.track_task(task_id, mock_task)
        
        # Simulate completion
        self.tracker.update_progress(task_id, 1.0)
        
        # Verify task result
        assert self.tracker.get_task_result(task_id) == "task_result"

    def test_get_task_errors(self):
        """Test retrieving task errors."""
        task_id = "test_task_id"
        
        def mock_task():
            raise ValueError("Task failed")
        
        self.tracker.track_task(task_id, mock_task)
        
        # Simulate failure
        with patch.object(self.tracker, '_execute_task') as mock_execute:
            mock_execute.side_effect = ValueError("Task failed")
            
            try:
                self.tracker._execute_task(task_id)
            except ValueError:
                pass
        
        # Verify task error
        assert self.tracker.get_task_error(task_id) == "Task failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])