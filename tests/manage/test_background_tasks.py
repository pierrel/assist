#!/usr/bin/env python3
"""
Tests for the BackgroundTaskQueue module.

This module tests the scheduling, progress tracking, and completion of background tasks.
"""

import pytest
from unittest.mock import MagicMock, patch
from manage.background_tasks import BackgroundTaskQueue


class TestBackgroundTaskQueue:
    """Test the BackgroundTaskQueue class."""

    def setup_method(self):
        """Set up a fresh BackgroundTaskQueue for each test."""
        self.queue = BackgroundTaskQueue()

    def test_schedule_task(self):
        """Test scheduling a new task."""
        task_id = self.queue.schedule_task(lambda: "done", "arg1", arg2="arg2")
        assert task_id is not None
        assert self.queue.get_task_status(task_id) == "pending"

    def test_update_progress(self):
        """Test updating task progress."""
        task_id = self.queue.schedule_task(lambda: "done", "arg1")
        self.queue.update_progress(task_id, 0.5)
        assert self.queue.get_task_status(task_id) == "in_progress"
        assert self.queue.get_progress(task_id) == 0.5

    def test_task_completion(self):
        """Test task completion and result retrieval."""
        result = "completed"
        
        def mock_task():
            return result
        
        task_id = self.queue.schedule_task(mock_task)
        self.queue.update_progress(task_id, 1.0)
        assert self.queue.get_task_status(task_id) == "completed"
        assert self.queue.get_task_result(task_id) == result

    def test_task_failure(self):
        """Test task failure and error handling."""
        error_msg = "Task failed"
        
        def mock_task():
            raise ValueError(error_msg)
        
        task_id = self.queue.schedule_task(mock_task)
        with patch.object(self.queue, 'update_progress') as mock_update:
            try:
                self.queue._execute_task(task_id)
            except ValueError as e:
                assert str(e) == error_msg
                mock_update.assert_called_with(task_id, 1.0)
                assert self.queue.get_task_status(task_id) == "failed"

    def test_cancel_task(self):
        """Test cancelling a task."""
        task_id = self.queue.schedule_task(lambda: "done")
        self.queue.cancel_task(task_id)
        assert self.queue.get_task_status(task_id) == "cancelled"

    def test_multiple_tasks(self):
        """Test scheduling and tracking multiple tasks."""
        task_ids = []
        for i in range(3):
            task_id = self.queue.schedule_task(lambda i=i: f"task_{i}_done")
            task_ids.append(task_id)
            self.queue.update_progress(task_id, 0.33)
        
        for task_id in task_ids:
            assert self.queue.get_task_status(task_id) == "in_progress"
            assert self.queue.get_progress(task_id) == 0.33

    def test_get_task_status(self):
        """Test retrieving task status."""
        task_id = self.queue.schedule_task(lambda: "done")
        assert self.queue.get_task_status(task_id) == "pending"
        self.queue.update_progress(task_id, 0.5)
        assert self.queue.get_task_status(task_id) == "in_progress"
        self.queue.update_progress(task_id, 1.0)
        assert self.queue.get_task_status(task_id) == "completed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])