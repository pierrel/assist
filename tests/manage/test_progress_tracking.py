#!/usr/bin/env python3
"""
Test cases for verifying progress tracking in background tasks.
This includes:
- Partial progress updates
- Completion/failure states
- Progress tracking accuracy
"""

import pytest
from manage.background_tasks import BackgroundTaskQueue
from manage.progress_tracker import ProgressTracker
from manage.web import TASK_QUEUE, PROGRESS_TRACKER


class TestProgressTracking:
    """Test cases for verifying progress tracking."""

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
        
    def test_completion_state(self):
        """Test that a completed task is tracked correctly."""
        def successful_task(*args, **kwargs):
            return "Task completed successfully"
        
        # Schedule a task
        task_id = TASK_QUEUE.schedule_task(successful_task, arg1="test")
        
        # Simulate progress update to completion
        TASK_QUEUE.update_progress(task_id, 1.0)
        
        # Verify progress tracking
        assert PROGRESS_TRACKER.get_progress(task_id) == 1.0
        assert PROGRESS_TRACKER.get_task_status(task_id) == "completed"
        assert PROGRESS_TRACKER.get_task_result(task_id) == "Task completed successfully"
        
    def test_failure_state(self):
        """Test that a failed task is tracked correctly."""
        def failing_task(*args, **kwargs):
            raise Exception("Task failed")
        
        # Schedule a task
        task_id = TASK_QUEUE.schedule_task(failing_task, arg1="test")
        
        # Simulate progress update to failure
        TASK_QUEUE.update_progress(task_id, 1.0, error="Task failed")
        
        # Verify progress tracking
        assert PROGRESS_TRACKER.get_progress(task_id) == 1.0
        assert PROGRESS_TRACKER.get_task_status(task_id) == "failed"
        assert PROGRESS_TRACKER.get_task_error(task_id) == "Task failed"
        
    def test_multiple_tasks_progress_tracking(self):
        """Test that multiple tasks are tracked correctly."""
        def task1(*args, **kwargs):
            return "Task 1 completed"
        
        def task2(*args, **kwargs):
            raise Exception("Task 2 failed")
        
        def task3(*args, **kwargs):
            TASK_QUEUE.update_progress(task_id, 0.5)
            raise Exception("Task 3 failed midway")
        
        # Schedule multiple tasks
        task_id_1 = TASK_QUEUE.schedule_task(task1, arg1="test")
        task_id_2 = TASK_QUEUE.schedule_task(task2, arg1="test")
        task_id_3 = TASK_QUEUE.schedule_task(task3, arg1="test")
        
        # Simulate progress updates
        TASK_QUEUE.update_progress(task_id_1, 1.0)
        TASK_QUEUE.update_progress(task_id_2, 1.0, error="Task 2 failed")
        TASK_QUEUE.update_progress(task_id_3, 0.5)
        TASK_QUEUE.update_progress(task_id_3, 1.0, error="Task 3 failed midway")
        
        # Verify progress tracking for all tasks
        assert PROGRESS_TRACKER.get_progress(task_id_1) == 1.0
        assert PROGRESS_TRACKER.get_task_status(task_id_1) == "completed"
        assert PROGRESS_TRACKER.get_task_result(task_id_1) == "Task 1 completed"
        
        assert PROGRESS_TRACKER.get_progress(task_id_2) == 1.0
        assert PROGRESS_TRACKER.get_task_status(task_id_2) == "failed"
        assert PROGRESS_TRACKER.get_task_error(task_id_2) == "Task 2 failed"
        
        assert PROGRESS_TRACKER.get_progress(task_id_3) == 1.0
        assert PROGRESS_TRACKER.get_task_status(task_id_3) == "failed"
        assert PROGRESS_TRACKER.get_task_error(task_id_3) == "Task 3 failed midway"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])