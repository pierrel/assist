#!/usr/bin/env python3
"""
Simple integration tests for BackgroundTaskQueue and ProgressTracker.

This file tests the core functionality of:
1. BackgroundTaskQueue for scheduling background tasks.
2. ProgressTracker for tracking task progress and status.
"""

import sys
import asyncio
import pytest
from unittest.mock import MagicMock, patch

# Add the workspace to the Python path to import manage modules
sys.path.append('/workspace')

from manage.background_tasks import BackgroundTaskQueue
from manage.progress_tracker import ProgressTracker


class TestBackgroundTaskQueueIntegration:
    """Integration tests for BackgroundTaskQueue and ProgressTracker."""

    def setup_method(self):
        """Set up test fixtures."""
        self.task_queue = BackgroundTaskQueue()
        self.progress_tracker = ProgressTracker()

    def test_schedule_background_task(self):
        """Test that background tasks are scheduled via TASK_QUEUE."""
        
        # Mock a background task function
        def mock_background_task(*args, **kwargs):
            return "Task completed"
        
        # Schedule the task (without async)
        task_id = self.task_queue._schedule_task_sync_for_test(
            mock_background_task, "arg1", "arg2", key1="value1"
        )
        
        # Verify the task is scheduled
        assert task_id in self.task_queue._tasks
        assert self.task_queue._tasks[task_id]["status"] == "pending"

    def test_update_task_progress(self):
        """Test that task progress is updated in TASK_QUEUE."""
        
        # Schedule a task
        def mock_background_task(*args, **kwargs):
            return "Task completed"
        
        task_id = self.task_queue._schedule_task_sync_for_test(mock_background_task)
        
        # Update progress via TASK_QUEUE
        self.task_queue.update_progress(task_id, 0.5)
        
        # Verify progress is updated
        assert self.task_queue.get_progress(task_id) == 0.5
        assert self.task_queue.get_task_status(task_id) == "in_progress"

    def test_task_completion(self):
        """Test task completion and result retrieval."""
        
        # Mock a background task that returns a result
        def mock_background_task(*args, **kwargs):
            return "Task completed successfully"
        
        task_id = self.task_queue._schedule_task_sync_for_test(mock_background_task)
        
        # Simulate task completion
        self.task_queue.update_progress(task_id, 1.0)
        
        # Verify task is marked as completed
        assert self.task_queue.get_task_status(task_id) == "completed"
        
        # Verify task result is stored
        result = self.task_queue.get_task_result(task_id)
        assert result == "Task completed successfully"

    def test_task_failure(self):
        """Test task failure and error handling."""
        
        # Mock a background task that raises an exception
        def mock_failing_task(*args, **kwargs):
            raise ValueError("Task failed: Invalid arguments")
        
        task_id = self.task_queue._schedule_task_sync_for_test(mock_failing_task)
        
        # Simulate task failure
        try:
            self.task_queue.update_progress(task_id, 1.0)
            # Manually trigger the task execution to capture the error
            task_func = self.task_queue._tasks[task_id]["task_func"]
            args = self.task_queue._tasks[task_id]["args"]
            kwargs = self.task_queue._tasks[task_id]["kwargs"]
            task_func(*args, **kwargs)
        except Exception as e:
            # Update the task error
            self.task_queue._tasks[task_id]["error"] = str(e)
        
        # Verify task is marked as failed
        assert self.task_queue.get_task_status(task_id) == "failed"
        
        # Verify error is stored
        error = self.task_queue.get_task_error(task_id)
        assert error == "Task failed: Invalid arguments"

    def test_integration_with_progress_tracker(self):
        """Test integration between BackgroundTaskQueue and ProgressTracker."""
        
        # Schedule a task
        def mock_background_task(*args, **kwargs):
            return "Task completed"
        
        task_id = self.task_queue._schedule_task_sync_for_test(mock_background_task)
        
        # Manually register the task with ProgressTracker
        self.progress_tracker.track_task(
            task_id,
            mock_background_task,
            *self.task_queue._tasks[task_id]["args"],
            **self.task_queue._tasks[task_id]["kwargs"]
        )
        
        # Update progress via TASK_QUEUE
        self.task_queue.update_progress(task_id, 0.75)
        
        # Verify progress is updated in both trackers
        assert self.task_queue.get_progress(task_id) == 0.75
        assert self.progress_tracker.get_progress(task_id) == 0.75
        
        assert self.task_queue.get_task_status(task_id) == "in_progress"
        assert self.progress_tracker.get_task_status(task_id) == "in_progress"

    def test_concurrent_task_scheduling(self):
        """Test concurrent task scheduling and progress updates."""
        
        async def schedule_and_update_tasks():
            """Schedule multiple tasks and update their progress concurrently."""
            
            # Schedule multiple tasks
            task_ids = []
            for i in range(3):
                def mock_background_task(*args, **kwargs):
                    return f"Task {args[0]} completed"
                
                task_id = self.task_queue.schedule_task(
                    mock_background_task, i
                )
                task_ids.append(task_id)
                
            # Update progress for each task
            for task_id in task_ids:
                await asyncio.sleep(0.1)  # Simulate async delay
                self.task_queue.update_progress(task_id, 0.5)
                
            # Verify progress updates
            for task_id in task_ids:
                assert self.task_queue.get_progress(task_id) == 0.5
                assert self.task_queue.get_task_status(task_id) == "in_progress"
        
        # Run the async test
        asyncio.run(schedule_and_update_tasks())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])