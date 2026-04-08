#!/usr/bin/env python3
"""
Integration tests for BackgroundTaskQueue and ProgressTracker in web.py.

This file tests the integration of:
1. BackgroundTaskQueue for scheduling background tasks.
2. ProgressTracker for tracking task progress and status.
3. Dynamic UI updates in render_thread based on task progress.
"""

import sys
import asyncio
import pytest
from unittest.mock import MagicMock, patch

# Add the workspace to the Python path to import manage modules
sys.path.append('/workspace')

from manage.web import create_thread_with_message, render_thread
from manage.background_tasks import BackgroundTaskQueue
from manage.progress_tracker import ProgressTracker


class TestWebIntegration:
    """Integration tests for BackgroundTaskQueue and ProgressTracker in web.py."""

    def setup_method(self):
        """Set up test fixtures."""
        self.task_queue = BackgroundTaskQueue()
        self.progress_tracker = ProgressTracker()
        
        # Mock the global TASK_QUEUE and PROGRESS_TRACKER in web.py
        self.web_module = sys.modules['manage.web']
        self.web_module.TASK_QUEUE = self.task_queue
        self.web_module.PROGRESS_TRACKER = self.progress_tracker

    def teardown_method(self):
        """Clean up after tests."""
        # Restore original TASK_QUEUE and PROGRESS_TRACKER
        self.web_module.TASK_QUEUE = BackgroundTaskQueue()
        self.web_module.PROGRESS_TRACKER = ProgressTracker()

    def test_schedule_background_task(self):
        """Test that background tasks are scheduled via TASK_QUEUE."""
        
        # Mock a background task function
        def mock_background_task(*args, **kwargs):
            return "Task completed"
        
        # Mock the task arguments
        task_args = ("arg1", "arg2")
        task_kwargs = {"key1": "value1"}
        
        # Schedule the task
        task_id = self.task_queue.schedule_task(
            mock_background_task, *task_args, **task_kwargs
        )
        
        # Verify the task is scheduled
        assert task_id in self.task_queue._tasks
        assert self.task_queue._tasks[task_id]["status"] == "pending"
        
        # Verify the task is registered with ProgressTracker
        assert task_id in self.progress_tracker._tasks
        assert self.progress_tracker._tasks[task_id]["status"] == "pending"

    def test_update_task_progress(self):
        """Test that task progress is updated in both TASK_QUEUE and PROGRESS_TRACKER."""
        
        # Schedule a task
        def mock_background_task(*args, **kwargs):
            return "Task completed"
        
        task_id = self.task_queue.schedule_task(mock_background_task)
        
        # Update progress via TASK_QUEUE
        self.task_queue.update_progress(task_id, 0.5)
        
        # Verify progress is updated in TASK_QUEUE
        assert self.task_queue.get_progress(task_id) == 0.5
        assert self.task_queue.get_task_status(task_id) == "in_progress"
        
        # Verify progress is also updated in PROGRESS_TRACKER
        assert self.progress_tracker.get_progress(task_id) == 0.5
        assert self.progress_tracker.get_task_status(task_id) == "in_progress"

    def test_task_completion(self):
        """Test task completion and result retrieval."""
        
        # Mock a background task that returns a result
        def mock_background_task(*args, **kwargs):
            return "Task completed successfully"
        
        task_id = self.task_queue.schedule_task(mock_background_task)
        
        # Simulate task completion
        self.task_queue.update_progress(task_id, 1.0)
        
        # Verify task is marked as completed
        assert self.task_queue.get_task_status(task_id) == "completed"
        assert self.progress_tracker.get_task_status(task_id) == "completed"
        
        # Verify task result is stored
        result = self.task_queue.get_task_result(task_id)
        assert result == "Task completed successfully"
        
        # Verify result is also available in PROGRESS_TRACKER
        progress_result = self.progress_tracker.get_task_result(task_id)
        assert progress_result == "Task completed successfully"

    def test_task_failure(self):
        """Test task failure and error handling."""
        
        # Mock a background task that raises an exception
        def mock_failing_task(*args, **kwargs):
            raise ValueError("Task failed: Invalid arguments")
        
        task_id = self.task_queue.schedule_task(mock_failing_task)
        
        # Simulate task failure
        try:
            self.task_queue.update_progress(task_id, 1.0)
            # Manually trigger the task execution to capture the error
            task_func = self.task_queue._tasks[task_id]["task_func"]
            args = self.task_queue._tasks[task_id]["args"]
            kwargs = self.task_queue._tasks[task_id]["kwargs"]
            task_func(*args, **kwargs)
        except Exception as e:
            # Update the task error in both trackers
            self.task_queue._tasks[task_id]["error"] = str(e)
            self.progress_tracker._tasks[task_id]["error"] = str(e)
        
        # Verify task is marked as failed
        assert self.task_queue.get_task_status(task_id) == "failed"
        assert self.progress_tracker.get_task_status(task_id) == "failed"
        
        # Verify error is stored
        error = self.task_queue.get_task_error(task_id)
        assert error == "Task failed: Invalid arguments"
        
        # Verify error is also available in PROGRESS_TRACKER
        progress_error = self.progress_tracker.get_task_error(task_id)
        assert progress_error == "Task failed: Invalid arguments"

    @patch('manage.web.render_thread')
    def test_render_thread_ui_updates(self, mock_render_thread):
        """Test that render_thread fetches task status/progress from PROGRESS_TRACKER."""
        
        # Schedule a task
        def mock_background_task(*args, **kwargs):
            return "Task completed"
        
        task_id = self.task_queue.schedule_task(mock_background_task)
        
        # Update task progress
        self.task_queue.update_progress(task_id, 0.75)
        
        # Mock the task_id to be used in render_thread
        mock_task_id = task_id
        
        # Call render_thread and verify it fetches progress from PROGRESS_TRACKER
        render_thread(mock_task_id)
        
        # Verify that PROGRESS_TRACKER was accessed
        assert self.progress_tracker.get_progress(mock_task_id) == 0.75
        assert self.progress_tracker.get_task_status(mock_task_id) == "in_progress"
        
        # Verify that render_thread was called with the correct task_id
        mock_render_thread.assert_called_once_with(mock_task_id)

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
                assert self.progress_tracker.get_progress(task_id) == 0.5
                assert self.task_queue.get_task_status(task_id) == "in_progress"
                assert self.progress_tracker.get_task_status(task_id) == "in_progress"
        
        # Run the async test
        asyncio.run(schedule_and_update_tasks())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])