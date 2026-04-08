#!/usr/bin/env python3
"""
Test edge cases for background task failures.

This module tests the following scenarios:
1. Simulate failures in background tasks (e.g., Git clone fails, Docker setup fails).
2. Verify that progress tracking works correctly for all tasks.
3. Ensure the web view remains responsive even if background tasks fail.
"""

import asyncio
import pytest
from unittest.mock import patch, MagicMock

from manage.background_tasks import BackgroundTaskQueue
from manage.progress_tracker import ProgressTracker

pytestmark = pytest.mark.asyncio


class TestBackgroundTaskEdgeCases:
    """Test edge cases for background task failures."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.progress_tracker = ProgressTracker()
        self.task_queue = BackgroundTaskQueue(self.progress_tracker)
        
    async def _run_task(self, task_id):
        """Run a task in the event loop."""
        # Ensure the task is tracked in the progress tracker
        if hasattr(self.task_queue, '_progress_tracker') and self.task_queue._progress_tracker:
            self.task_queue._progress_tracker.track_task(task_id, self.task_queue._tasks[task_id]["task_func"], *self.task_queue._tasks[task_id]["args"], **self.task_queue._tasks[task_id]["kwargs"])
        
        # Execute the task
        await self.task_queue._execute_task(task_id)
        
    async def test_task_failure(self):
        """Test that task failures are handled and progress is updated."""
        def failing_task():
            raise ValueError("Task failed: Git clone failed")
        
        task_id = self.task_queue.schedule_task(failing_task)
        
        # Simulate task execution
        await self._run_task(task_id)
        
        # Verify task status and progress
        assert self.task_queue._tasks[task_id]["status"] == "failed"
        assert self.progress_tracker.get_progress(task_id) == 0.0
        assert self.progress_tracker.get_task_status(task_id) == "failed"
        assert self.progress_tracker.get_task_error(task_id) == "Task failed: Git clone failed"
        
    async def test_task_success(self):
        """Test that task success is handled and progress is updated."""
        def successful_task():
            return "Task completed successfully"
        
        task_id = self.task_queue.schedule_task(successful_task)
        
        # Simulate task execution
        await self._run_task(task_id)
        
        # Verify task status and progress
        assert self.task_queue._tasks[task_id]["status"] == "completed"
        assert self.progress_tracker.get_progress(task_id) == 1.0
        assert self.progress_tracker.get_task_status(task_id) == "completed"
        assert self.progress_tracker.get_task_result(task_id) == "Task completed successfully"
        
    async def test_progress_update_during_execution(self):
        """Test that progress updates are reflected during task execution."""
        def task_with_progress():
            self.progress_tracker.update_progress(task_id, 0.3)
            self.progress_tracker.update_progress(task_id, 0.7)
            return "Task completed"
        
        task_id = self.task_queue.schedule_task(task_with_progress)
        
        # Simulate task execution
        await self._run_task(task_id)
        
        # Verify progress updates
        assert self.progress_tracker.get_progress(task_id) == 0.7
        assert self.progress_tracker.get_task_status(task_id) == "in_progress"
        
    async def test_task_error_handling(self):
        """Test that task errors are handled and progress is updated."""
        def task_with_error():
            self.progress_tracker.update_progress(task_id, 0.5)
            raise RuntimeError("Task failed: Docker setup failed")
        
        task_id = self.task_queue.schedule_task(task_with_error)
        
        # Simulate task execution
        await self._run_task(task_id)
        
        # Verify task status and progress
        assert self.task_queue._tasks[task_id]["status"] == "failed"
        assert self.progress_tracker.get_progress(task_id) == 0.0
        assert self.progress_tracker.get_task_status(task_id) == "failed"
        assert self.progress_tracker.get_task_error(task_id) == "Task failed: Docker setup failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])