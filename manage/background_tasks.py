"""
Background Task Queue for handling asynchronous operations like description generation,
Git clone, and Docker setup.

This module provides a lightweight in-memory queue for scheduling tasks and tracking progress.
"""

import asyncio
import uuid
from typing import Any, Callable, Dict, Optional, Tuple


class BackgroundTaskQueue:
    """
    A lightweight in-memory queue for scheduling background tasks and tracking progress.
    
    Attributes:
        _tasks: Dict[str, Dict[str, Any]]
            A dictionary to store task information and progress.
        _queue: asyncio.Queue
            A queue to manage task execution order.
    """

    def __init__(self):
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._queue = asyncio.Queue()

    def schedule_task(self, task_func: Callable, *args, **kwargs) -> str:
        """
        Schedule a new task for execution.
        
        Args:
            task_func: The function to execute as a background task.
            *args: Positional arguments to pass to the task function.
            **kwargs: Keyword arguments to pass to the task function.
            
        Returns:
            str: The task ID.
        """
        task_id = str(uuid.uuid4())
        self._tasks[task_id] = {
            "status": "pending",
            "progress": 0.0,
            "task_func": task_func,
            "args": args,
            "kwargs": kwargs,
            "result": None,
            "error": None,
        }
        asyncio.create_task(self._queue.put(task_id))
        return task_id

    def update_progress(self, task_id: str, progress: float) -> None:
        """
        Update the progress of a task.
        
        Args:
            task_id: The ID of the task to update.
            progress: The progress value (0.0 to 1.0).
        """
        if task_id in self._tasks:
            self._tasks[task_id]["progress"] = progress
            if progress == 1.0:
                self._tasks[task_id]["status"] = "completed"
            elif progress > 0.0:
                self._tasks[task_id]["status"] = "in_progress"

    def get_task_status(self, task_id: str) -> str:
        """
        Get the status of a task.
        
        Args:
            task_id: The ID of the task.
            
        Returns:
            str: The status of the task.
        """
        return self._tasks.get(task_id, {}).get("status", "unknown")

    def get_progress(self, task_id: str) -> float:
        """
        Get the progress of a task.
        
        Args:
            task_id: The ID of the task.
            
        Returns:
            float: The progress value (0.0 to 1.0).
        """
        return self._tasks.get(task_id, {}).get("progress", 0.0)

    def get_task_result(self, task_id: str) -> Any:
        """
        Get the result of a task.
        
        Args:
            task_id: The ID of the task.
            
        Returns:
            Any: The result of the task.
        """
        return self._tasks.get(task_id, {}).get("result")

    def get_task_error(self, task_id: str) -> Optional[str]:
        """
        Get the error of a task.
        
        Args:
            task_id: The ID of the task.
            
        Returns:
            Optional[str]: The error message, if any.
        """
        return self._tasks.get(task_id, {}).get("error")

    def cancel_task(self, task_id: str) -> None:
        """
        Cancel a task.
        
        Args:
            task_id: The ID of the task to cancel.
        """
        if task_id in self._tasks:
            self._tasks[task_id]["status"] = "cancelled"

    async def _execute_task(self, task_id: str) -> None:
        """
        Execute a task and update its status and result.
        
        Args:
            task_id: The ID of the task to execute.
        """
        if task_id not in self._tasks:
            return
            
        task = self._tasks[task_id]
        task["status"] = "in_progress"
        self.update_progress(task_id, 0.1)
        
        try:
            result = task["task_func"](*task["args"], **task["kwargs"])
            task["result"] = result
            task["status"] = "completed"
            self.update_progress(task_id, 1.0)
        except Exception as e:
            task["error"] = str(e)
            task["status"] = "failed"
            self.update_progress(task_id, 1.0)

    async def process_queue(self) -> None:
        """
        Process the task queue.
        """
        while True:
            task_id = await self._queue.get()
            await self._execute_task(task_id)
            self._queue.task_done()