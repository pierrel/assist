#!/usr/bin/env python3

"""
Test script to verify the ProgressTracker class.
"""

import sys
import os

# Add the workspace to the Python path
sys.path.append(os.path.abspath('/workspace'))

from manage.progress_tracker import ProgressTracker

def test_progress_tracker():
    """Test the ProgressTracker class."""
    tracker = ProgressTracker()
    
    # Test tracking a task
    def dummy_task(a, b):
        return a + b
    
    task_id = "test_task_1"
    tracker.track_task(task_id, dummy_task, 2, 3)
    
    # Check initial status
    assert tracker.get_task_status(task_id) == "pending"
    assert tracker.get_progress(task_id) == 0.0
    assert tracker.get_task_result(task_id) is None
    assert tracker.get_task_error(task_id) is None
    
    # Simulate progress update
    tracker.update_progress(task_id, 0.5)
    assert tracker.get_progress(task_id) == 0.5
    assert tracker.get_task_status(task_id) == "in_progress"
    
    # Simulate task completion
    tracker._execute_task(task_id)
    assert tracker.get_task_status(task_id) == "completed"
    assert tracker.get_progress(task_id) == 1.0
    assert tracker.get_task_result(task_id) == 5
    assert tracker.get_task_error(task_id) is None
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_progress_tracker()