# Integration Plan: BackgroundTaskQueue and ProgressTracker

## Overview
Integrate `BackgroundTaskQueue` and `ProgressTracker` in `web.py` to ensure:
1. Background tasks (e.g., Git clone, Docker setup) are scheduled via `BackgroundTaskQueue`.
2. Dynamic UI updates in `render_thread` reflect task progress and completion via `ProgressTracker`.

## Reason for Change
- **Current State**: `BackgroundTaskQueue` and `ProgressTracker` are initialized but not integrated. Background tasks are not scheduled via `TASK_QUEUE`, and `render_thread` does not dynamically update the UI based on task progress.
- **Problem**: Tasks are not scheduled efficiently, and the UI lacks real-time updates for task progress.

## Proposed Tests
### Test Cases to Add
1. **Task Scheduling and Progress Tracking**:
   - **File**: `tests/manage/test_web_integration.py` (new file).
   - **Scenarios**:
     - Schedule a background task via `TASK_QUEUE` and verify progress tracking via `PROGRESS_TRACKER`.
     - Simulate task completion and verify UI updates in `render_thread`.
     - Test error handling for task failures.
   - **Edge Cases**:
     - Invalid task arguments.
     - Progress updates outside valid range (e.g., negative values).
     - Concurrent task scheduling and progress updates.

2. **UI Integration**:
   - **File**: `tests/manage/test_web_integration.py`.
   - **Scenarios**:
     - Verify that `render_thread` fetches task status/progress from `PROGRESS_TRACKER`.
     - Test dynamic UI updates for task progress and completion.

### Test Files to Update
- **`test_background_tasks.py`**: Add integration tests with `ProgressTracker`.
- **`test_progress_tracker.py`**: Add integration tests with `BackgroundTaskQueue`.

## Proposed Changes
### Files to Modify
1. **`manage/web.py`**:
   - **Changes**:
     - Replace FastAPI's `BackgroundTasks` with `TASK_QUEUE.schedule_task` for background tasks (e.g., Git clone, Docker setup).
     - Integrate `PROGRESS_TRACKER` with `render_thread` to dynamically update the UI based on task progress.
     - Add logic to poll `PROGRESS_TRACKER` for task updates (e.g., via AJAX or WebSocket).
   - **Functions to Update**:
     - `create_thread_with_message`: Schedule background tasks via `TASK_QUEUE`.
     - `render_thread`: Fetch task status/progress from `PROGRESS_TRACKER` and update UI.

2. **`manage/background_tasks.py`**:
   - **Changes**:
     - Add a method to `BackgroundTaskQueue` to automatically register tasks with `ProgressTracker` when scheduled.
     - Example: `register_with_progress_tracker(task_id, task_func, *args, **kwargs)`.

3. **`manage/progress_tracker.py`**:
   - **Changes**:
     - Add a method to `ProgressTracker` to handle task errors and log them.
     - Example: `log_task_error(task_id, error_message)`.

### High-Level Approach
1. **Task Scheduling**:
   - Modify `create_thread_with_message` to use `TASK_QUEUE.schedule_task` for background tasks.
   - Ensure `PROGRESS_TRACKER` is updated via `update_progress` during task execution.

2. **Dynamic UI Updates**:
   - Modify `render_thread` to fetch task status/progress from `PROGRESS_TRACKER`.
   - Update UI elements (e.g., progress bars, completion messages) dynamically.

3. **Error Handling**:
   - Add logic to capture and log task errors in both `BackgroundTaskQueue` and `ProgressTracker`.

## Expected Outcomes
1. **Functionality**:
   - Background tasks are scheduled via `BackgroundTaskQueue`.
   - UI dynamically updates to reflect task progress and completion.
   - Task errors are captured and logged.

2. **Test Results**:
   - All new integration tests pass.
   - Existing tests for `BackgroundTaskQueue` and `ProgressTracker` continue to pass.

3. **Performance**:
   - No significant performance degradation.
   - Efficient task scheduling and progress tracking.

## Risks / Considerations
1. **Backward Compatibility**:
   - Ensure changes do not break existing functionality in `web.py`.
   - Test integration with FastAPI's `BackgroundTasks`.

2. **Concurrency Issues**:
   - Ensure thread-safe access to `BackgroundTaskQueue` and `ProgressTracker`.
   - Test concurrent task scheduling and progress updates.

3. **Edge Cases**:
   - Handle invalid task arguments and progress updates gracefully.
   - Ensure task errors are logged and do not crash the application.

4. **UI Updates**:
   - Ensure dynamic UI updates do not interfere with other UI components.
   - Test UI updates for various task scenarios (e.g., success, failure, cancellation).