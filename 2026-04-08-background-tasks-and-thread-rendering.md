# Implementation Plan: Background Task Queue and Thread Rendering

## Reason for Change
The current implementation of thread creation and rendering is synchronous and blocking, causing delays in the web view's initial load. Background tasks like description generation, Git clone, and Docker setup are not scheduled asynchronously, leading to a poor user experience.

This change will:
- Schedule background tasks asynchronously using FastAPI's `BackgroundTasks`.
- Create minimal threads with only the request and a placeholder description.
- Render only the request initially and dynamically update the web view as background tasks complete.
- Add progress tracking for background tasks.

## Proposed Tests

### Test Files and Scenarios
1. **`test_thread_loading_new.py` (Existing Tests)**
   - **Test Case**: `test_new_thread_with_message_minimal_request`
     - **Scenario**: Create a thread with minimal data (request only) and verify that background tasks are scheduled asynchronously.
     - **Expected**: Thread is created successfully, and background tasks introduce a delay.
   - **Test Case**: `test_render_thread_with_minimal_data`
     - **Scenario**: Render a thread with only the request and verify that the web view updates dynamically.
     - **Expected**: Rendered content includes only the request initially, and updates as background tasks complete.
   - **Test Case**: `test_background_task_execution`
     - **Scenario**: Verify that background tasks (description generation, Git clone, Docker setup) are executed asynchronously and return success statuses.
     - **Expected**: Background tasks complete successfully and return `{'status': 'completed'}`.

2. **New Test File: `test_background_task_progress.py`**
   - **Test Case**: `test_progress_tracking_for_background_tasks`
     - **Scenario**: Verify that progress tracking is implemented for background tasks.
     - **Expected**: Progress updates are reported in real-time, and the web view reflects these updates.
   - **Test Case**: `test_background_task_failure_handling`
     - **Scenario**: Simulate a failure in a background task (e.g., Git clone fails) and verify that the error is handled gracefully.
     - **Expected**: Error status is reported, and the web view updates to reflect the failure.

3. **New Test File: `test_thread_minimal_creation.py`**
   - **Test Case**: `test_thread_creation_with_minimal_data`
     - **Scenario**: Create a thread with only the request and a placeholder description.
     - **Expected**: Thread is created successfully, and background tasks are scheduled.
   - **Test Case**: `test_thread_creation_with_invalid_data`
     - **Scenario**: Attempt to create a thread with invalid data (e.g., empty request).
     - **Expected**: Thread creation fails with a `ValueError`.

## Proposed Changes

### Files to Modify
1. **`/workspace/manage/web.py`**
   - **Function**: `new_thread_with_message`
     - **Change**: Create a minimal thread with only the request and a placeholder description.
     - **Action**: Schedule background tasks for description generation, Git clone, and Docker setup using `BackgroundTasks`.
   - **Function**: `render_thread`
     - **Change**: Render only the request initially.
     - **Action**: Dynamically update the web view as background tasks complete.

2. **`/workspace/manage/background_tasks.py` (New File)**
   - **Function**: `execute_background_tasks`
     - **Change**: Implement a function to execute background tasks (description generation, Git clone, Docker setup).
     - **Action**: Add progress tracking and error handling.

3. **`/workspace/manage/progress_tracker.py` (New File)**
   - **Function**: `track_progress`
     - **Change**: Implement a progress tracker for background tasks.
     - **Action**: Report progress updates in real-time.

### High-Level Approach
1. **Thread Creation**:
   - Use `BackgroundTasks` to schedule background tasks asynchronously.
   - Create a minimal thread with only the request and a placeholder description.

2. **Thread Rendering**:
   - Render only the request initially.
   - Use WebSocket or AJAX to dynamically update the web view as background tasks complete.

3. **Progress Tracking**:
   - Implement a lightweight progress tracker using FastAPI's `BackgroundTasks` and a dictionary to store task statuses.

## Expected Outcomes

### User-Visible Behavior
- The web view loads instantly with only the request displayed.
- Background tasks (description generation, Git clone, Docker setup) run asynchronously.
- The web view updates dynamically as background tasks complete, showing progress and results.
- Errors in background tasks are reported gracefully, and the web view updates to reflect failures.

### Test Results
- All existing tests in `test_thread_loading_new.py` pass.
- New tests in `test_background_task_progress.py` and `test_thread_minimal_creation.py` pass.
- Background tasks are executed asynchronously and introduce delays as expected.
- Progress tracking is implemented and reported in real-time.

### Performance and Correctness
- Thread creation and rendering are non-blocking.
- Background tasks are executed asynchronously without blocking the web view.
- Progress tracking is lightweight and does not impact performance.
- Error handling is robust and user-friendly.

## Risks and Considerations

### Edge Cases
- **Empty Requests**: Ensure that thread creation fails gracefully if the request is empty.
- **Background Task Failures**: Handle failures in background tasks (e.g., Git clone fails) without crashing the application.
- **Dynamic Updates**: Ensure that the web view updates correctly even if background tasks complete out of order.

### Backward Compatibility
- Maintain existing thread creation and rendering logic for non-minimal cases.
- Ensure that existing tests continue to pass.
- Avoid breaking changes to the API or data structures.

### Performance Considerations
- **Background Tasks**: Ensure that background tasks do not consume excessive resources.
- **Progress Tracking**: Keep progress tracking lightweight to avoid performance overhead.
- **Web View Updates**: Use efficient mechanisms (e.g., WebSocket or AJAX) to update the web view dynamically.

### Decisions to Consider
- **Background Task Library**: Use FastAPI's built-in `BackgroundTasks` to avoid introducing new dependencies.
- **Progress Tracking**: Implement a simple dictionary-based tracker to avoid complexity.
- **Dynamic Updates**: Use WebSocket for real-time updates if available, otherwise fall back to AJAX polling.

## Plan for Implementation

1. **Step 1**: Update `new_thread_with_message` to create a minimal thread and schedule background tasks.
2. **Step 2**: Update `render_thread` to render only the request initially and support dynamic updates.
3. **Step 3**: Implement `execute_background_tasks` in `/workspace/manage/background_tasks.py` to handle background tasks and progress tracking.
4. **Step 4**: Implement `track_progress` in `/workspace/manage/progress_tracker.py` to report progress updates.
5. **Step 5**: Write new tests for background task progress tracking and minimal thread creation.
6. **Step 6**: Run all tests to ensure backward compatibility and correctness.

## Approval Request
Please review the plan and let me know if you would like to proceed with the implementation. The plan is available at `/workspace/2026-04-08-background-tasks-and-thread-rendering.md`.