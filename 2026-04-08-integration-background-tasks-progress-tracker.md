# Integration Plan: BackgroundTaskQueue and ProgressTracker

## **Reason for Change**
The current implementation of `web.py` uses FastAPI's `BackgroundTasks` for scheduling background tasks, but it does not leverage the dedicated `BackgroundTaskQueue` and `ProgressTracker` classes. This results in:
- No explicit task scheduling for Git clone and Docker setup.
- No progress tracking in the web UI, preventing dynamic updates.
- No integration with `ProgressTracker` for real-time progress updates.

This plan ensures that:
1. `BackgroundTaskQueue` is used for scheduling background tasks.
2. `ProgressTracker` is integrated into `render_thread` for dynamic UI updates.
3. Git clone and Docker setup tasks are explicitly scheduled.

## **Proposed Tests**
### **Test Files and Scenarios**
1. **`test_web_integration.py`**
   - **Test**: `test_schedule_background_task`
     - **Scenario**: Verify that `TASK_QUEUE.schedule_task` is used to schedule background tasks.
   - **Test**: `test_update_task_progress`
     - **Scenario**: Verify that progress updates are correctly tracked in both `TASK_QUEUE` and `PROGRESS_TRACKER`.
   - **Test**: `test_task_completion`
     - **Scenario**: Verify that task completion and result retrieval work as expected.

2. **`test_thread_rendering.py`**
   - **Test**: `test_render_thread_with_task_progress`
     - **Scenario**: Verify that `render_thread` fetches task progress from `PROGRESS_TRACKER` and renders it dynamically.
   - **Test**: `test_render_thread_with_completed_tasks`
     - **Scenario**: Verify that completed task results are rendered in the UI.
   - **Test**: `test_render_thread_with_failed_tasks`
     - **Scenario**: Verify that failed task errors are rendered in the UI.

3. **New Test File: `test_background_task_scheduling.py`**
   - **Test**: `test_schedule_git_clone_task`
     - **Scenario**: Verify that Git clone tasks are explicitly scheduled via `TASK_QUEUE`.
   - **Test**: `test_schedule_docker_setup_task`
     - **Scenario**: Verify that Docker setup tasks are explicitly scheduled via `TASK_QUEUE`.

### **Test Coverage**
- **Normal Cases**: Task scheduling, progress updates, and UI rendering.
- **Edge Cases**: Missing dependencies (e.g., Docker, Git), task failures, and progress tracking errors.
- **Error Cases**: Invalid task arguments, missing task IDs, and race conditions.

## **Proposed Changes**
### **Files to Modify**
1. **`/workspace/manage/web.py`**
   - **Function**: `create_thread_with_message`
     - Replace FastAPI's `BackgroundTasks` with `TASK_QUEUE.schedule_task`.
     - Explicitly schedule Git clone and Docker setup tasks.
   - **Function**: `render_thread`
     - Add logic to query `PROGRESS_TRACKER` for task status and progress.
     - Render progress bars and status indicators dynamically.
   - **Function**: `_process_message`
     - Update to use `TASK_QUEUE` for task execution and `PROGRESS_TRACKER` for progress updates.

2. **`/workspace/manage/background_tasks.py`**
   - **No changes required** (already supports task scheduling and progress tracking).

3. **`/workspace/manage/progress_tracker.py`**
   - **No changes required** (already supports progress tracking and status updates).

### **High-Level Approach**
1. **Replace FastAPI's `BackgroundTasks` with `TASK_QUEUE.schedule_task`** in `create_thread_with_message`.
2. **Explicitly schedule Git clone and Docker setup tasks** in `create_thread_with_message` or `_process_message`.
3. **Integrate `PROGRESS_TRACKER` into `render_thread`** to fetch task status and progress dynamically.
4. **Update `_process_message`** to use `TASK_QUEUE` for task execution and `PROGRESS_TRACKER` for progress updates.

## **Expected Outcomes**
1. **User-Visible Behavior**
   - Threads are created with minimal data and background tasks are scheduled asynchronously.
   - The web view renders only the request initially and updates dynamically as background tasks complete.
   - Progress bars and status indicators are displayed for background tasks.

2. **Test Results**
   - All existing tests pass.
   - New tests validate task scheduling, progress tracking, and dynamic UI updates.

3. **Performance and Correctness**
   - Background tasks do not block the web view.
   - Progress tracking is lightweight and does not overcomplicate the codebase.
   - Error handling is robust for missing dependencies (e.g., Docker, Git).

## **Risks and Considerations**
1. **Backward Compatibility**
   - Ensure that existing tests and functionality remain intact.
   - Verify that `BackgroundTaskQueue` and `ProgressTracker` are compatible with FastAPI's async patterns.

2. **Error Handling**
   - Add robust error handling for missing dependencies (e.g., Docker, Git).
   - Ensure that task failures are gracefully handled and displayed in the UI.

3. **Dynamic UI Updates**
   - Ensure that `render_thread` correctly fetches and displays task progress and status.
   - Avoid race conditions when updating the UI dynamically.

4. **Testing**
   - Write comprehensive tests for task scheduling, progress tracking, and UI rendering.
   - Test edge cases such as missing dependencies and task failures.

## **Implementation Steps**
1. **Modify `create_thread_with_message`** to use `TASK_QUEUE.schedule_task` for background tasks.
2. **Explicitly schedule Git clone and Docker setup tasks** in `create_thread_with_message` or `_process_message`.
3. **Update `render_thread`** to query `PROGRESS_TRACKER` for task status and progress.
4. **Update `_process_message`** to use `TASK_QUEUE` for task execution and `PROGRESS_TRACKER` for progress updates.
5. **Write new tests** for task scheduling, progress tracking, and dynamic UI updates.
6. **Run all tests** to validate the implementation.

---