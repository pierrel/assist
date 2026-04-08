# Implementation Plan: Background Task Queue and Thread Rendering

## Reason for Change
This change introduces a robust background task queue system for handling asynchronous operations like description generation, Git cloning, and Docker setup. It also enhances thread rendering to dynamically update the web view as tasks complete. The goal is to:
- Decouple long-running operations from the user experience.
- Provide real-time progress tracking for background tasks.
- Ensure thread rendering is minimal initially and dynamically updates as tasks complete.

## Proposed Tests

### 1. Background Task Queue Tests
- **File**: `/workspace/tests/manage/test_background_tasks.py`
- **Scenarios**:
  - **Normal Case**: Schedule a background task and verify progress updates.
  - **Edge Case**: Schedule multiple tasks and verify progress tracking for each.
  - **Error Case**: Handle task failures gracefully and report errors.

### 2. Thread Creation Logic Tests
- **File**: `/workspace/tests/manage/test_thread_creation.py`
- **Scenarios**:
  - **Normal Case**: Create a thread with a placeholder description and verify background tasks are scheduled.
  - **Edge Case**: Create a thread with invalid input and verify error handling.

### 3. Thread Rendering Logic Tests
- **File**: `/workspace/tests/manage/test_thread_rendering.py`
- **Scenarios**:
  - **Normal Case**: Render a thread with only the request initially, then dynamically update as background tasks complete.
  - **Edge Case**: Render a thread with no background tasks and verify minimal rendering.

### 4. Progress Tracking Tests
- **File**: `/workspace/tests/manage/test_progress_tracker.py`
- **Scenarios**:
  - **Normal Case**: Track progress for a background task and verify real-time updates.
  - **Edge Case**: Track progress for multiple tasks and verify updates for each.

## Proposed Changes

### 1. Background Task Queue (`/workspace/manage/background_tasks.py`)
- **New Module**: `BackgroundTaskQueue` class to manage task scheduling and progress tracking.
- **Key Features**:
  - Use `asyncio.Queue` for task scheduling.
  - Track task progress with a dictionary of task IDs and statuses.
  - Support for task cancellation and error handling.
- **Functions**:
  - `schedule_task(task_func, *args, **kwargs)`: Schedule a new task.
  - `update_progress(task_id, progress)`: Update task progress.
  - `get_task_status(task_id)`: Get the status of a task.
  - `cancel_task(task_id)`: Cancel a task.

### 2. Thread Creation Logic (`/workspace/manage/web.py`)
- **Update `new_thread_with_message`**:
  - Create a minimal thread with only the request and a placeholder description.
  - Schedule background tasks for description generation, Git clone, and Docker setup using `BackgroundTasks`.
  - Use the `BackgroundTaskQueue` to manage task scheduling.
- **New Functions**:
  - `schedule_background_tasks(thread_id, request)`: Schedule background tasks for a thread.

### 3. Thread Rendering Logic (`/workspace/manage/web.py`)
- **Update `render_thread`**:
  - Render only the request initially.
  - Dynamically update the web view as background tasks complete.
  - Add support for rendering task progress indicators.
- **New Functions**:
  - `render_task_progress(task_id)`: Render progress for a background task.
  - `update_rendered_thread(thread_id)`: Update the rendered thread with new task results.

### 4. Progress Tracking (`/workspace/manage/progress_tracker.py`)
- **New Module**: `ProgressTracker` class to track progress for background tasks.
- **Key Features**:
  - Track task progress in real-time.
  - Support for task status updates (e.g., pending, in_progress, completed, failed).
  - Store task results and errors.
- **Functions**:
  - `track_task(task_id, task_func, *args, **kwargs)`: Track a task and update progress.
  - `get_task_results(task_id)`: Get the results of a task.
  - `get_task_errors(task_id)`: Get any errors from a task.

## Expected Outcomes

### User-Visible Behavior
- **Thread Creation**: Threads are created with minimal information and background tasks are scheduled asynchronously.
- **Thread Rendering**: Threads are rendered with only the initial request, and dynamically updated as background tasks complete.
- **Progress Tracking**: Users can see real-time progress for background tasks.

### Test Results
- All new tests must pass.
- Existing tests must continue to pass.

### Performance and Correctness
- Background tasks should not block the user experience.
- Progress tracking should be accurate and real-time.
- Thread rendering should be minimal initially and dynamically updated.

## Risks / Considerations

### Edge Cases
- **Task Failures**: Handle task failures gracefully and report errors to the user.
- **Concurrent Tasks**: Ensure progress tracking works correctly for multiple concurrent tasks.
- **Thread Deletion**: Ensure background tasks are cancelled if the thread is deleted.

### Backward Compatibility
- Ensure changes do not break existing functionality.
- Maintain compatibility with existing thread rendering and creation logic.

### Decisions
- **Task Queue**: Use `asyncio.Queue` for simplicity and reliability.
- **Progress Tracking**: Use a dictionary to track task statuses and results.
- **Thread Rendering**: Dynamically update the rendered thread as tasks complete.

### Dependencies
- Ensure all dependencies (e.g., FastAPI, asyncio) are available and compatible.

---