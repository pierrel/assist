# Implementation Plan: Background Task Queue and Thread Updates

## Overview
This plan outlines the implementation of a background task queue for handling non-blocking operations (e.g., description generation, Git clone, Docker setup) in the `/workspace` project. The changes will:
- Create a new background task system using FastAPI's `BackgroundTasks`.
- Update thread creation and rendering logic to support minimal initial rendering and dynamic updates.
- Add progress tracking for background tasks.

---

## Reason for Change
The current system blocks the web view while background tasks (e.g., description generation, Git clone, Docker setup) are running. This degrades user experience and can cause timeouts. The goal is to:
- Allow the web view to load immediately with minimal data.
- Dynamically update the view as background tasks complete.
- Provide real-time progress tracking for background tasks.

---

## Proposed Tests
### New Test Files
1. **`test_background_task_progress.py`**
   - **Purpose**: Validate progress tracking for background tasks.
   - **Scenarios**:
     - Task starts with 0% progress.
     - Progress updates in real-time (e.g., 25%, 50%, 100%).
     - Task completion triggers final updates.
     - Error handling (e.g., task failure, network issues).
   - **Location**: `/workspace/tests/manage/`.

2. **`test_thread_minimal_creation.py`**
   - **Purpose**: Validate minimal thread creation and dynamic rendering.
   - **Scenarios**:
     - Thread created with only a request and placeholder description.
     - Web view renders only the request initially.
     - Background tasks (e.g., description generation) update the view dynamically.
     - Thread state reflects progress (e.g., `pending`, `in_progress`, `completed`).
   - **Location**: `/workspace/tests/manage/`.

### Updated Test File
- **`test_thread_loading_new.py`**
  - **Update**: Add assertions to verify that background tasks are scheduled and progress is tracked.

---

## Proposed Changes
### 1. New Files
#### **`/workspace/manage/background_tasks.py`**
- **Purpose**: Define background task functions and their dependencies.
- **Functions**:
  - `generate_description(tid, text)`: Generates a description for a thread.
  - `clone_git_repo(tid, repo_url)`: Clones a Git repository.
  - `setup_docker(tid, dockerfile)`: Sets up Docker for a thread.
- **Dependencies**: Use FastAPI's `BackgroundTasks` for scheduling.
- **Progress Tracking**: Integrate with `progress_tracker.py`.

#### **`/workspace/manage/progress_tracker.py`**
- **Purpose**: Track progress for background tasks in real-time.
- **Features**:
  - Store task progress (e.g., `0%`, `25%`, `100%`).
  - Update progress dynamically (e.g., via callbacks or polling).
  - Handle errors (e.g., task failure, network issues).
- **Data Structure**: Use a dictionary to map task IDs to progress objects.

### 2. Updated Files
#### **`/workspace/manage/web.py`**
- **`new_thread_with_message`**:
  - **Change**: Create a minimal thread with only the request and a placeholder description.
  - **Action**: Schedule background tasks for description generation, Git clone, and Docker setup using `BackgroundTasks`.
  - **Example**:
    ```python
    def new_thread_with_message(request, text):
        tid = create_thread()
        create_message(tid, text, role="user")
        
        background_tasks = BackgroundTasks()
        background_tasks.add_task(generate_description, tid, text)
        background_tasks.add_task(clone_git_repo, tid, request.repo_url)
        background_tasks.add_task(setup_docker, tid, request.dockerfile)
        
        return redirect_to_thread(tid, background_tasks=background_tasks)
    ```

- **`render_thread`**:
  - **Change**: Render only the request initially.
  - **Action**: Dynamically update the web view as background tasks complete.
  - **Example**:
    ```python
    def render_thread(tid):
        thread = get_thread(tid)
        progress = get_progress(tid)  # From progress_tracker
        
        html = f"<div class='request'>{thread.request}</div>"
        if progress:
            html += f"<div class='progress'>{progress.percent}%</div>"
        if thread.description:
            html += f"<div class='description'>{thread.description}</div>"
        
        return html
    ```

---

## Expected Outcomes
1. **User Experience**:
   - Web view loads immediately with minimal data.
   - Progress updates appear in real-time (e.g., 25%, 50%, 100%).
   - No blocking or timeouts during background task execution.

2. **Codebase**:
   - All existing tests pass (e.g., `test_thread_loading_new.py`).
   - New tests validate background task progress and minimal thread creation.
   - No breaking changes to existing APIs or workflows.

3. **Progress Tracking**:
   - Real-time updates for background tasks.
   - Error handling for task failures.

---

## Risks / Considerations
1. **Backward Compatibility**:
   - Ensure existing thread creation and rendering logic remains compatible.
   - Avoid breaking changes to the `Thread` model or API.

2. **Progress Tracking Overhead**:
   - Real-time progress updates may introduce minor latency.
   - Ensure progress tracking does not block the main thread.

3. **Error Handling**:
   - Background task failures should not crash the web view.
   - Provide clear error messages for users (e.g., "Git clone failed: Invalid URL").

4. **Testing**:
   - New tests must cover edge cases (e.g., task cancellation, network issues).
   - Mock background tasks in tests to avoid flakiness.

---

## Implementation Steps
1. **Create `progress_tracker.py`**: Define progress tracking logic and data structures.
2. **Create `background_tasks.py`**: Define background task functions and integrate with `progress_tracker`.
3. **Update `web.py`**: Modify `new_thread_with_message` and `render_thread` to support minimal rendering and dynamic updates.
4. **Add Tests**: Implement `test_background_task_progress.py` and `test_thread_minimal_creation.py`.
5. **Run Tests**: Verify all existing and new tests pass.

---

## Open Questions
1. Should progress tracking use polling or callbacks?
   - **Proposed**: Use callbacks for real-time updates.
2. How should errors be handled in the web view?
   - **Proposed**: Display user-friendly error messages without breaking the UI.
3. Should background tasks be retried on failure?
   - **Proposed**: Yes, with a maximum retry limit (e.g., 3 attempts).