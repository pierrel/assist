# Plan for Testing Edge Cases in Background Tasks

## Overview
This plan outlines the testing strategy for edge cases in background tasks, focusing on:
1. Simulating failures in background tasks (e.g., Git clone, Docker setup).
2. Verifying progress tracking for all tasks.
3. Ensuring the web view remains responsive even if background tasks fail.

## Reason for Change
The current implementation lacks explicit tests for edge cases such as:
- Background task failures (e.g., Git clone errors, Docker setup failures).
- Progress tracking during partial progress updates or failures.
- Web view responsiveness during background task failures.

This plan ensures that the system handles these edge cases gracefully and provides appropriate feedback to users.

---

## Proposed Tests

### 1. Simulate Background Task Failures
#### **Test Cases**
- **Git Clone Failures**:
  - **Scenario**: Simulate a Git clone failure by overriding `DomainManager` in `web.py` to raise a `GitCommandError`.
  - **Expected Behavior**:
    - Progress tracking updates to `failed` with an error message.
    - Web view displays an error message (e.g., `error-msg` class) without crashing.
  - **Test File**: `tests/manage/test_background_tasks.py`

- **Docker Setup Failures**:
  - **Scenario**: Mock `SandboxManager.get_sandbox_backend()` to return `None` or raise a `DockerError`.
  - **Expected Behavior**:
    - Progress tracking updates to `failed` with an error message.
    - Web view displays an error message (e.g., `error-msg` class) without crashing.
  - **Test File**: `tests/manage/test_background_tasks.py`

- **General Task Failures**:
  - **Scenario**: Override `task_func` in `BackgroundTaskQueue.schedule_task()` to raise exceptions (e.g., `RuntimeError`).
  - **Expected Behavior**:
    - Progress tracking updates to `failed` with an error message.
    - Web view displays an error message (e.g., `error-msg` class) without crashing.
  - **Test File**: `tests/manage/test_background_tasks.py`

#### **Test Implementation**
```python
# Example: Simulate a Git clone failure
class MockDomainManager:
    def clone_repo(self, url):
        raise GitCommandError("Failed to clone repository")

def test_git_clone_failure():
    # Override DomainManager in web.py
    web.DOMAIN_MANAGER = MockDomainManager()
    
    # Trigger a background task that requires Git clone
    response = client.post("/threads/with-message", json={"message": "test"})
    
    # Verify progress tracking and UI feedback
    assert response.status_code == 200
    assert "error-msg" in response.text
    assert PROGRESS_TRACKER.get_task_status(task_id) == "failed"
```

---

### 2. Verify Progress Tracking
#### **Test Cases**
- **Partial Progress Updates**:
  - **Scenario**: Simulate a task that updates progress to 0.5 but fails before completion.
  - **Expected Behavior**:
    - Progress tracking updates to `in_progress` with progress = 0.5.
    - Web view displays a progress indicator (e.g., progress bar) and a warning message.
  - **Test File**: `tests/manage/test_progress_tracker.py`

- **Completion/Failure States**:
  - **Scenario**: Test tasks that complete successfully (progress = 1.0, status = `completed`) and tasks that fail (progress < 1.0, status = `failed`).
  - **Expected Behavior**:
    - Progress tracking updates to `completed` or `failed` with appropriate progress and error messages.
    - Web view displays success/error messages and updates the UI accordingly.
  - **Test File**: `tests/manage/test_progress_tracker.py`

#### **Test Implementation**
```python
# Example: Test partial progress update
def test_partial_progress_update():
    task_id = TASK_QUEUE.schedule_task(
        lambda: None,  # Simulate a task that fails
        arg1="test"
    )
    
    # Simulate progress update
    TASK_QUEUE.update_progress(task_id, 0.5)
    
    # Verify progress tracking
    assert PROGRESS_TRACKER.get_progress(task_id) == 0.5
    assert PROGRESS_TRACKER.get_task_status(task_id) == "in_progress"
```

---

### 3. Web View Responsiveness
#### **Test Cases**
- **UI Updates During Failures**:
  - **Scenario**: Trigger a background task failure while viewing `/thread/{tid}`.
  - **Expected Behavior**:
    - Web view updates to show error messages (e.g., `error-msg` class) without crashing.
    - Progress indicators are hidden or updated to reflect the failure.
  - **Test File**: `tests/manage/test_web_integration.py`

- **Progress Indicators**:
  - **Scenario**: Simulate a long-running task (e.g., `asyncio.sleep(5)` in `task_func`).
  - **Expected Behavior**:
    - Progress bars or indicators update dynamically.
    - Web view remains responsive and updates in real-time.
  - **Test File**: `tests/manage/test_web_integration.py`

#### **Test Implementation**
```python
# Example: Test UI responsiveness during failure
def test_ui_responsiveness_during_failure():
    # Trigger a background task failure
    task_id = TASK_QUEUE.schedule_task(
        lambda: None,  # Simulate a task that fails
        arg1="test"
    )
    
    # Simulate failure
    TASK_QUEUE.update_progress(task_id, 0.5)
    TASK_QUEUE.update_progress(task_id, 1.0, error="Task failed")
    
    # Trigger a web request to render the thread
    response = client.get(f"/thread/{task_id}")
    
    # Verify UI responsiveness
    assert response.status_code == 200
    assert "error-msg" in response.text
    assert "Task failed" in response.text
```

---

## Proposed Changes

### Files to Modify
1. **`/workspace/manage/background_tasks.py`**:
   - Add methods to simulate failures (e.g., override `task_func` to raise exceptions).
   - Ensure progress tracking updates correctly during failures.

2. **`/workspace/manage/progress_tracker.py`**:
   - Add methods to verify progress tracking during partial updates and failures.

3. **`/workspace/manage/web.py`**:
   - Mock `DomainManager` and `SandboxManager` to simulate Git clone and Docker setup failures.
   - Ensure the web view updates dynamically during background task failures.

4. **Test Files**:
   - Add new test cases to `tests/manage/test_background_tasks.py` and `tests/manage/test_progress_tracker.py`.

### High-Level Approach
1. **Simulate Failures**:
   - Override `task_func` in `BackgroundTaskQueue` to raise exceptions.
   - Mock `DomainManager` and `SandboxManager` to simulate Git clone and Docker setup failures.

2. **Verify Progress Tracking**:
   - Update `ProgressTracker` to handle partial progress updates and failures.
   - Add assertions to verify progress tracking updates correctly.

3. **Ensure Web View Responsiveness**:
   - Update `web.py` to handle dynamic UI updates during background task failures.
   - Add tests to verify UI responsiveness and error handling.

---

## Expected Outcomes

### User-Visible Behavior
- Background task failures are handled gracefully.
- Progress tracking updates dynamically and reflects the current state of tasks.
- The web view remains responsive and provides appropriate feedback (e.g., error messages, progress indicators).

### Test Results
- All new tests pass, confirming that edge cases are handled correctly.
- Existing tests continue to pass, ensuring backward compatibility.

### Performance and Correctness
- Progress tracking remains efficient and accurate.
- Web view updates are dynamic and responsive.
- Error handling is robust and user-friendly.

---

## Risks and Considerations

### Edge Cases
- **Partial Progress Updates**: Ensure progress tracking updates correctly even if a task fails midway.
- **Concurrent Task Failures**: Verify that progress tracking and UI updates handle multiple concurrent task failures.

### Backward Compatibility
- Ensure changes do not break existing functionality or tests.
- Verify that progress tracking and UI updates are backward-compatible with existing code.

### Error Handling
- Add robust error handling for missing dependencies (e.g., Docker, Git).
- Ensure error messages are clear and user-friendly.

### Dynamic UI Updates
- Ensure `render_thread` correctly fetches and displays task progress and status.
- Verify that UI updates are responsive and do not cause performance issues.

---

## Next Steps
1. **Implement Test Cases**:
   - Add new test cases to `tests/manage/test_background_tasks.py` and `tests/manage/test_progress_tracker.py`.
2. **Simulate Failures**:
   - Override `task_func` in `BackgroundTaskQueue` to raise exceptions.
   - Mock `DomainManager` and `SandboxManager` to simulate Git clone and Docker setup failures.
3. **Verify Progress Tracking**:
   - Update `ProgressTracker` to handle partial progress updates and failures.
4. **Ensure Web View Responsiveness**:
   - Update `web.py` to handle dynamic UI updates during background task failures.
5. **Run Tests**:
   - Execute all tests to confirm that edge cases are handled correctly.
6. **Document Results**:
   - Record outcomes for each edge case and note any uncovered bugs.

---

## Approval Request
Please review the plan and let me know if you'd like me to proceed with implementing the tests and changes.