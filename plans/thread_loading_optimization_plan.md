# Thread Loading Optimization Plan

## Overview
Optimize thread creation and loading to ensure the web view loads instantly with only the request in the thread. Move the following operations to the background:
- Thread description generation
- Git repository cloning
- Docker container setup

This plan leverages the existing background task pattern in the codebase to ensure consistency and minimal disruption.

---

## Current Bottlenecks
1. **Thread Description Generation**: The `description()` method in `thread.py` generates a description by invoking the chat model on filtered messages, which can be slow for long conversations.
2. **Git Clone Setup**: The `clone_repo()` method in `domain_manager.py` clones repositories, which can introduce delays, especially for large repos.
3. **Docker Container Setup**: The `get_sandbox_backend()` method in `sandbox_manager.py` sets up Docker containers, which can also cause delays.

---

## Proposed Solution
### 1. Modify Thread Creation to Load Instantly
- **Action**: Create a minimal thread with only the request and a placeholder description.
- **Implementation**: 
  - In `web.py`, modify the `new_thread_with_message` method to create a thread with a placeholder description (e.g., "Loading...").
  - Defer the generation of the full description, Git clone, and Docker setup to background tasks.
  - Use the existing background task pattern in the codebase (e.g., `_process_message` or other async operations).

### 2. Update Thread Loading to Show Only the Request
- **Action**: Modify the `render_thread` method to display only the request in the web view initially.
- **Implementation**: 
  - In `web.py`, update `render_thread` to render only the request message if no description is available.
  - Add a loading indicator or placeholder for the description and other setup steps.

### 3. Implement Background Tasks for Setup
- **Action**: Use the existing background task pattern to handle the following operations:
  - Generate the thread description.
  - Clone the repository.
  - Set up the Docker container.
- **Implementation**: 
  - Use the existing `_process_message` pattern or similar async operations.
  - Update the thread state to reflect progress (e.g., "Description generated," "Git clone complete").
  - Refresh the web view dynamically as background tasks complete.

### 4. Update the Thread Class to Support Background Tasks
- **Action**: Modify the `Thread` class to support background task tracking.
- **Implementation**: 
  - Add a `background_tasks` attribute to track ongoing tasks.
  - Add methods to update task progress and results.

---

## Implementation Plan
### Step 1: Identify the Existing Background Task Pattern
- **Action**: Review the codebase to identify the existing background task pattern.
- **Files to Review**:
  - `/workspace/manage/web.py` (e.g., `_process_message`)
  - `/workspace/assist/thread.py`

### Step 2: Modify Thread Creation Logic
- **Action**: Update `new_thread_with_message` in `web.py` to create a minimal thread.
- **Implementation**:
  ```python
  def new_thread_with_message(self, request):
      tid = MANAGER.new()
      thread = MANAGER.get(tid)
      thread.add_message(request)  # Add only the request
      thread.description = "Loading..."  # Placeholder
      # Schedule background tasks
      schedule_description_generation(tid)
      schedule_git_clone(tid, request.json['repo_url'])
      schedule_docker_setup(tid)
  ```

### Step 3: Update Thread Rendering Logic
- **Action**: Modify `render_thread` in `web.py` to show only the request initially.
- **Implementation**:
  ```python
  def render_thread(thread):
      if not thread.description:
          return [{"role": "user", "content": "Loading..."}]
      return render_full_thread(thread)
  ```

### Step 4: Implement Background Tasks
- **Action**: Use the existing background task pattern to schedule and execute tasks.
- **Implementation**:
  - Example for description generation:
    ```python
    def schedule_description_generation(tid):
        # Use the existing background task pattern
        _process_message(tid, {"type": "generate_description"})
    ```

### Step 5: Update Frontend to Show Progress
- **Action**: Add a loading indicator for the description and setup steps.
- **Implementation**:
  - Use WebSockets or polling to update the web view dynamically.

---

## Files to Modify
1. `/workspace/manage/web.py`:
   - Update `new_thread_with_message`.
   - Update `render_thread`.
2. `/workspace/assist/thread.py`:
   - Add support for background tasks.

---

## Next Steps
1. **Implement Background Task Queue**:
   - Confirm the existing background task pattern in the codebase.
2. **Update Thread Creation Logic**:
   - Modify `new_thread_with_message` to create a minimal thread.
3. **Update Thread Rendering**:
   - Modify `render_thread` to show only the request initially.
4. **Add Progress Tracking**:
   - Update the frontend to show progress for background tasks.

---

## Testing Plan
1. **Write Tests**:
   - Test thread creation with minimal loading.
   - Test background task execution (description generation, Git clone, Docker setup).
   - Test web view rendering with only the request initially.
2. **Confirm Failing Tests**:
   - Ensure tests fail before implementing the changes to confirm the current behavior.

---

## References
- Existing background task pattern: `_process_message` in `/workspace/manage/web.py`
- Thread description generation: `description()` in `/workspace/assist/thread.py`
- Git clone setup: `clone_repo()` in `/workspace/assist/domain_manager.py`
- Docker container setup: `get_sandbox_backend()` in `/workspace/assist/sandbox_manager.py`