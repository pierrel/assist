# Plan to Address Large Diffs Failing to Load in Web View

## Overview
The issue involves very large diffs in the web view failing to load due to excessive information. This plan outlines a solution to handle large diffs by truncating or summarizing them, both in the backend and frontend.

---

## Root Cause Analysis
1. **Backend (`DomainManager`)**:
   - The `git_diff_main` method generates diffs as raw strings and returns them as `Change` objects.
   - No logic exists to truncate or summarize large diffs, so they are passed as-is to the web view.

2. **Frontend (`manage/web.py`)**:
   - The `render_diff` function uses Pygments to highlight diffs in HTML.
   - Diffs are rendered directly in the HTML template without checks for size or truncation.
   - Large diffs overwhelm the page, causing performance issues.

---

## Proposed Solution
### 1. Backend: Truncate or Summarize Large Diffs
- Add a threshold (e.g., 1000 lines or 50KB) for diff size.
- If a diff exceeds this threshold, truncate it and add a summary or warning message.

**Example Logic:**
```python
def truncate_diff(diff: str, max_lines: int = 1000) -> str:
    lines = diff.splitlines()
    if len(lines) <= max_lines:
        return diff
    else:
        return f"Diff truncated (showing first {max_lines} lines):\n" + "\n".join(lines[:max_lines])
```

### 2. Frontend: Optimize Diff Rendering
- Modify the `render_diff` function to handle truncated diffs gracefully.
- Add a toggle button to show/hide the full diff (if available).

**Example HTML Template Update:**
```html
<div class="diff-container">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <button class="diff-toggle" onclick="toggleDiff('diff-1')">▶ Show diff</button>
    </div>
    <div id="diff-1" class="diff-content" style="display: none;">
        {diff_content}
    </div>
    {warning_message}
</div>
```

### 3. Configuration
- Add a configuration option (e.g., in `manage/web.py` or a settings file) to define the threshold for diff truncation.

**Example:**
```python
DIFF_TRUNCATE_THRESHOLD = 1000  # Lines
```

---

## Implementation Plan
1. **Update `DomainManager`:**
   - Add a method to truncate large diffs.
   - Modify `git_diff_main` to apply truncation if needed.

2. **Update `render_diff` in `manage/web.py`:**
   - Add logic to handle truncated diffs and warnings.
   - Update the HTML template to include toggle buttons and warnings.

3. **Test the Changes:**
   - Verify that large diffs are truncated and displayed with warnings.
   - Ensure that small diffs are rendered as before.

---

## Next Steps
- Implement the truncation logic in `DomainManager`.
- Update the `render_diff` function and HTML template in `manage/web.py`.
- Test the changes to ensure they resolve the issue.

---

## Files Involved
- `/workspace/assist/domain_manager.py` (Backend logic for diff generation)
- `/workspace/manage/web.py` (Frontend logic for diff rendering)

---

## Assumptions
- The threshold for truncation will be set to 1000 lines by default.
- The warning message will be displayed if a diff is truncated.
- The toggle button will allow users to expand/collapse the diff content.

---

## Risks and Mitigations
- **Risk:** Truncating diffs may hide important changes.
  **Mitigation:** Provide a warning message and allow users to expand the diff if needed.

- **Risk:** Changes to the frontend may break existing functionality.
  **Mitigation:** Test thoroughly to ensure compatibility with existing features.

---

## Open Questions
- Should the threshold for truncation be configurable via a settings file?
- Should users be able to adjust the threshold dynamically?

---

## References
- Backend: `DomainManager` class in `/workspace/assist/domain_manager.py`
- Frontend: `render_diff` function in `/workspace/manage/web.py`