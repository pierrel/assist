# Plan for Handling Large Diffs in Web View

## Reason for Change
Large diffs (e.g., binary files, multi-line changes) are rendered as raw HTML without any size-based optimizations. This causes:
- Performance issues (slow load times or browser crashes).
- Degraded user experience due to overwhelming content.

The goal is to implement **pagination** and **truncation with warnings** to ensure large diffs are handled efficiently without breaking the existing architecture.

## Proposed Tests

### Test Scenarios
1. **Normal Diffs (Small)**
   - Verify that small diffs (e.g., < 100 lines) are rendered as-is without any changes.
   - **Test File**: `tests/manage/test_web.py` (new or existing).

2. **Large Diffs (Exceeding Threshold)**
   - Verify that diffs exceeding the threshold (e.g., 500 lines) are truncated or paginated.
   - **Test File**: `tests/manage/test_web.py`.

3. **Edge Cases**
   - Empty diffs: Ensure no errors are raised.
   - Diffs exactly at the threshold: Ensure they are handled gracefully.
   - Binary file diffs: Ensure special handling (e.g., warnings or truncation).
   - **Test File**: `tests/manage/test_web.py`.

4. **UI Controls**
   - Verify that pagination controls (e.g., "Show more" buttons) are rendered correctly.
   - **Test File**: `tests/manage/test_web.py` (or a separate UI test file).

5. **JavaScript Dynamic Loading**
   - Verify that the `showMoreDiff()` function works as expected.
   - **Test File**: `tests/manage/test_web_ui.js` (if applicable).

### Test Implementation
```python
# Example test for render_diff()
def test_render_diff_truncation():
    large_diff = """diff --git a/file.txt b/file.txt
index 1234567..7654321 100644
--- a/file.txt
+++ b/file.txt
@@ -1,1000 +1,1000 @@
+Line 1
+Line 2
+... (1000 lines of diff)
"""
    truncated_diff = render_diff(large_diff, max_lines=500)
    assert "<!-- Diff truncated -->" in truncated_diff
    assert len(truncated_diff.splitlines()) <= 500

def test_render_thread_pagination():
    # Mock a large diff
    large_diff = Change(path="file.txt", diff="""diff --git a/file.txt b/file.txt
index 1234567..7654321 100644
--- a/file.txt
+++ b/file.txt
@@ -1,1000 +1,1000 @@
+Line 1
+Line 2
+... (1000 lines of diff)
""")
    
    # Mock thread and domain manager
    thread = Thread()
    dm = DomainManager()
    dm.main_diff = lambda: [large_diff]
    
    # Render thread and check for pagination
    html = render_thread("test_tid", thread)
    assert "<!-- Diff truncated -->" in html or "Show more" in html

# Example test for UI controls
@patch("markdown.markdown")
def test_render_thread_ui_controls(markdown_mock):
    # Mock a large diff
    large_diff = Change(path="file.txt", diff="""diff --git a/file.txt b/file.txt
index 1234567..7654321 100644
--- a/file.txt
+++ b/file.txt
@@ -1,1000 +1,1000 @@
+Line 1
+Line 2
+... (1000 lines of diff)
""")
    
    # Mock thread and domain manager
    thread = Thread()
    dm = DomainManager()
    dm.main_diff = lambda: [large_diff]
    
    # Render thread
    html = render_thread("test_tid", thread)
    
    # Check for pagination controls
    assert "Show more" in html
    assert "diff-container" in html
    assert "diff-toggle" in html
```

## Proposed Changes

### Files to Modify
1. **`/workspace/manage/web.py`**
   - **Function**: `render_diff(text: str, max_lines: int = None) -> str`
     - Add logic to truncate diffs if they exceed `max_lines`.
     - Append a warning comment (e.g., `<!-- Diff truncated -->`).
   - **Function**: `render_thread(tid: str, chat: Thread) -> str`
     - Add logic to detect large diffs (e.g., by line count).
     - Call `render_diff()` with `max_lines` if the diff exceeds the threshold.
     - Add pagination controls (e.g., "Show more" buttons) to the collapsible container.

2. **JavaScript (if applicable)**
   - **Function**: `showMoreDiff(diffId)`
     - Implement dynamic loading of additional diff chunks or reveal hidden content.

### High-Level Approach
1. **Size Check in `render_thread()`**
   - Measure the size of the diff content (e.g., `len(diff_content.splitlines())`).
   - Define a threshold (e.g., 500 lines) for triggering pagination or truncation.

2. **Truncation in `render_diff()`**
   - Update the function to handle truncated diffs:
     ```python
     def render_diff(text: str, max_lines: int = None) -> str:
         if max_lines and len(text.splitlines()) > max_lines:
             text = text[:max_lines] + "\n<!-- Diff truncated -->"
         return highlight(text, DiffLexer(), HtmlFormatter(nowrap=False))
     ```

3. **Pagination Controls in `render_thread()`**
   - Extend the collapsible container to include "Show more" buttons:
     ```html
     <div class="diff-container">
         <div class="diff-toggle">
             <button onclick="toggleDiff('{diff_id}')">Show diff</button>
             <button class="show-more" onclick="showMoreDiff('{diff_id}')">Show more</button>
         </div>
         <div id="{diff_id}" class="diff-content" style="display: none;">
             {diff_content[:200]} <!-- Initial chunk -->
         </div>
     </div>
     ```

4. **JavaScript for Dynamic Loading**
   - Implement `showMoreDiff()` to load additional diff chunks or reveal hidden content:
     ```javascript
     function showMoreDiff(diffId) {
         const diffContent = document.getElementById(diffId);
         const currentText = diffContent.textContent;
         const moreText = currentText.replace(/<!-- Diff truncated -->.*/, '');
         diffContent.textContent = moreText;
     }
     ```

## Expected Outcomes

### User-Visible Behavior
- Large diffs will be truncated or paginated, improving load times and preventing browser crashes.
- Users will see a "Show more" button to load additional diff content dynamically.
- Warnings (e.g., "Diff truncated") will indicate when content is hidden.

### Test Results
- All existing tests for small diffs will continue to pass.
- New tests for large diffs will pass, verifying truncation and pagination.
- UI tests will pass, confirming that pagination controls are rendered correctly.

### Performance
- Reduced initial load size for large diffs.
- Improved responsiveness in the web view.

## Risks / Considerations

### Edge Cases
- **Binary File Diffs**: May need special handling (e.g., warnings or truncation).
- **Diffs Exactly at Threshold**: Should be handled gracefully without truncation.
- **Empty Diffs**: Should not cause errors.

### Backward Compatibility
- No breaking changes to the existing API or rendering logic.
- Small diffs will continue to render as-is.

### User Experience
- Ensure that pagination controls are intuitive and accessible.
- Provide clear warnings when diffs are truncated.

### Testing
- Test with a variety of diff sizes and types (e.g., text files, binary files).
- Verify that the JavaScript dynamic loading works as expected.

## Next Steps
1. **Implement the changes** in `/workspace/manage/web.py` and any necessary JavaScript files.
2. **Write tests** for the new functionality.
3. **Run all tests** to ensure backward compatibility and correctness.
4. **Get a critique** of the changes to address any bugs or style issues.

Please review this plan and let me know if you'd like to proceed with implementation.