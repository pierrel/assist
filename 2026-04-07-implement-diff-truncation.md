# Implementation Plan: Diff Truncation for Large Diffs

## Reason for Change
Large diffs in the web view can cause performance issues and clutter the UI. This change introduces:
- A truncation method in `DomainManager` to limit diff size to 1000 lines.
- A warning message for truncated diffs.
- Toggle functionality in the web view to expand/collapse truncated diffs.
- Backward compatibility for small diffs.

## Proposed Tests
### Test Files
- **`/workspace/tests/assist/test_domain_manager.py`**: Add tests for the `truncate_diff` method and `git_diff_main` to ensure truncation works as expected.
- **`/workspace/tests/manage/test_web.py`**: Add tests for `render_diff` to verify that truncated diffs are rendered with warnings and toggle buttons.

### Test Scenarios
1. **Normal Case**: Small diffs (< 1000 lines) should render as before.
2. **Truncated Case**: Large diffs (> 1000 lines) should be truncated with a warning and toggle button.
3. **Toggle Functionality**: Ensure the toggle button works to expand/collapse truncated diffs.
4. **Edge Case**: Empty diffs or diffs with exactly 1000 lines should not be truncated.

## Proposed Changes
### Files to Modify
1. **`/workspace/assist/domain_manager.py`**:
   - Add a `truncate_diff` method to truncate diffs exceeding 1000 lines.
   - Update the `git_diff_main` method to apply truncation logic.
   - Include a warning message in the truncated diff.

2. **`/workspace/manage/web.py`**:
   - Modify the `render_diff` function to handle truncated diffs and warnings.
   - Update the HTML template for diff rendering to include toggle buttons for expanding/collapsing truncated diffs.

### Implementation Details
- **Truncation Logic**:
  - If a diff exceeds 1000 lines, truncate it and prepend a warning message.
  - The warning message should indicate that the diff has been truncated.

- **Toggle Functionality**:
  - Add a toggle button in the HTML template for truncated diffs.
  - Use JavaScript to handle expanding/collapsing the diff content.

## Expected Outcomes
- Large diffs (> 1000 lines) will be truncated with a warning and toggle button.
- Small diffs (< 1000 lines) will render as before.
- The toggle button will allow users to expand/collapse truncated diffs.
- All existing tests should pass, and new tests should verify the truncation and toggle functionality.

## Risks / Considerations
- **Backward Compatibility**: Ensure that small diffs are not affected by the truncation logic.
- **Performance**: Truncation should not introduce significant performance overhead.
- **User Experience**: The warning message and toggle button should be clear and intuitive.
- **Edge Cases**: Handle empty diffs and diffs with exactly 1000 lines gracefully.

## Implementation Steps
1. Implement the `truncate_diff` method in `DomainManager`.
2. Update `git_diff_main` to apply truncation logic.
3. Modify `render_diff` in `web.py` to handle truncated diffs and warnings.
4. Update the HTML template to include toggle buttons.
5. Write tests for the truncation logic and toggle functionality.
6. Run all tests to ensure backward compatibility and correctness.