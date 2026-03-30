"""
Standalone implementation of the diff truncation logic for testing.

This module implements the truncation logic of the `render_diff` function
without relying on the `manage.web` module or its dependencies.
"""

import os
from typing import List


def truncate_diff(diff: str, limit: int) -> str:
    """Truncate a diff if it exceeds the specified line limit.
    
    Args:
        diff: The diff content as a string.
        limit: The maximum number of lines to keep.
        
    Returns:
        The truncated diff or the original diff if no truncation is needed.
    """
    lines = diff.splitlines()
    if len(lines) <= limit:
        return diff
    
    # Truncate to the limit, ensuring we don't cut in the middle of a hunk
    truncated_lines = lines[:limit]
    
    # Find the last complete hunk (lines starting with '@@')
    last_hunk_index = -1
    for i in range(len(truncated_lines) - 1, -1, -1):
        if truncated_lines[i].startswith('@@'):
            last_hunk_index = i
            break
    
    # If no hunk found, just truncate to the limit
    if last_hunk_index == -1:
        truncated_lines = lines[:limit]
    else:
        truncated_lines = lines[:last_hunk_index + 1]
    
    truncated_diff = "\n".join(truncated_lines)
    return f"{truncated_diff}\n... (diff truncated) ..."


def render_diff(diff: str) -> str:
    """Render a diff with truncation logic.
    
    Args:
        diff: The diff content as a string.
        
    Returns:
        The rendered diff with truncation applied if necessary.
    """
    limit = int(os.environ.get("DIFF_TRUNCATE_LIMIT", 1000))
    return truncate_diff(diff, limit)


def test_render_diff_with_small_diff():
    """Test that small diffs are rendered without truncation."""
    small_diff = """diff --git a/file.txt b/file.txt
index 1234567..7654321 100644
--- a/file.txt
+++ b/file.txt
@@ -1 +1 @@
-old content
+new content
"""
    result = render_diff(small_diff)
    assert "old content" in result
    assert "new content" in result
    assert "... (diff truncated) ..." not in result


def test_render_diff_with_large_diff():
    """Test that large diffs are truncated."""
    # Create a large diff
    large_diff_lines = [
        "diff --git a/large_file.txt b/large_file.txt",
        "index 1234567..7654321 100644",
        "--- a/large_file.txt",
        "+++ b/large_file.txt",
    ]
    for i in range(1, 1500):
        large_diff_lines.append(f"@@ -{i}+{i + 1} @@")
        large_diff_lines.append(f"-old line {i}")
        large_diff_lines.append(f"+new line {i}")
    large_diff = "\n".join(large_diff_lines)
    
    # Set a small truncate limit for testing
    os.environ["DIFF_TRUNCATE_LIMIT"] = "50"
    try:
        result = render_diff(large_diff)
        assert "... (diff truncated) ..." in result
        assert "old line 1" in result
        assert "new line 1" in result
        assert "old line 1499" not in result
        assert "new line 1499" not in result
    finally:
        # Clean up environment variable
        del os.environ["DIFF_TRUNCATE_LIMIT"]


def test_render_diff_with_empty_diff():
    """Test that empty diffs are rendered without truncation."""
    empty_diff = ""
    result = render_diff(empty_diff)
    assert "... (diff truncated) ..." not in result


def test_render_diff_with_no_truncate_limit():
    """Test that diffs are not truncated if no limit is set."""
    # Create a large diff
    large_diff_lines = [
        "diff --git a/large_file.txt b/large_file.txt",
        "index 1234567..7654321 100644",
        "--- a/large_file.txt",
        "+++ b/large_file.txt",
    ]
    for i in range(1, 10):
        large_diff_lines.append(f"@@ -{i}+{i + 1} @@")
        large_diff_lines.append(f"-old line {i}")
        large_diff_lines.append(f"+new line {i}")
    large_diff = "\n".join(large_diff_lines)
    
    result = render_diff(large_diff)
    assert "... (diff truncated) ..." not in result
    assert "old line 1" in result
    assert "new line 1" in result
    assert "old line 9" in result
    assert "new line 9" in result