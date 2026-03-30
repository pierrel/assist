"""
Test cases for the `render_diff` function in `manage/web.py`.

This module tests the truncation logic for large diffs and ensures that
the function works correctly for small and empty diffs.
"""

import os
from manage.web import render_diff


def test_render_diff_with_small_diff():
    """Test that small diffs are rendered without truncation."""
    small_diff = """diff --git a/file.txt b/file.txt
index 1234567..7654321 100644
--- a/file.txt
+++ b/file.txt
@@ -1 +1 @@
-old content\n+new content\n"""
    result = render_diff(small_diff)
    assert "old content" in result
    assert "new content" in result
    assert "... (diff truncated) ..." not in result


def test_render_diff_with_empty_diff():
    """Test that empty diffs are rendered without errors."""
    empty_diff = ""
    result = render_diff(empty_diff)
    assert result == "<div class=\"highlight\">\n</div>"


def test_render_diff_with_large_diff():
    """Test that large diffs are truncated."""
    # Create a large diff with more lines than the default truncate limit
    # Create a large diff with more lines than the default truncate limit
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


def test_render_diff_with_truncate_limit_override():
    """Test that truncation limit can be overridden via environment variable."""
    # Create a large diff with more lines than the default truncate limit
    large_diff_lines = [
        "diff --git a/large_file.txt b/large_file.txt",
        "index 1234567..7654321 100644",
        "--- a/large_file.txt",
        "+++ b/large_file.txt",
    ]
    for i in range(1, 2000):
        large_diff_lines.append(f"@@ -{i}+{i + 1} @@")
        large_diff_lines.append(f"-old line {i}")
        large_diff_lines.append(f"+new line {i}")
    large_diff = "\n".join(large_diff_lines)
    
    # Set a truncate limit that is larger than the diff
    os.environ["DIFF_TRUNCATE_LIMIT"] = "3000"
    try:
        result = render_diff(large_diff)
        assert "... (diff truncated) ..." not in result
    finally:
        # Clean up environment variable
        del os.environ["DIFF_TRUNCATE_LIMIT"]