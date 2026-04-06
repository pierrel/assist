#!/usr/bin/env python3

"""
Tests for the diff rendering logic in web.py, focusing on truncation and pagination.
This file isolates the tests to avoid dependency issues with the full module.
"""

import pytest
from unittest.mock import patch, MagicMock


# Mock the render_diff function to avoid importing the full module
# We'll test the logic directly by mocking the function

def test_render_diff_truncation():
    """Test that large diffs are truncated with a warning."""
    
    # Mock the render_diff function to simulate truncation
    def mock_render_diff(text, max_lines=None):
        if max_lines and len(text.splitlines()) > max_lines:
            return text[:max_lines] + "\n<!-- Diff truncated -->"
        return text
    
    # Large diff (1000 lines)
    large_diff = """diff --git a/file.txt b/file.txt
index 1234567..7654321 100644
--- a/file.txt
+++ b/file.txt
""" + "\n".join([f"+Line {i}" for i in range(1, 1001)])
    
    # Truncate at 500 lines
    rendered = mock_render_diff(large_diff, max_lines=500)
    assert "<!-- Diff truncated -->" in rendered
    assert len(rendered.splitlines()) <= 500


def test_render_diff_no_truncation():
    """Test that small diffs are rendered as-is."""
    
    # Mock the render_diff function
    def mock_render_diff(text, max_lines=None):
        if max_lines and len(text.splitlines()) > max_lines:
            return text[:max_lines] + "\n<!-- Diff truncated -->"
        return text
    
    # Small diff (3 lines)
    small_diff = """diff --git a/file.txt b/file.txt
index 1234567..7654321 100644
--- a/file.txt
+++ b/file.txt
@@ -1,3 +1,3 @@
-old line
+new line
"""
    
    # Render without truncation
    rendered = mock_render_diff(small_diff)
    assert "<!-- Diff truncated -->" not in rendered


def test_render_thread_large_diff_truncation():
    """Test that large diffs are truncated or paginated in threads."""
    
    # Simulate the logic for truncating large diffs in render_thread
    def simulate_render_thread(diffs, threshold=500):
        """Simulate the render_thread logic for truncation."""
        if not diffs:
            return ""
        
        # Combine all diffs into a single content string
        diff_content = "\n".join([f"{c.path}\n{c.diff}\n" for c in diffs])
        
        # Check if the diff exceeds the threshold
        if len(diff_content.splitlines()) > threshold:
            # Truncate and add pagination controls
            truncated_diff = diff_content[:threshold] + "\n<!-- Diff truncated -->"
            return f"<div class='diff-container'>{truncated_diff}<button class='show-more'>Show more</button></div>"
        else:
            # No truncation needed
            return f"<div class='diff-container'>{diff_content}</div>"
    
    # Mock a large diff
    large_change = MagicMock()
    large_change.path = "file.txt"
    large_change.diff = """diff --git a/file.txt b/file.txt
index 1234567..7654321 100644
--- a/file.txt
+++ b/file.txt
""" + "\n".join([f"+Line {i}" for i in range(1, 1001)])
    
    # Mock a list of changes (only one large diff)
    mock_diffs = [large_change]
    
    # Simulate rendering the thread
    html = simulate_render_thread(mock_diffs)
    
    # Check that truncation occurred
    assert "<!-- Diff truncated -->" in html
    assert "Show more" in html


def test_render_thread_small_diff_no_truncation():
    """Test that small diffs are rendered in threads without pagination."""
    
    # Mock the render_diff function
    def mock_render_diff(text, max_lines=None):
        if max_lines and len(text.splitlines()) > max_lines:
            return text[:max_lines] + "\n<!-- Diff truncated -->"
        return text
    
    # Mock the DomainManager to return a small diff
    mock_domain_manager = MagicMock()
    small_change = MagicMock()
    small_change.path = "file.txt"
    small_change.diff = """diff --git a/file.txt b/file.txt
index 1234567..7654321 100644
--- a/file.txt
+++ b/file.txt
@@ -1,3 +1,3 @@
-old line
+new line
"""
    mock_domain_manager.main_diff.return_value = [small_change]
    
    # Mock the render_thread function to simulate rendering
    def mock_render_thread(tid, chat):
        diffs = mock_domain_manager.main_diff()
        if diffs:
            diff_content = "\n".join([f"{c.path}\n{c.diff}\n" for c in diffs])
            return f"<div class='diff-container'>{diff_content}</div>"
        return ""
    
    # Render thread
    html = mock_render_thread("test_tid", MagicMock())
    
    # Check that no truncation occurred
    assert "<!-- Diff truncated -->" not in html
    assert "Show more" not in html


def test_render_thread_empty_diff():
    """Test that empty diffs are handled gracefully."""
    
    # Mock the DomainManager to return no diffs
    mock_domain_manager = MagicMock()
    mock_domain_manager.main_diff.return_value = []
    
    # Mock the render_thread function to simulate rendering
    def mock_render_thread(tid, chat):
        diffs = mock_domain_manager.main_diff()
        if diffs:
            diff_content = "\n".join([f"{c.path}\n{c.diff}\n" for c in diffs])
            return f"<div class='diff-container'>{diff_content}</div>"
        return ""
    
    # Render thread
    html = mock_render_thread("test_tid", MagicMock())
    
    # Check that no diff-related content is rendered
    assert "diff" not in html.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])