"""
Test cases for the `render_thread` function in `manage/web.py`.

This module tests the integration of diff rendering logic within the
`render_thread` function and ensures that diffs are rendered correctly
when included in a thread.
"""

import os
from unittest.mock import MagicMock, patch
from manage.web import render_diff


def test_render_thread_with_diff():
    """Test that diffs are rendered correctly within a thread."""
    # Mock the Thread and DomainManager classes
    mock_chat = MagicMock()
    mock_chat.get_messages.return_value = [
        {"role": "user", "content": "Test message"},
        {"role": "diff", "content": "diff --git a/file.txt b/file.txt\nindex 1234567..7654321 100644\n--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old content\n+new content\n"}
    ]
    
    # Mock the DomainManager to return a diff
    mock_dm = MagicMock()
    mock_dm.main_diff.return_value = [
        MagicMock(path="file.txt", diff="diff --git a/file.txt b/file.txt\nindex 1234567..7654321 100644\n--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old content\n+new content\n")
    ]
    
    # Mock the _get_domain_manager function
    with patch("manage.web._get_domain_manager", return_value=mock_dm):
        # Mock the get_cached_description function
        with patch("manage.web.get_cached_description", return_value="Test Thread"):
            result = render_thread("test-tid", mock_chat)
            assert "old content" in result
            assert "new content" in result
            assert "diff-container" in result
            assert "Show diff" in result


def test_render_thread_with_large_diff():
    """Test that large diffs are truncated within a thread."""
    # Mock the Thread and DomainManager classes
    mock_chat = MagicMock()
    # Create a large diff for the chat message
    large_diff_chat_lines = [
        "diff --git a/large_file.txt b/large_file.txt",
        "index 1234567..7654321 100644",
        "--- a/large_file.txt",
        "+++ b/large_file.txt",
    ]
    for i in range(1, 1500):
        large_diff_chat_lines.append(f"@@ -{i}+{i + 1} @@")
        large_diff_chat_lines.append(f"-old line {i}")
        large_diff_chat_lines.append(f"+new line {i}")
    large_diff_chat = "\n".join(large_diff_chat_lines)
    
    mock_chat.get_messages.return_value = [
        {"role": "user", "content": "Test message"},
        {"role": "diff", "content": large_diff_chat},
    ]
    
    # Mock the DomainManager to return a large diff
    mock_dm = MagicMock()
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
    mock_dm.main_diff.return_value = [
        MagicMock(path="large_file.txt", diff=large_diff)
    ]
    
    # Mock the _get_domain_manager function
    with patch("manage.web._get_domain_manager", return_value=mock_dm):
        # Mock the get_cached_description function
        with patch("manage.web.get_cached_description", return_value="Test Thread"):
            # Set a small truncate limit for testing
            os.environ["DIFF_TRUNCATE_LIMIT"] = "50"
            try:
                # Mock the ThreadManager and Thread classes to avoid dependency issues
                with patch("manage.web.ThreadManager", autospec=True) as mock_thread_manager:
                    mock_thread = MagicMock()
                    mock_thread_manager.return_value.get.return_value = mock_thread
                    
                    result = render_thread("test-tid", mock_chat)
                    assert "... (diff truncated) ..." in result
                    assert "old line 1" in result
                    assert "new line 1" in result
                    assert "old line 1499" not in result
                    assert "new line 1499" not in result
            finally:
                # Clean up environment variable
                del os.environ["DIFF_TRUNCATE_LIMIT"]