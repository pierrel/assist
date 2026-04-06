#!/usr/bin/env python3

"""
Tests for the web.py module, focusing on diff rendering and handling large diffs.
"""

import pytest
from unittest.mock import patch, MagicMock
from manage.web import render_diff, render_thread
from assist.domain_manager import Change
from assist.thread import Thread


class MockDomainManager:
    """Mock DomainManager for testing."""
    
    def __init__(self, changes=None):
        self.changes = changes or []
    
    def main_diff(self):
        return self.changes


class MockThread:
    """Mock Thread for testing."""
    
    def __init__(self, messages=None):
        self.messages = messages or []
    
    def append(self, msg):
        self.messages.append(msg)


@pytest.fixture
def mock_domain_manager():
    """Fixture for a mock DomainManager."""
    return MockDomainManager()


@pytest.fixture
def mock_thread():
    """Fixture for a mock Thread."""
    return MockThread()


@pytest.fixture
def small_diff():
    """Fixture for a small diff."""
    return """diff --git a/file.txt b/file.txt
index 1234567..7654321 100644
--- a/file.txt
+++ b/file.txt
@@ -1,3 +1,3 @@
-old line
+new line
"""


@pytest.fixture
def large_diff():
    """Fixture for a large diff."""
    return """diff --git a/file.txt b/file.txt
index 1234567..7654321 100644
--- a/file.txt
+++ b/file.txt
@@ -1,1000 +1,1000 @@
+Line 1
+Line 2
+... (1000 lines of diff)
"""


@pytest.fixture
def large_change():
    """Fixture for a large Change object."""
    return Change(path="file.txt", diff="""diff --git a/file.txt b/file.txt
index 1234567..7654321 100644
--- a/file.txt
+++ b/file.txt
@@ -1,1000 +1,1000 @@
+Line 1
+Line 2
+... (1000 lines of diff)
""")


def test_render_diff_small(small_diff):
    """Test that small diffs are rendered as-is."""
    rendered = render_diff(small_diff)
    assert rendered is not None
    assert "<!-- Diff truncated -->" not in rendered


def test_render_diff_large_truncation(large_diff):
    """Test that large diffs are truncated with a warning."""
    rendered = render_diff(large_diff, max_lines=500)
    assert "<!-- Diff truncated -->" in rendered
    assert len(rendered.splitlines()) <= 500


def test_render_thread_small_diff(mock_domain_manager, mock_thread, small_diff):
    """Test that small diffs are rendered in threads without pagination."""
    # Mock a small diff
    small_change = Change(path="file.txt", diff=small_diff)
    mock_domain_manager.changes = [small_change]
    
    # Mock the render_diff function to avoid actual rendering
    with patch("manage.web.render_diff") as mock_render_diff:
        mock_render_diff.return_value = "<div>Rendered diff</div>"
        
        # Render thread
        html = render_thread("test_tid", mock_thread)
        
        # Check that render_diff was called without max_lines
        mock_render_diff.assert_called_once_with(small_diff, max_lines=None)
        assert "Show more" not in html


def test_render_thread_large_diff(mock_domain_manager, mock_thread, large_change):
    """Test that large diffs are truncated or paginated in threads."""
    # Mock a large diff
    mock_domain_manager.changes = [large_change]
    
    # Mock the render_diff function to capture truncation
    with patch("manage.web.render_diff") as mock_render_diff:
        mock_render_diff.return_value = "<div>Truncated diff</div>"
        
        # Render thread
        html = render_thread("test_tid", mock_thread)
        
        # Check that render_diff was called with max_lines
        mock_render_diff.assert_called_once_with(large_change.diff, max_lines=500)
        assert "Show more" in html or "<!-- Diff truncated -->" in html


def test_render_thread_empty_diff(mock_domain_manager, mock_thread):
    """Test that empty diffs are handled gracefully."""
    # Mock an empty diff
    mock_domain_manager.changes = []
    
    # Render thread
    html = render_thread("test_tid", mock_thread)
    
    # Check that no diff-related content is rendered
    assert "diff" not in html.lower()


def test_render_thread_ui_controls(mock_domain_manager, mock_thread, large_change):
    """Test that UI controls (e.g., 'Show more') are rendered for large diffs."""
    # Mock a large diff
    mock_domain_manager.changes = [large_change]
    
    # Render thread
    html = render_thread("test_tid", mock_thread)
    
    # Check for pagination controls
    assert "Show more" in html
    assert "diff-container" in html
    assert "diff-toggle" in html


if __name__ == "__main__":
    pytest.main([__file__, "-v"])