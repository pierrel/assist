# tests/assist/test_domain_manager.py
"""
Tests for the DomainManager truncation logic.
"""

import pytest
from assist.domain_manager import Change, truncate_diff


@pytest.mark.parametrize(
    "diff_text, expected_truncated_diff, expected_warning",
    [
        # Case 1: Diff with fewer than 1000 lines (no truncation)
        ("diff --git a/file.txt b/file.txt\n\n--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-hello\n+world\n",
         "diff --git a/file.txt b/file.txt\n\n--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-hello\n+world\n",
         None),
        
        # Case 2: Diff with exactly 1000 lines (no truncation)
        ("\n".join(["line " + str(i) for i in range(1000)]),
         "\n".join(["line " + str(i) for i in range(1000)]),
         None),
        
        # Case 3: Diff with more than 1000 lines (truncation)
        ("\n".join(["line " + str(i) for i in range(1500)]),
         "\n".join(["line " + str(i) for i in range(1000)]),
         "Diff truncated due to length"),
        
        # Case 4: Empty diff (no truncation)
        ("", "", None),
    ],
)
def test_truncate_diff(diff_text, expected_truncated_diff, expected_warning):
    """Test the truncate_diff function."""
    truncated_diff, warning = truncate_diff(diff_text, max_lines=1000)
    assert truncated_diff == expected_truncated_diff
    assert warning == expected_warning


@pytest.mark.parametrize(
    "diff_text, expected_warning",
    [
        # Case 1: Diff with warning (truncated)
        ("\n".join(["line " + str(i) for i in range(1500)]),
         "Diff truncated due to length"),
        
        # Case 2: Diff without warning (not truncated)
        ("\n".join(["line " + str(i) for i in range(500)]),
         None),
    ],
)
def test_change_with_warning(diff_text, expected_warning):
    """Test the Change class with warning messages."""
    truncated_diff, warning = truncate_diff(diff_text, max_lines=1000)
    change = Change(path="test.txt", diff=truncated_diff, warning=warning)
    assert change.warning == expected_warning


if __name__ == "__main__":
    pytest.main([__file__, "-v"])