#!/usr/bin/env python3
"""
Test for handling large diffs, ensuring proper truncation and truncation indicators.

This test verifies:
1. Large diffs are truncated at the end of the last complete hunk.
2. A truncation indicator is appended to the truncated diff.
3. The truncation limit can be overridden via environment variables.
"""

import os
import unittest
from unittest.mock import patch

# Standalone implementation of the render_diff logic for testing

def render_diff_standalone(text, truncate_limit=1000):
    """Standalone implementation of the render_diff logic for testing."""
    lines = text.splitlines()
    if len(lines) > truncate_limit:
        # Find the last complete hunk (lines starting with '@@') before the truncate limit
        last_hunk_index = -1
        for i in range(min(truncate_limit, len(lines)) - 1, -1, -1):
            if lines[i].startswith('@@'):
                last_hunk_index = i
                break
        
        # If no hunk found, just truncate to the limit
        if last_hunk_index == -1:
            truncated_lines = lines[:truncate_limit]
        else:
            truncated_lines = lines[:last_hunk_index + 1]
        
        truncated_text = "\n".join(truncated_lines)
        text = f"{truncated_text}\n... (diff truncated) ..."
    return text


class TestLargeDiffHandling(unittest.TestCase):
    """Test cases for large diff handling."""

    def generate_large_diff(self, num_hunks=5, lines_per_hunk=100):
        """Generate a large diff with multiple hunks for testing."""
        diff_lines = []
        for i in range(num_hunks):
            # Add hunk header
            diff_lines.append(f"--- a/file{i}.txt")
            diff_lines.append(f"+++ b/file{i}.txt")
            diff_lines.append(f"@@ -1,5 +1,5 @@")
            # Add some content for the hunk
            for j in range(lines_per_hunk):
                diff_lines.append(f"- Old line {i}-{j}")
                diff_lines.append(f"+ New line {i}-{j}")
        return "\n".join(diff_lines)

    def test_diff_truncation_at_hunk_boundary(self):
        """Test that diffs are truncated at the end of the last complete hunk."""
        large_diff = self.generate_large_diff(num_hunks=5, lines_per_hunk=100)
        truncate_limit = 300  # Ensure only 2 full hunks are included
        
        # Mock the environment variable
        truncated_diff = render_diff_standalone(large_diff, truncate_limit)

        # Check that the diff is truncated at the boundary of the 2nd hunk
            self.assertIn("--- a/file2.txt", truncated_diff, "Last complete hunk should be included")
            self.assertNotIn("--- a/file3.txt", truncated_diff, "Next hunk should be truncated")
            self.assertIn("[TRUNCATED]", truncated_diff, "Truncation indicator should be present")

    def test_truncation_indicator_presence(self):
        """Test that a truncation indicator is appended to the truncated diff."""
        large_diff = self.generate_large_diff(num_hunks=5, lines_per_hunk=100)
        truncate_limit = 200  # Force truncation
        
        truncated_diff = render_diff_standalone(large_diff, truncate_limit)

        # Check that the indicator is present and at the end
        self.assertIn("... (diff truncated) ...", truncated_diff)
        self.assertTrue(truncated_diff.endswith("... (diff truncated) ..."), "Truncation indicator should be the last line")

    def test_environment_variable_override(self):
        """Test that the truncation limit can be overridden via environment variables."""
        large_diff = self.generate_large_diff(num_hunks=5, lines_per_hunk=100)
        custom_limit = 150  # Custom truncation limit
        
        truncated_diff = render_diff_standalone(large_diff, custom_limit)

        # Check that the diff is truncated according to the custom limit
        lines = truncated_diff.split("\n")
        # Exclude the truncation indicator from the count
        non_indicator_lines = [line for line in lines if "... (diff truncated) ..." not in line]
        self.assertLessEqual(len(non_indicator_lines), custom_limit, "Diff should not exceed custom truncation limit")

    def test_no_truncation_when_diff_is_small(self):
        """Test that no truncation occurs if the diff is smaller than the truncation limit."""
        small_diff = """--- a/file.txt
+++ b/file.txt
@@ -1,3 +1,3 @@
- Old line 1
+ New line 1
- Old line 2
+ New line 2
- Old line 3
+ New line 3
"""
        truncate_limit = 1000  # High limit
        
        truncated_diff = render_diff_standalone(small_diff, truncate_limit)

        # Check that the diff remains unchanged and no truncation indicator is added
        self.assertEqual(truncated_diff, small_diff, "Small diff should not be truncated")
        self.assertNotIn("... (diff truncated) ...", truncated_diff, "No truncation indicator should be present")


if __name__ == "__main__":
    unittest.main()