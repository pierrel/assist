import unittest
import os

# Standalone implementation of render_diff for testing

def render_diff_standalone(text: str, start_line: int = None, end_line: int = None, truncate_limit: int = 1000) -> str:
    """Standalone version of render_diff for testing.
    
    Args:
        text: The diff text.
        start_line: Starting line for pagination (optional).
        end_line: Ending line for pagination (optional).
        truncate_limit: Truncation limit (default: 1000).
    """
    lines = text.splitlines()
    
    # Handle pagination if start_line and end_line are provided
    if start_line is not None and end_line is not None:
        start_line = max(0, start_line)
        end_line = min(len(lines), end_line)
        paginated_lines = lines[start_line:end_line]
        paginated_text = "\n".join(paginated_lines)
        text = f"{paginated_text}\n... (lines {start_line + 1}-{end_line} of {len(lines)}) ..."
    else:
        # Truncate the diff if it exceeds the limit
        if len(lines) > truncate_limit:
            last_hunk_index = -1
            for i in range(min(truncate_limit, len(lines)) - 1, -1, -1):
                if lines[i].startswith('@@'):
                    last_hunk_index = i
                    break
            
            if last_hunk_index == -1:
                truncated_lines = lines[:truncate_limit]
            else:
                truncated_lines = lines[:last_hunk_index + 1]
            
            truncated_text = "\n".join(truncated_lines)
            text = f"{truncated_text}\n... (diff truncated) ..."
    
    return text

class TestPagination(unittest.TestCase):
    
    def setUp(self):
        # Generate a large diff for testing
        self.large_diff = """--- a/file1.txt
+++ b/file1.txt
@@ -1,5 +1,5 @@
- Old line 1
+ New line 1
- Old line 2
+ New line 2
- Old line 3
+ New line 3
- Old line 4
+ New line 4
- Old line 5
+ New line 5
--- a/file2.txt
+++ b/file2.txt
@@ -1,3 +1,3 @@
- Old line A
+ New line A
- Old line B
+ New line B
- Old line C
+ New line C
--- a/file3.txt
+++ b/file3.txt
@@ -1,10 +1,10 @@
- Old line X
+ New line X
- Old line Y
+ New line Y
- Old line Z
+ New line Z
- Old line 10
+ New line 10
- Old line 11
+ New line 11
- Old line 12
+ New line 12
- Old line 13
+ New line 13
- Old line 14
+ New line 14
"""

    def test_pagination_within_bounds(self):
        """Test that pagination works correctly within the bounds of the diff."""
        start_line = 5
        end_line = 10
        
        paginated_diff = render_diff_standalone(self.large_diff, start_line, end_line)
        
        # Check that the paginated diff contains only the specified lines
        lines = paginated_diff.split("\n")
        # Exclude the pagination indicator
        non_indicator_lines = [line for line in lines if "... (lines" not in line]
        
        self.assertEqual(len(non_indicator_lines), end_line - start_line)
        
        # Check that the pagination indicator is present
        self.assertIn("... (lines", paginated_diff)

    def test_pagination_out_of_bounds(self):
        """Test that pagination works correctly when start_line or end_line is out of bounds."""
        start_line = 0
        end_line = 100  # Beyond the actual number of lines
        
        paginated_diff = render_diff_standalone(self.large_diff, start_line, end_line)
        
        # Check that the paginated diff does not exceed the actual number of lines
        lines = paginated_diff.split("\n")
        non_indicator_lines = [line for line in lines if "... (lines" not in line]
        
        self.assertLessEqual(len(non_indicator_lines), len(self.large_diff.split("\n")))
        
        # Check that the pagination indicator is present
        self.assertIn("... (lines", paginated_diff)

    def test_no_pagination(self):
        """Test that truncation occurs if start_line and end_line are not provided."""
        truncate_limit = 10  # Force truncation
        
        truncated_diff = render_diff_standalone(self.large_diff, truncate_limit=truncate_limit)
        
        # Check that the diff is truncated
        self.assertIn("... (diff truncated) ...", truncated_diff)


if __name__ == "__main__":
    unittest.main()