from pygments.formatters import HtmlFormatter
from pygments.styles import Style
from pygments.token import Token

# Define a custom style for diff lines
class CustomDiffStyle(Style):
    styles = {
        Token.Diff.Add: "background-color: #d4edda;",  # Light green for added lines
        Token.Diff.Delete: "background-color: #f8d7da;",  # Light red for deleted lines
    }

# Custom HtmlFormatter for diff output with explicit styling for added and deleted lines.
# - Lines starting with `+` (added) will have a light green background (#d4edda).
# - Lines starting with `-` (deleted) will have a light red background (#f8d7da).
class CustomHtmlFormatter(HtmlFormatter):
    def __init__(self, **options):
        # Apply the custom style to enforce background colors for diff tokens
        options['style'] = CustomDiffStyle
        super().__init__(**options)

# Example usage:
# formatter = CustomHtmlFormatter(nowrap=False)
# diff_html = highlight(diff_text, DiffLexer(), formatter)