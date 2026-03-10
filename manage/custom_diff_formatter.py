from pygments.formatters import HtmlFormatter
from pygments.styles import Style
from pygments.token import Token

# Define a custom style for diff lines
class CustomDiffStyle(Style):
    styles = {
        Token.Diff.Add: "background-color: #d4edda;",  # Light green for added lines
        Token.Diff.Delete: "background-color: #f8d7da;",  # Light red for deleted lines
    }

# Create a custom HtmlFormatter with the custom style
class CustomHtmlFormatter(HtmlFormatter):
    def __init__(self, **options):
        options['style'] = CustomDiffStyle
        super().__init__(**options)

# Example usage:
# formatter = CustomHtmlFormatter(nowrap=False)
# diff_html = highlight(diff_text, DiffLexer(), formatter)