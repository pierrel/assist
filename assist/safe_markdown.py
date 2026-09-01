"""Small allow-list Markdown renderer for untrusted assistant text."""
from __future__ import annotations

import html
from html.parser import HTMLParser
from urllib.parse import urlparse

import markdown


_ALLOWED_TAGS = frozenset({
    "a", "blockquote", "br", "code", "del", "em", "h1", "h2", "h3", "h4",
    "h5", "h6", "hr", "li", "ol", "p", "pre", "strong", "table", "tbody",
    "td", "th", "thead", "tr", "ul",
})
_VOID_TAGS = frozenset({"br", "hr"})


def _safe_href(value: str) -> str | None:
    parsed = urlparse(value)
    if parsed.scheme.lower() not in {"", "http", "https", "mailto"}:
        return None
    if parsed.scheme == "" and value.startswith("//"):
        return None
    return value


class _Sanitizer(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in _ALLOWED_TAGS:
            return
        attributes = ""
        if tag == "a":
            href = next((value for key, value in attrs if key.lower() == "href" and value), None)
            if href is not None:
                safe = _safe_href(href)
                if safe is not None:
                    attributes = f' href="{html.escape(safe, quote=True)}"'
        self.parts.append(f"<{tag}{attributes}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag in _ALLOWED_TAGS and tag not in _VOID_TAGS:
            self.parts.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        # Text nodes do not use quote delimiters; leaving apostrophes literal
        # preserves existing Markdown output while still escaping markup.
        self.parts.append(html.escape(data, quote=False))


def render_markdown(text: str, *, extensions: list[str] | tuple[str, ...] = ()) -> str:
    """Render Markdown while allowing no raw HTML or executable URL scheme."""
    parser = _Sanitizer()
    parser.feed(markdown.markdown(text, extensions=list(extensions)))
    parser.close()
    return "".join(parser.parts)
