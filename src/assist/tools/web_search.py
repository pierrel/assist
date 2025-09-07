from __future__ import annotations

import html
import re
from typing import List

import requests
from langchain_core.tools import tool


@tool
def site_search(domain: str, query: str) -> List[str]:
    """Search within ``domain`` for pages matching ``query``.

    Uses DuckDuckGo to retrieve up to 5 result URLs and titles.
    """
    params = {"q": f"site:{domain} {query}"}
    resp = requests.get("https://duckduckgo.com/html/", params=params, timeout=10)
    results: List[str] = []
    pattern = re.compile(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
    for m in pattern.finditer(resp.text):
        url = html.unescape(m.group(1))
        title = re.sub(r"<.*?>", "", m.group(2))
        results.append(f"{title} - {url}")
        if len(results) >= 5:
            break
    return results


@tool
def page_search(url: str, query: str) -> List[str]:
    """Return snippets from ``url`` containing ``query``.

    Fetches the page and returns up to 5 text snippets around each match.
    """
    resp = requests.get(url, timeout=10)
    text = re.sub(r"<[^>]+>", " ", resp.text)
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    snippets: List[str] = []
    for match in pattern.finditer(text):
        start = max(0, match.start() - 40)
        end = min(len(text), match.end() + 40)
        snippet = re.sub(r"\s+", " ", text[start:end])
        snippets.append(snippet.strip())
        if len(snippets) >= 5:
            break
    return snippets if snippets else ["No matches"]


__all__ = ["site_search", "page_search"]
