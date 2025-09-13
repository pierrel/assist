from __future__ import annotations

import html
import re
from typing import List

import requests
from langchain_core.tools import tool
from langchain_tavily import TavilySearch


class SearchWeb(TavilySearch):
    """one_line: Searches the public web; returns up to `max_results` entries with title, url and content.

    when_to_use:
    - Need information from across the web.
    - Verify facts with multiple sources.
    - Explore recent news or updates.
    when_not_to_use:
    - Restrict results to a single site → use search_site.
    - Already have the URL to inspect → use search_page.
    - Lacking internet access or Tavily credentials.
    args_schema:
    - query (str): Search terms, e.g. "latest AI research".
    preconditions_permissions:
    - Requires `TAVILY_API_KEY` environment variable.
    side_effects:
    - Sends HTTP requests; idempotent: true; retry_safe: true.
    cost_latency: "~200-1000ms; API billing may apply"
    pagination_cursors:
    - input_cursor: none
    - next_cursor: none
    errors:
    - network_error: Request failed; check key and connectivity.
    returns:
    - results (list[dict]): Items with "title", "url", "content".
    - brief_summary (str): Short result count summary.
    examples:
    - input: {"query": "python dataclasses"}
      output: {"results": [...], "brief_summary": "10 results"}
    version: "1.0"
    owner: "assist"
    """

    name: str = "search_web"

    def _run(self, query: str, config: dict | None = None, **kwargs):  # type: ignore[override]
        data = super()._run(query, **kwargs)
        results = data.get("results", [])
        data["brief_summary"] = (
            f"{len(results)} result" if len(results) == 1 else f"{len(results)} results"
        )
        return data

    async def _arun(self, query: str, config: dict | None = None, **kwargs):  # type: ignore[override]
        data = await super()._arun(query, **kwargs)
        results = data.get("results", [])
        data["brief_summary"] = (
            f"{len(results)} result" if len(results) == 1 else f"{len(results)} results"
        )
        return data


@tool
def search_site(domain: str, query: str) -> dict:
    """one_line: Searches a domain for pages matching a query; returns up to five "title - URL" results.

    when_to_use:
    - Need web pages from a specific site.
    - Verify information hosted on a known domain.
    - Restrict search scope for precision.
    when_not_to_use:
    - Require results across many domains.
    - Domain is unknown or inaccessible.
    - Operating without internet access.
    args_schema:
    - domain (str): Target domain, e.g. "example.com".
    - query (str): Search terms, e.g. "privacy policy".
    preconditions_permissions:
    - Domain must be publicly reachable.
    side_effects:
    - Sends HTTP requests; idempotent: true; retry_safe: true.
    cost_latency: "~200-1000ms; free"
    pagination_cursors:
    - input_cursor: none
    - next_cursor: none
    errors:
    - network_error: Request failed; check connectivity and retry.
    returns:
    - results (list[str]): "Title - URL" entries.
    - brief_summary (str): Short result count summary.
    examples:
    - input: {"domain": "example.com", "query": "about"}
      output: {"results": ["About Us - https://example.com/about"],
               "brief_summary": "1 result"}
    - input: {"domain": "wikipedia.org", "query": "AI"}
      output: {"results": ["Artificial intelligence - https://wikipedia.org/..."],
               "brief_summary": "5 results"}
    version: "1.0"
    owner: "assist"
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
    summary = f"{len(results)} result" if len(results) == 1 else f"{len(results)} results"
    return {"results": results, "brief_summary": summary}


@tool
def search_page(url: str, query: str) -> dict:
    """one_line: Finds up to five snippets from a page containing a query string.

    when_to_use:
    - Extract context from a single known URL.
    - Verify that a page mentions specific terms.
    - Retrieve short quotes from a document.
    when_not_to_use:
    - Need to search across multiple pages or domains.
    - URL content is inaccessible.
    - No internet access is available.
    args_schema:
    - url (str): Web page URL, e.g. "https://example.com".
    - query (str): Text to locate, e.g. "license".
    preconditions_permissions:
    - URL must be publicly reachable.
    side_effects:
    - Sends HTTP requests; idempotent: true; retry_safe: true.
    cost_latency: "~200-1000ms; free"
    pagination_cursors:
    - input_cursor: none
    - next_cursor: none
    errors:
    - network_error: Request failed; check connectivity and retry.
    returns:
    - snippets (list[str]): Text around each match or ["No matches"].
    - brief_summary (str): Short result count summary.
    examples:
    - input: {"url": "https://example.com", "query": "contact"}
      output: {"snippets": ["Call us ..."], "brief_summary": "1 match"}
    - input: {"url": "https://example.com", "query": "foo"}
      output: {"snippets": ["No matches"], "brief_summary": "0 matches"}
    version: "1.0"
    owner: "assist"
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
    summary = f"{len(snippets)} match" if len(snippets) == 1 else f"{len(snippets)} matches"
    if not snippets:
        snippets = ["No matches"]
    return {"snippets": snippets, "brief_summary": summary}


__all__ = ["SearchWeb", "search_site", "search_page"]
