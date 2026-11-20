"""Frozen search and page responses for later real-agent research trials.

Host search and page responses are substituted, and browser navigation is
declared unavailable. The agent, its specialists, Runs and workspace remain
real. A fixture miss is reported as such, never as a live site's absence or
failure.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

import requests

from edd.eval.research_live_trace import append_private


class _Response:
    def __init__(self, *, payload=None, body="", status=200):
        self._payload = payload
        self.text = body
        self.status_code = status
        self.headers = {"Content-Type": "text/html; charset=utf-8"}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"frozen source returned HTTP {self.status_code}")

    def close(self):
        pass


class FrozenRetrieval:
    def __init__(self, artifact: Path, search_url: str, fixture_name: str):
        manifest = json.loads(Path(__file__).with_name("research_live_sources.json").read_text())
        self.pages = {row["url"]: row for row in manifest["fixtures"][fixture_name]}
        self.citation_tiers = {
            urlsplit(url).path.removeprefix("/tier/") for url in self.pages
            if urlsplit(url).path.startswith("/tier/")
            and "/" not in urlsplit(url).path.removeprefix("/tier/")
        }
        self.fixture_name = fixture_name
        self.search_url = search_url.rstrip("/") + "/search"
        self.artifact = artifact

    def _record(self, kind: str, **fields):
        append_private(self.artifact / "fixture.jsonl", kind,
                       fixture=self.fixture_name, **fields)

    def _search(self, query: str) -> _Response:
        lowered = query.casefold()
        matches = []
        for row in self.pages.values():
            score = sum(tag.casefold() in lowered for tag in row["tags"])
            if score:
                matches.append((score, row))
        matches.sort(key=lambda item: (-item[0], item[1]["url"]))
        rows = [{"title": row["title"], "url": row["url"],
                 "content": row["snippet"]} for _, row in matches[:8]]
        self._record("frozen_search", query=query, returned_urls=[row["url"] for row in rows])
        return _Response(payload={"results": rows, "unresponsive_engines": []},
                         body=json.dumps({"results": rows}))

    def _page(self, url: str) -> _Response:
        row = self.pages.get(url)
        if row is None and self.fixture_name.startswith("helio"):
            row = self._citation_page(url)
        if row is None:
            self._record("frozen_url_miss", url=url)
            raise requests.ConnectionError("URL has no recorded response in frozen evaluation corpus")
        self._record("frozen_page", url=url, status=row["status"])
        return _Response(body=row["body"], status=row["status"])

    def _citation_page(self, url: str) -> dict | None:
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or parsed.netloc != "helio-specs.test"
                or parsed.query or parsed.fragment):
            return None
        parts = parsed.path.strip("/").split("/")
        if (parsed.path != "/" + "/".join(parts)
                or len(parts) not in {3, 4, 5} or parts[0] != "cite"
                or parts[1] not in self.citation_tiers
                or parts[2] not in {str(i) for i in range(12)}
                or any(part not in {str(i) for i in range(8)} for part in parts[3:])):
            return None
        more = " ".join(f"https://helio-specs.test{parsed.path}/{index}"
                        for index in range(8)) if len(parts) < 5 else ""
        return {"body": f"Background reference for {parts[1]}; no tier hardware specification. "
                        f"Further citations: {more}", "status": 200}

    def get(self, url, **kwargs):
        target = str(url)
        if target == self.search_url:
            return self._search(str((kwargs.get("params") or {}).get("q", "")))
        return self._page(target)

    def post(self, url, **kwargs):
        self._record("frozen_post_miss", url=str(url))
        raise requests.ConnectionError("POST has no recorded response in frozen evaluation corpus")

    def pdf(self, url):
        response = self._page(url)
        response.raise_for_status()
        return response.text


def install_frozen_retrieval(artifact: Path, search_url: str,
                             fixture_name: str) -> FrozenRetrieval:
    """Keep host retrieval and browser navigation inside the frozen corpus."""
    from assist import tools
    from assist.browser.manager import BrowserSession, BrowserUnavailable

    fixture = FrozenRetrieval(artifact, search_url, fixture_name)
    tools.requests.get = fixture.get
    tools.requests.post = fixture.post
    tools._read_pdf_url = fixture.pdf

    def unavailable_browser(_session, operation: str, **args):
        fixture._record("frozen_browser_unavailable", operation=operation,
                        url=args.get("url"))
        raise BrowserUnavailable("browser pages are outside this frozen evaluation corpus")

    BrowserSession.command = unavailable_browser
    return fixture
