"""Explore a specific website and download a file — the runaway this pins.

Live 2026-07-21 (thread …527c2a44): with egress granted, the agent tried to
find the Fellow Espress manual PDF by crawling the Shopify site with raw
`curl` in the sandbox — following asset URL after asset URL
(`/cdn/shop/t/…/assets/lit-element.js`, …) until it hit the 500-step
recursion limit. `curl` returns raw asset-soup HTML with no dedup guard; the
`read_url` reread-breaker (and clean extraction) never applied because the
MAIN agent HAD no `read_url` (it was on the research subagent only); this
change adds it, guarded, plus the explore-website skill.

This eval reproduces that shape against a FULLY MOCKED fictional site (no
real network — good citizens): the sandbox `curl` is stubbed to serve raw
asset-heavy HTML, and `read_url`'s host-side fetch is mocked to serve the
same site. The goal test asserts a BOUNDED download (find the file, fetch it
once, never crawl assets); it fails on the pre-fix agent (curl-only, no
skill) and passes once the agent has `read_url` + the explore-website skill.
"""
import re
import tempfile
from unittest import TestCase, mock

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol

from assist.agent import create_agent, AgentHarness
from assist.checkpoint_rollback import invoke_with_rollback
from assist.model_manager import select_assistant_model

from .utils import (create_filesystem, prompt_rewrite_web_main_spec,
                    skill_was_loaded, stub_research_subagent)

# --- the fictional site (clearly not real; .example is reserved) --------------
_BASE = "https://www.fathomcoffee.example"
_PDF = "/cdn/shop/files/Fathom_Tide_Manual_v3.pdf"

# Shopify-style asset soup — the crawl trap. The real thread followed dozens
# of these dead-end JS chunks.
_ASSETS = "".join(
    f'<script src="/cdn/shop/t/1284/assets/{n}.js"></script>'
    for n in ("lit-element", "custom-element", "property.CoOl", "directive-helpers",
              "preload-helper", "BaseElement", "cart-drawer", "predictive-search"))
_HEAD = f'<html><head>{_ASSETS}</head><body>'
_FOOT = ('<footer><a href="/pages/warranty">Warranty</a>'
         '<a href="/collections/kettles">Shop kettles</a></footer></body></html>')

# raw HTML the SANDBOX `curl` returns (asset soup + real hrefs). The manual is
# NOT on the landing page — it's two navigation hops deep, like the real site.
_RAW = {
    f"{_BASE}/": _HEAD + (
        '<main><h1>Fathom Coffee</h1>'
        '<p>Precision pour-over gear.</p>'
        '<nav><a href="/pages/support">Support &amp; manuals</a>'
        '<a href="/collections/kettles">Kettles</a>'
        '<a href="/pages/warranty">Warranty</a></nav></main>') + _FOOT,
    f"{_BASE}/pages/support": _HEAD + (
        '<main><h1>Support</h1>'
        '<p>Guides and manuals for your Fathom gear.</p>'
        '<ul><li><a href="/pages/tide-manual">Tide Kettle manual</a></li>'
        '<li><a href="/pages/faq">FAQ</a></li></ul></main>') + _FOOT,
    f"{_BASE}/pages/tide-manual": _HEAD + (
        '<main><h1>Tide Kettle manual</h1>'
        '<p>Download the full user manual (PDF):</p>'
        f'<a href="{_PDF}">Fathom Tide Kettle — User Manual (v3, PDF)</a>'
        '</main>') + _FOOT,
    f"{_BASE}/pages/faq": _HEAD + '<main><h1>FAQ</h1><p>Common questions.</p></main>' + _FOOT,
    f"{_BASE}/collections/kettles": _HEAD + '<main><h1>Kettles</h1></main>' + _FOOT,
    f"{_BASE}/pages/warranty": _HEAD + '<main><h1>Warranty</h1></main>' + _FOOT,
}
# asset URLs return JS junk with no links — the crawl dead-ends
_ASSET_JS = "console.log('shopify chunk');// bundled module, no content\n"


def _url_in(command: str) -> str | None:
    m = re.search(r"https?://[^\s'\"]+", command)
    return m.group(0).rstrip("'\"") if m else None


class _MockSiteBackend(FilesystemBackend, SandboxBackendProtocol):
    """Sandbox stub that serves the fictional site over `curl` and records
    what the agent fetched, so the test can see a crawl."""

    def __init__(self, root_dir):
        super().__init__(root_dir=root_dir, virtual_mode=False)
        self.work_dir = root_dir
        self.page_fetches: list[str] = []     # curl GETs of a page (the crawl signal)
        self.asset_hits: list[str] = []       # curl GETs of /assets/ dead-ends
        self.downloads: list[str] = []        # curl -o of a file

    def execute(self, command, timeout=None):
        url = _url_in(command)
        if url and "curl" in command:
            is_download = bool(re.search(r"-[oO]\b", command))
            if url.startswith(_BASE) and url.endswith(".pdf"):
                if is_download:
                    self.downloads.append(url)
                    return ExecuteResponse(
                        output="  % Total    % Received\n100 2411k  100 2411k\n",
                        exit_code=0)
                return ExecuteResponse(output="%PDF-1.5 (binary)", exit_code=0)
            if "/assets/" in url:
                self.asset_hits.append(url)
                return ExecuteResponse(output=_ASSET_JS, exit_code=0)
            page = _RAW.get(url.split("#")[0].rstrip("/") if url != f"{_BASE}/"
                            else url) or _RAW.get(url)
            if page is not None:
                self.page_fetches.append(url)
                return ExecuteResponse(output=page, exit_code=0)
            return ExecuteResponse(output="curl: (22) 404 Not Found", exit_code=22)
        if command.strip().startswith("pdfinfo"):
            if self.downloads:
                return ExecuteResponse(output="Pages: 24\nFile size: 2411234 bytes",
                                       exit_code=0)
            return ExecuteResponse(output="pdfinfo: file not found", exit_code=1)
        return ExecuteResponse(output="", exit_code=0)


class _FakeResp:
    def __init__(self, text): self.text, self.status_code = text, 200
    def raise_for_status(self): pass


def _mock_requests_get(url, **kwargs):
    """read_url's host-side fetch → the same site (page HTML, so extraction +
    the link surfacing see real hrefs). Assets/PDF aren't read_url targets."""
    body = _RAW.get(url.split("#")[0]) or _RAW.get(url.rstrip("/"))
    return _FakeResp(body if body is not None else "<html><body>Not found</body></html>")


class TestWebsiteDownload(TestCase):
    def setUp(self):
        self.model = select_assistant_model(0.1)

    def _run(self, prompt):
        root = tempfile.mkdtemp()
        create_filesystem(root, {})
        backend = _MockSiteBackend(root)
        with mock.patch("assist.tools.requests.get", _mock_requests_get), \
             stub_research_subagent():
            agent = create_agent(self.model, root, sandbox_backend=backend,
                                 spec=prompt_rewrite_web_main_spec())
            harness = AgentHarness(agent, thread_id="eval-web-dl")
            # 220 = headroom for a legit ~5-tool-call navigation through the
            # deep middleware stack (each model turn is many supersteps), still
            # far below a real crawl-to-death (prod's backstop is 500). The
            # crawl is caught by the metric assertions below (asset_hits /
            # over-fetch / no download), not by this limit — a regression that
            # crawls trips those long before 220.
            invoke_with_rollback(
                agent, {"messages": [{"role": "user", "content": prompt}]},
                {"configurable": {"thread_id": "eval-web-dl"},
                 "recursion_limit": 220})
            return harness, backend

    def test_download_converges_without_crawling(self):
        """Find the manual (2 nav hops deep on an asset-heavy site) and
        download it — WITHOUT crawling asset URLs to death."""
        agent, backend = self._run(
            "Download the Fathom Tide Kettle user manual PDF from "
            f"{_BASE}/ and save it to /workspace. Tell me its page count.")
        self.assertTrue(backend.downloads,
                        f"never downloaded the manual; page fetches="
                        f"{len(backend.page_fetches)}, asset hits="
                        f"{len(backend.asset_hits)}")
        self.assertTrue(any(_PDF in d for d in backend.downloads),
                        f"downloaded the wrong URL: {backend.downloads}")
        self.assertEqual(backend.asset_hits, [],
                         f"crawled asset dead-ends (the runaway): "
                         f"{backend.asset_hits[:8]}")
        self.assertLessEqual(len(backend.page_fetches), 3,
                             f"over-fetched pages via curl (should navigate "
                             f"with read_url): {backend.page_fetches}")
        self.assertTrue(skill_was_loaded(agent, "explore-website"),
                        "did not load the explore-website skill")
