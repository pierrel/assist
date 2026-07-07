"""Eval: can the research agent reach content BURIED DEEP in a page (beyond the
read_url cap)?  This is the gate for the read_url-to-file change: read_url should
write the full extracted content to a file + return a preview, and the agent should
GREP the file for specifics.

BASELINE (current main, read_url capped at 24000 chars): the fact ~32000 chars into
the page is TRUNCATED away — the agent can't see it → should FAIL.  WITH the change:
the full page is offloaded to a file and the agent greps it → should PASS.

Rate-limit-free: `requests.get` is mocked to return a canned large page (no SearXNG,
no network) — the "new eval shape" (mock the network/search boundary); read_url's REAL
extraction + cap/offload logic runs on it.
"""
import os
import tempfile
from unittest import TestCase
from unittest.mock import patch

from assist.agent import AgentHarness, create_research_agent
from assist.model_manager import select_assistant_model
from assist.sandbox_manager import SandboxManager
from edd.eval.utils import build_thread_repo, cleanup_workspace

os.environ.setdefault("ASSIST_MODEL_URL", "http://127.0.0.1:8000/v1")

# A canned article about a FICTIONAL venue (so the model CANNOT answer from training
# data — it must actually read the page).  The DEEP FACT sits ~32000 chars into the
# extracted text — PAST read_url's current 24000-char cap — so the baseline (capped)
# can't see it; only writing the full page to a file + grepping it reaches the fact.
_CAPACITY = "88,204"   # a made-up, unguessable number for a made-up venue
_DEEP_FACT = f"The Vantalux Coliseum's exact seating capacity is {_CAPACITY} seats."
_TOP = ("<article><h1>Vantalux Coliseum</h1>"
        + ("<p>The Vantalux Coliseum is a multipurpose arena in the fictional city of "
           "Brindlemark. It hosts the regional Gravball championships and is noted for "
           "its retractable canopy and the famous echoing north stand. </p>") * 200)
_HTML = _TOP + f"<h2>Facts</h2><p>{_DEEP_FACT} It opened in 2011.</p></article>"


class _Resp:
    text = _HTML
    def raise_for_status(self): pass


def _mock_get(url, **kw):
    return _Resp()


def _mock_search(query, max_results=5):
    """Search the web via the self-hosted SearXNG metasearch. Returns JSON results."""
    import json
    return json.dumps([{"url": "https://example.com/vantalux-coliseum",
                        "title": "Vantalux Coliseum", "content": "arena in Brindlemark"}])


class TestReadUrlDeepContent(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="read_url_deep_")
        self.addCleanup(cleanup_workspace, self.tmp)
        self.workspace = build_thread_repo(self.tmp, "assist/read-url-thread")
        self.sandbox = SandboxManager.get_sandbox_backend(self.workspace)
        if self.sandbox is None:
            self.skipTest("Docker sandbox unavailable")
        self.addCleanup(SandboxManager.cleanup, self.workspace)

    def test_agent_reaches_deep_fact(self):
        """The answer (88,204) lives ~32000 chars into the page — past the 24k cap.
        Baseline (main) can't see it → fails; with read_url→file + grep → passes."""
        with patch("assist.tools.requests.get", _mock_get), \
             patch("assist.tools.search_internet", _mock_search), \
             patch("assist.agent.search_internet", _mock_search):
            agent = AgentHarness(create_research_agent(self.model, self.workspace))
            out = agent.message(
                "Research the Vantalux Coliseum and tell me its EXACT seating capacity "
                "(the precise number of seats). Read the source page for the exact figure.")
        ans = str(out)
        found = _CAPACITY in ans or _CAPACITY.replace(",", "") in ans.replace(",", "")
        self.assertTrue(found,
                        f"agent did not reach the deep fact (capacity {_CAPACITY}) — the 24k "
                        f"cap truncates it, so it needs read_url->file + grep. Answer:\n{ans[:600]}")
