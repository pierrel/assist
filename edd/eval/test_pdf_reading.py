"""Behavior evals for the ``pdf`` skill + ``read_pdf`` tool.

These run the general agent inside a Docker sandbox.  The skill's
contract is "orient → narrow → read" — this suite walks the small
model through each mode and one anti-test.

Each test gets a fresh workspace with a generated PDF fixture copied
in.  The sandbox image must contain ``poppler`` (added in the same PR
as ``read_pdf``) — without it, ``pdftotext`` doesn't exist inside the
container and every test would fail with the same shell-not-found
error.

Eval cadence per CLAUDE.md and saved feedback:
- 3× post-impl baseline.
- 5× post-reviewer.
- 10× stability before PR.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

import unittest
from unittest import TestCase

from langchain_core.messages import AIMessage

from assist.agent import AgentHarness, create_agent
from assist.model_manager import select_chat_model
from assist.sandbox_manager import SandboxManager


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FIXTURE_DIR = os.path.join(REPO, "tests", "fixtures", "pdf")


def _cleanup_workspace(path: str) -> None:
    """Mirror of test_calculate_skill.py's helper — delete root-owned files."""
    try:
        subprocess.run(
            ['docker', 'run', '--rm', '-v', f'{path}:/cleanup',
             'alpine', 'sh', '-c',
             'chmod -R 777 /cleanup 2>/dev/null; rm -rf /cleanup/*'],
            check=False, timeout=60,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
    shutil.rmtree(path, ignore_errors=True)


class TestPdfReading(TestCase):
    """Evals for the pdf skill.

    Real Docker sandbox required — read_pdf shells out to pdftotext
    inside the container.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = select_chat_model(0.1)

    def setUp(self):
        self.workspace = tempfile.mkdtemp(prefix="pdf_reading_eval_")
        self.sandbox = SandboxManager.get_sandbox_backend(self.workspace)
        if self.sandbox is None:
            self.skipTest(
                "Docker sandbox unavailable — is Docker running and "
                "assist-sandbox built (with poppler)?"
            )

    def tearDown(self):
        SandboxManager.cleanup(self.workspace)
        _cleanup_workspace(self.workspace)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _copy_fixture(self, name: str, dest_name: str | None = None) -> str:
        """Copy ``tests/fixtures/pdf/<name>`` into the workspace.

        Returns the workspace-relative filename so tests can reference
        it the way the agent will (no host path leaks into the prompt).
        """
        dest = dest_name or name
        src_path = os.path.join(FIXTURE_DIR, name)
        if not os.path.exists(src_path):
            raise unittest.SkipTest(
                f"Fixture {name!r} not found — regenerate with "
                f"`.venv/bin/python tests/fixtures/pdf/generate.py`."
            )
        shutil.copy(src_path, os.path.join(self.workspace, dest))
        return dest

    def _create_agent(self):
        return AgentHarness(create_agent(
            self.model,
            self.workspace,
            sandbox_backend=self.sandbox,
        ))

    def _read_pdf_calls(self, agent) -> list[dict]:
        """Args dict for every read_pdf tool call across the conversation."""
        calls = []
        for m in agent.all_messages():
            if not isinstance(m, AIMessage) or not m.tool_calls:
                continue
            for tc in m.tool_calls:
                if tc.get("name") == "read_pdf":
                    calls.append(tc.get("args") or {})
        return calls

    def _skill_was_loaded(self, agent, skill_name: str) -> bool:
        """Mirrors ``test_calculate_skill._skill_was_loaded``."""
        path_needle = f"/skills/{skill_name}/"
        for m in agent.all_messages():
            if not isinstance(m, AIMessage) or not m.tool_calls:
                continue
            for tc in m.tool_calls:
                args = tc.get("args") or {}
                if (tc.get("name") == "load_skill"
                        and args.get("name") == skill_name):
                    return True
                for v in args.values():
                    if isinstance(v, str) and path_needle in v:
                        return True
        return False

    # ------------------------------------------------------------------
    # Mode 1: orient
    # ------------------------------------------------------------------

    def test_orient_then_answer_page_count(self):
        """Orient mode — 'how many pages' is the canonical orient question."""
        self._copy_fixture("sample.pdf")
        agent = self._create_agent()
        res = agent.message("How many pages is sample.pdf?")

        # Skill loaded, read_pdf called with no extra args.
        self.assertTrue(
            self._skill_was_loaded(agent, "pdf"),
            "Agent did not load the pdf skill on a PDF question.",
        )
        calls = self._read_pdf_calls(agent)
        self.assertTrue(
            any("search" not in c and "pages" not in c for c in calls),
            f"Expected at least one orient-mode read_pdf call (no "
            f"search=, no pages=).  Calls: {calls}",
        )

        # Answer mentions "5".  Tolerate phrasing.
        self.assertTrue(
            re.search(r"\b5\b", res),
            f"Response should mention 5 pages.  Got: {res[:300]}",
        )

    # ------------------------------------------------------------------
    # Mode 2: search
    # ------------------------------------------------------------------

    def test_search_finds_specific_term(self):
        """Find mode — keyword question should drive a search call."""
        self._copy_fixture("sample.pdf")
        agent = self._create_agent()
        res = agent.message(
            "What does sample.pdf say about dosage?"
        )

        self.assertTrue(
            self._skill_was_loaded(agent, "pdf"),
            "Agent did not load the pdf skill on a PDF keyword question.",
        )
        calls = self._read_pdf_calls(agent)
        self.assertTrue(
            any(c.get("search") for c in calls),
            f"Expected at least one read_pdf call with a search= arg.  "
            f"Calls: {calls}",
        )

        # Answer should reference dosage info — page 3 of sample.pdf
        # has "Adult dosage is 5 mg per kilogram" / "Maximum dosage
        # must not exceed 500 mg per day".  We just check that the
        # response mentions a dosage figure.
        self.assertTrue(
            re.search(r"\b(5 mg|500 mg|2\.5 mg)\b", res, re.IGNORECASE),
            f"Response should cite a specific dosage from page 3.  "
            f"Got: {res[:400]}",
        )

    # ------------------------------------------------------------------
    # Mode 3: page-range read
    # ------------------------------------------------------------------

    def test_page_range_targeted_read(self):
        """Read mode — explicit page reference should drive a pages= call."""
        self._copy_fixture("sample.pdf")
        agent = self._create_agent()
        res = agent.message("What does page 2 of sample.pdf cover?")

        self.assertTrue(
            self._skill_was_loaded(agent, "pdf"),
            "Agent did not load the pdf skill on a PDF page-range question.",
        )
        calls = self._read_pdf_calls(agent)
        self.assertTrue(
            any(c.get("pages") for c in calls),
            f"Expected at least one read_pdf call with pages= arg.  "
            f"Calls: {calls}",
        )

        # Page 2 covers patient screening / allergies.
        self.assertTrue(
            re.search(r"(screen|allerg|contraindicat)", res, re.IGNORECASE),
            f"Response should mention page-2 content (screening / "
            f"allergies / contraindications).  Got: {res[:400]}",
        )

    # ------------------------------------------------------------------
    # Anti-test: don't dump the full PDF
    # ------------------------------------------------------------------

    def test_does_not_dump_full_pdf(self):
        """Anti-test — open question on a 60-page PDF must not trigger
        a giant pages= range or full-doc read.

        Predicate per the design doc: fail iff any read_pdf call has a
        pages= covering >20 pages, or matches 1-{total}.  Search and
        small ranges are fine.  big.pdf has a unique token "bluefin"
        on page 43 — the agent doesn't need to know that, but if it
        searches for relevant keywords it'll find what it needs."""
        self._copy_fixture("big.pdf")
        agent = self._create_agent()
        res = agent.message(
            "Give me a one-paragraph overview of big.pdf."
        )

        # Skill loaded.
        self.assertTrue(
            self._skill_was_loaded(agent, "pdf"),
            "Agent did not load the pdf skill on a PDF overview question.",
        )

        calls = self._read_pdf_calls(agent)
        self.assertTrue(calls, "Agent never called read_pdf.")

        # Each pages= must be either a single page or a range of <=20
        # pages.  No 1-{total} reads.
        for c in calls:
            pages = c.get("pages")
            if not pages:
                continue  # orient or search — fine.
            # Parse "N" or "N-M".
            if "-" in pages:
                a, _, b = pages.partition("-")
                try:
                    span = int(b) - int(a) + 1
                except ValueError:
                    continue  # malformed — separate failure mode.
            else:
                span = 1
            self.assertLessEqual(
                span, 20,
                f"Agent dumped {span} pages with pages={pages!r}.  The "
                f"skill says read no more than ~20 pages at a time.",
            )

        # Response must be non-empty (the overview is what was asked).
        self.assertGreater(len(res.strip()), 30)

    # ------------------------------------------------------------------
    # Edge case: no extractable text
    # ------------------------------------------------------------------

    def test_handles_empty_extract_gracefully(self):
        """When pdftotext returns no text, the tool says so explicitly.

        Strictly tests the *tool's* error message reaches the model
        intact — we mock by giving the model a 0-byte PDF placeholder
        which pdftotext refuses to parse.  The expected behaviour is
        that the agent surfaces an honest "couldn't read it" answer
        rather than hallucinating content.
        """
        # Create a fake "PDF" with the magic bytes but no body — the
        # magic-byte check passes inside the sandbox (host check is
        # skipped when a sandbox is bound), pdftotext fails on the
        # malformed body.
        path = os.path.join(self.workspace, "broken.pdf")
        with open(path, "wb") as f:
            f.write(b"%PDF-1.4\n%%EOF\n")

        agent = self._create_agent()
        res = agent.message("What is in broken.pdf?")

        # Agent should have at least attempted read_pdf.
        calls = self._read_pdf_calls(agent)
        self.assertTrue(
            calls,
            "Agent should have tried read_pdf on the user's PDF question.",
        )

        # Answer shouldn't claim specific content.  Soft check:
        # response contains a hedge word.
        hedges = ("couldn't", "could not", "unable", "no text", "empty",
                  "broken", "corrupt", "error", "failed")
        self.assertTrue(
            any(h in res.lower() for h in hedges),
            f"Response should hedge / surface the failure rather than "
            f"hallucinating content.  Got: {res[:400]}",
        )
