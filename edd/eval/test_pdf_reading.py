"""Behavior evals for the ``pdf`` skill.

The skill teaches the model to use ``execute`` with ``pdftotext`` and
``pdfinfo`` for PDF questions — not a dedicated tool.  These tests
walk the small model through orient / find / read, plus an anti-test
("don't dump the whole PDF") and a graceful-degradation case.

Each test gets a fresh workspace with a generated PDF fixture copied
in.  The sandbox image must contain ``poppler`` (added to
``Dockerfile.sandbox``) — without it, ``pdftotext`` doesn't exist
inside the container and every test would fail with the same
shell-not-found error.

Eval cadence per CLAUDE.md and saved feedback:
- 3× post-impl baseline.
- 5× post-reviewer.
- 10× stability before PR (skip if wall-clock cost outweighs gain).
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

    Real Docker sandbox required — the model runs ``pdftotext`` /
    ``pdfinfo`` via ``execute`` inside the container.
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

    def _execute_commands(self, agent) -> list[str]:
        """Command strings from every ``execute`` tool call."""
        commands = []
        for m in agent.all_messages():
            if not isinstance(m, AIMessage) or not m.tool_calls:
                continue
            for tc in m.tool_calls:
                if tc.get("name") == "execute":
                    args = tc.get("args") or {}
                    cmd = args.get("command", "")
                    if cmd:
                        commands.append(cmd)
        return commands

    def _used_read_file_on_pdf(self, agent) -> bool:
        """True if the model called ``read_file`` on a .pdf path.

        The skill's hardest-line rule.  ``read_file`` on a PDF returns
        a multimodal content block the local model can't consume; the
        next API call 400s.  This method lets tests assert the rule
        was respected.
        """
        for m in agent.all_messages():
            if not isinstance(m, AIMessage) or not m.tool_calls:
                continue
            for tc in m.tool_calls:
                if tc.get("name") not in ("read_file", "read"):
                    continue
                args = tc.get("args") or {}
                p = args.get("file_path") or args.get("path") or ""
                if isinstance(p, str) and p.lower().endswith(".pdf"):
                    return True
        return False

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
        """Orient mode — 'how many pages' is the canonical orient question.

        Expect: skill loaded, model ran ``pdfinfo`` (the cheap orient
        command), answer mentions the correct page count.
        """
        self._copy_fixture("sample.pdf")
        agent = self._create_agent()
        res = agent.message("How many pages is sample.pdf?")

        self.assertTrue(
            self._skill_was_loaded(agent, "pdf"),
            "Agent did not load the pdf skill on a PDF question.",
        )
        self.assertFalse(
            self._used_read_file_on_pdf(agent),
            "Agent read_file'd a PDF — that returns multimodal blocks "
            "the local model can't parse.  Skill says use execute.",
        )
        cmds = self._execute_commands(agent)
        self.assertTrue(
            any("pdfinfo" in c for c in cmds),
            f"Expected at least one execute() call running pdfinfo "
            f"to get the page count.  Commands: {cmds}",
        )

        self.assertTrue(
            re.search(r"\b5\b", res),
            f"Response should mention 5 pages.  Got: {res[:300]}",
        )

    # ------------------------------------------------------------------
    # Mode 2: search
    # ------------------------------------------------------------------

    def test_search_finds_specific_term(self):
        """Find mode — keyword question should drive a pdftotext|grep pipeline."""
        self._copy_fixture("sample.pdf")
        agent = self._create_agent()
        res = agent.message("What does sample.pdf say about dosage?")

        self.assertTrue(
            self._skill_was_loaded(agent, "pdf"),
            "Agent did not load the pdf skill on a PDF keyword question.",
        )
        self.assertFalse(
            self._used_read_file_on_pdf(agent),
            "Agent read_file'd a PDF — should have used execute.",
        )
        cmds = self._execute_commands(agent)
        # Either a piped pdftotext|grep, or a separate pdftotext + grep
        # within the same command (or sequential), all count.
        self.assertTrue(
            any("pdftotext" in c for c in cmds),
            f"Expected at least one execute() call running pdftotext.  "
            f"Commands: {cmds}",
        )

        self.assertTrue(
            re.search(r"\b(5 mg|500 mg|2\.5 mg)\b", res, re.IGNORECASE),
            f"Response should cite a specific dosage from page 3.  "
            f"Got: {res[:400]}",
        )

    # ------------------------------------------------------------------
    # Mode 3: page-range read
    # ------------------------------------------------------------------

    def test_page_range_targeted_read(self):
        """Read mode — explicit page reference should drive pdftotext -f/-l."""
        self._copy_fixture("sample.pdf")
        agent = self._create_agent()
        res = agent.message("What does page 2 of sample.pdf cover?")

        self.assertTrue(
            self._skill_was_loaded(agent, "pdf"),
            "Agent did not load the pdf skill on a PDF page-range question.",
        )
        self.assertFalse(
            self._used_read_file_on_pdf(agent),
            "Agent read_file'd a PDF — should have used execute.",
        )
        cmds = self._execute_commands(agent)
        # Should see a pdftotext call.  Bonus credit if it scoped to
        # page 2 specifically (-f 2 -l 2 or similar) — but a full
        # extract followed by reading the relevant chunk is also a
        # legit way to answer this on a 5-page doc.
        self.assertTrue(
            any("pdftotext" in c for c in cmds),
            f"Expected pdftotext in execute() commands.  Got: {cmds}",
        )

        self.assertTrue(
            re.search(r"(screen|allerg|contraindicat)", res, re.IGNORECASE),
            f"Response should mention page-2 content (screening / "
            f"allergies / contraindications).  Got: {res[:400]}",
        )

    # ------------------------------------------------------------------
    # Anti-test: don't dump a giant PDF
    # ------------------------------------------------------------------

    def test_does_not_dump_full_pdf(self):
        """Anti-test — open question on a 60-page PDF must not trigger
        a full-document pdftotext.

        Predicate: fail iff any pdftotext command lacks both ``-f`` and
        ``-l`` flags (i.e. extracts the whole document) when the input
        is the 60-page big.pdf.  Orient (page 1 only) and small ranges
        are fine.  Search via piped grep is fine even without -f/-l
        because grep filters before the model sees the output.
        """
        self._copy_fixture("big.pdf")
        agent = self._create_agent()
        res = agent.message("Give me a one-paragraph overview of big.pdf.")

        self.assertTrue(
            self._skill_was_loaded(agent, "pdf"),
            "Agent did not load the pdf skill on a PDF overview question.",
        )
        self.assertFalse(
            self._used_read_file_on_pdf(agent),
            "Agent read_file'd a PDF — should have used execute.",
        )

        cmds = self._execute_commands(agent)
        self.assertTrue(cmds, "Agent never called execute.")

        # Any pdftotext call without -f/-l AND without a downstream
        # grep is a full-document dump.
        for cmd in cmds:
            if "pdftotext" not in cmd:
                continue
            has_first_flag = "-f " in cmd or "-f" in cmd.split()
            has_last_flag = "-l " in cmd or "-l" in cmd.split()
            has_pipe_filter = "|" in cmd  # downstream grep / head / tail
            if not (has_first_flag or has_last_flag or has_pipe_filter):
                self.fail(
                    f"Agent ran a full-document pdftotext on a 60-page PDF: "
                    f"{cmd!r}.  Expected -f/-l scope or a pipe filter."
                )

        self.assertGreater(len(res.strip()), 30)

    # ------------------------------------------------------------------
    # Edge case: no extractable text
    # ------------------------------------------------------------------

    def test_handles_empty_extract_gracefully(self):
        """When pdftotext fails on a malformed PDF, the agent surfaces it."""
        # Magic bytes valid, body malformed.
        path = os.path.join(self.workspace, "broken.pdf")
        with open(path, "wb") as f:
            f.write(b"%PDF-1.4\n%%EOF\n")

        agent = self._create_agent()
        res = agent.message("What is in broken.pdf?")

        self.assertFalse(
            self._used_read_file_on_pdf(agent),
            "Agent read_file'd a PDF — should have used execute.",
        )
        cmds = self._execute_commands(agent)
        self.assertTrue(
            cmds,
            "Agent should have tried execute on the user's PDF question.",
        )

        hedges = ("couldn't", "could not", "unable", "no text", "empty",
                  "broken", "corrupt", "error", "failed", "malformed")
        self.assertTrue(
            any(h in res.lower() for h in hedges),
            f"Response should hedge / surface the failure rather than "
            f"hallucinating content.  Got: {res[:400]}",
        )
