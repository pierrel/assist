"""Unit tests for ``assist/tools.py`` — focused on ``read_pdf``.

The tool shells out to ``pdftotext`` (and ``pdfinfo`` for orient mode);
these tests stub the shell layer with a fake sandbox handle so they
run hermetically — no host-installed poppler required.  The behavior
evals in ``edd/eval/test_pdf_reading.py`` exercise the real pipeline
inside a Docker sandbox.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from assist import tools
from assist.tools import (
    _format_orient,
    _format_search,
    _gather_search_hits,
    _parse_page_range,
    _split_into_pages,
    read_pdf,
)


HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DIR = os.path.join(HERE, "fixtures", "pdf")
SAMPLE_PDF = os.path.join(FIXTURE_DIR, "sample.pdf")
BIG_PDF = os.path.join(FIXTURE_DIR, "big.pdf")


@dataclass
class _FakeResp:
    output: str
    exit_code: int = 0
    truncated: bool = False


class _FakeSandbox:
    """Minimal ``execute`` stub that scripts pdfinfo/pdftotext output.

    ``script`` is a list of (regex_or_substr, exit_code, output) tuples.
    On each ``execute`` call the first entry whose first element is a
    substring of the command is consumed and its (exit_code, output)
    returned as an ``ExecuteResponse``-shaped object.  Unmatched
    commands raise so test failures are obvious.
    """

    def __init__(self, script: list[tuple[str, int, str]]):
        self._script = list(script)
        self.calls: list[str] = []

    def execute(self, command: str):
        self.calls.append(command)
        for i, (needle, code, out) in enumerate(self._script):
            if needle in command:
                self._script.pop(i)
                return _FakeResp(output=out, exit_code=code)
        raise AssertionError(f"unscripted sandbox command: {command!r}")


def _with_sandbox(script):
    """Context-manager-style helper: bind a fake sandbox via the ContextVar."""
    from assist.sandbox import set_active_sandbox, reset_active_sandbox
    sb = _FakeSandbox(script)
    token = set_active_sandbox(sb)
    return sb, token


def _release(token):
    from assist.sandbox import reset_active_sandbox
    reset_active_sandbox(token)


# --- Pure helpers -------------------------------------------------------

class TestParsePageRange:
    def test_single_page(self):
        assert _parse_page_range("5") == (5, 5)

    def test_inclusive_range(self):
        assert _parse_page_range("5-10") == (5, 10)

    def test_rejects_comma_list(self):
        with pytest.raises(ValueError, match="comma"):
            _parse_page_range("5,7,12")

    def test_rejects_inverted(self):
        with pytest.raises(ValueError, match="inverted"):
            _parse_page_range("10-5")

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="1-based"):
            _parse_page_range("0")

    def test_rejects_malformed_range(self):
        with pytest.raises(ValueError, match="malformed range"):
            _parse_page_range("5-abc")

    def test_total_pages_check(self):
        # First page beyond the end → reject.
        with pytest.raises(ValueError, match="exceeds document length"):
            _parse_page_range("10", total_pages=5)


class TestSplitIntoPages:
    def test_strips_trailing_empty_after_form_feed(self):
        # pdftotext separates pages with form feed; the trailing one
        # often produces an empty string that shouldn't be a page.
        out = "page one\n\x0cpage two\n\x0c"
        assert _split_into_pages(out) == ["page one\n", "page two\n"]

    def test_preserves_empty_page_in_middle(self):
        out = "p1\n\x0c\x0cp3\n"
        assert _split_into_pages(out) == ["p1\n", "", "p3\n"]


class TestGatherSearchHits:
    def test_finds_matching_lines_with_context(self):
        out = "line1\nline2\nline3\n\x0cother page\nlooking for term\nthe end"
        hits = _gather_search_hits(out, "term")
        assert len(hits) == 1
        page, lines = hits[0]
        assert page == 2
        assert any("looking for term" in l for l in lines)

    def test_case_insensitive(self):
        out = "Hello World"
        assert _gather_search_hits(out, "WORLD")

    def test_one_hit_per_page(self):
        # Multiple matches on the same page yield only one entry.
        out = "term term term"
        hits = _gather_search_hits(out, "term")
        assert len(hits) == 1


# --- Mode wiring with a faked sandbox ----------------------------------

class TestReadPdfOrient:
    def test_returns_metadata_and_first_page(self):
        sb, token = _with_sandbox([
            ("pdfinfo", 0, "Pages:      5\nTitle:      Treatment\n"),
            ("pdftotext", 0, "Treatment Guide — overview\n\x0c"),
        ])
        try:
            out = read_pdf("doc.pdf")
        finally:
            _release(token)
        assert "PDF: doc.pdf" in out
        assert "5 pages" in out
        assert "=== PAGE 1 ===" in out
        assert "Treatment Guide" in out

    def test_image_only_pdf_surfaces_clear_message(self):
        # pdftotext returns empty output on image-only PDFs.
        sb, token = _with_sandbox([
            ("pdfinfo", 0, "Pages: 3\n"),
            ("pdftotext", 0, ""),
        ])
        try:
            out = read_pdf("scan.pdf")
        finally:
            _release(token)
        assert "image-based" in out
        assert "OCR is not currently supported" in out


class TestReadPdfSearch:
    def test_returns_matching_page_numbers(self):
        # Page 1 has nothing, page 2 has "dosage", page 3 has "dosage" too.
        full = "intro\n\x0cadult dosage info\n\x0cpediatric dosage\n"
        sb, token = _with_sandbox([("pdftotext", 0, full)])
        try:
            out = read_pdf("doc.pdf", search="dosage")
        finally:
            _release(token)
        assert "Found 'dosage'" in out
        assert "[2, 3]" in out
        assert "=== PAGE 2 (around match) ===" in out
        assert "=== PAGE 3 (around match) ===" in out

    def test_empty_search_rejected(self):
        out = read_pdf("doc.pdf", search="   ")
        assert out.startswith("Error: search=")

    def test_no_matches_returns_clear_message(self):
        sb, token = _with_sandbox([("pdftotext", 0, "no needle here\n")])
        try:
            out = read_pdf("doc.pdf", search="needle that is missing")
        finally:
            _release(token)
        assert "No matches" in out


class TestReadPdfPages:
    def test_single_page_extract(self):
        # pdftotext was called with -f 3 -l 3
        sb, token = _with_sandbox([("pdftotext -f 3 -l 3", 0, "page 3 text\n\x0c")])
        try:
            out = read_pdf("doc.pdf", pages="3")
        finally:
            _release(token)
        assert "=== PAGE 3 ===" in out
        assert "page 3 text" in out

    def test_inclusive_range(self):
        sb, token = _with_sandbox([
            ("pdftotext -f 5 -l 7", 0, "five\n\x0csix\n\x0cseven\n\x0c"),
        ])
        try:
            out = read_pdf("doc.pdf", pages="5-7")
        finally:
            _release(token)
        assert "=== PAGE 5 ===" in out
        assert "=== PAGE 6 ===" in out
        assert "=== PAGE 7 ===" in out


class TestReadPdfErrorPaths:
    def test_search_and_pages_combined_rejected(self):
        out = read_pdf("doc.pdf", search="x", pages="1")
        assert "either search= or pages=, not both" in out

    def test_encrypted_pdf_friendly_error(self):
        sb, token = _with_sandbox([
            ("pdfinfo", 1, ""),
            ("pdftotext", 1, "Error: Incorrect password\n"),
        ])
        try:
            out = read_pdf("locked.pdf")
        finally:
            _release(token)
        assert "password-protected" in out

    def test_pdftotext_failure_returns_short_error(self):
        sb, token = _with_sandbox([
            ("pdftotext -f 1 -l 1", 1, "Syntax Error: PDF file is damaged\n"),
        ])
        try:
            out = read_pdf("bad.pdf", pages="1")
        finally:
            _release(token)
        assert out.startswith("Error: pdftotext failed:")
        assert "PDF file is damaged" in out


# --- Host fallback ------------------------------------------------------

@pytest.mark.skipif(
    not all(
        any(
            os.access(os.path.join(p, b), os.X_OK)
            for p in os.environ.get("PATH", "").split(os.pathsep)
        )
        for b in ("pdftotext", "pdfinfo")
    ),
    reason="host poppler not installed; sandbox path covers production",
)
class TestReadPdfHostFallback:
    """Run against the real fixtures without a sandbox bound.

    Skipped automatically when the host doesn't have poppler — these
    tests catch host-fallback regressions on dev machines that have
    pdftotext/pdfinfo installed (e.g. CI), but production runs through
    the sandbox path which the previous classes cover with mocks.
    """

    def test_orient_real_fixture(self):
        out = read_pdf(SAMPLE_PDF)
        assert "5 pages" in out
        assert "Treatment Guide" in out

    def test_search_real_fixture(self):
        out = read_pdf(SAMPLE_PDF, search="dosage")
        assert "Found 'dosage'" in out
        assert "[3]" in out  # dosage section is page 3

    def test_pages_real_fixture(self):
        out = read_pdf(SAMPLE_PDF, pages="3")
        assert "=== PAGE 3 ===" in out
        assert "Adult dosage" in out


class TestReadPdfMagicByteCheck:
    def test_rejects_non_pdf_on_host(self, tmp_path):
        # No active sandbox → host magic-byte check fires.
        plain = tmp_path / "fake.pdf"
        plain.write_text("not a pdf at all")
        out = read_pdf(str(plain))
        assert "not a PDF" in out

    def test_missing_file_on_host(self, tmp_path):
        out = read_pdf(str(tmp_path / "does-not-exist.pdf"))
        assert "file not found" in out
