"""Generate small PDF fixtures used by tests/test_tools.py and the
edd/eval/test_pdf_reading.py behavior evals.

Run from the repo root:
    .venv/bin/python tests/fixtures/pdf/generate.py

Produces three files in this directory:
  - sample.pdf  : 5 pages of distinct prose, used for orient/search/read tests.
  - big.pdf     : 60 pages, used for the "don't dump the full PDF" anti-test.
  - encrypted.pdf : password-protected variant of sample.pdf.

The script is checked in so anyone can regenerate.  reportlab and
pypdf are dev-only dependencies — not required at runtime.  A
hand-crafted "image-only" fixture is intentionally not generated; the
image-only path is exercised by mocking ``pdftotext``'s empty output
in unit tests.
"""
from __future__ import annotations

import os

from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


HERE = os.path.dirname(os.path.abspath(__file__))


def _write_text_pdf(path: str, pages: list[list[str]]) -> None:
    """Write *pages* (a list of lines per page) as a real text PDF."""
    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter
    for page_lines in pages:
        y = height - 72  # 1-inch top margin
        for line in page_lines:
            c.drawString(72, y, line)
            y -= 14
        c.showPage()
    c.save()


def _generate_sample() -> None:
    """Five pages with distinct content for orient/search/read tests."""
    pages = [
        [
            "Treatment Guide — overview",
            "This document covers prescribing guidance.",
            "See page 3 for dosage instructions.",
        ],
        [
            "Page two — patient screening",
            "Verify allergies before administering.",
            "Note any contraindications in the chart.",
        ],
        [
            "Page three — dosage",
            "Adult dosage is 5 mg per kilogram.",
            "Pediatric dosage halves that to 2.5 mg per kilogram.",
            "Maximum dosage must not exceed 500 mg per day.",
        ],
        [
            "Page four — side effects",
            "Mild nausea is reported in 8 percent of patients.",
            "Discontinue use if symptoms persist.",
        ],
        [
            "Page five — references",
            "Consult the WHO formulary and the local guidelines.",
        ],
    ]
    _write_text_pdf(os.path.join(HERE, "sample.pdf"), pages)


def _generate_big() -> None:
    """Sixty pages of filler.  Used to test that the agent does not
    dump the full PDF when an open question would otherwise tempt a
    1-60 page read."""
    pages = []
    for i in range(1, 61):
        pages.append([
            f"Page {i} of 60 — filler",
            f"This is line two of page {i}.",
            f"Section {i // 5 + 1} marker on page {i}.",
        ])
    # Plant a unique token deep in the document for search-reach tests.
    pages[42][0] = "Page 43 of 60 — hidden marker token: bluefin"
    _write_text_pdf(os.path.join(HERE, "big.pdf"), pages)


def _generate_encrypted() -> None:
    """Password-protect sample.pdf as encrypted.pdf for the encrypted-error test."""
    src = os.path.join(HERE, "sample.pdf")
    dst = os.path.join(HERE, "encrypted.pdf")
    reader = PdfReader(src)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt(user_password="secret", owner_password="secret")
    with open(dst, "wb") as f:
        writer.write(f)


if __name__ == "__main__":
    _generate_sample()
    _generate_big()
    _generate_encrypted()
    for name in ("sample.pdf", "big.pdf", "encrypted.pdf"):
        size = os.path.getsize(os.path.join(HERE, name))
        print(f"{name}: {size} bytes")
