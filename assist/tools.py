import os
import re
import shlex
import subprocess
import time
import threading

import requests
from ddgs import DDGS

from assist.sandbox import get_active_sandbox

_last_call_time = 0.0
_rate_lock = threading.Lock()
_MIN_DELAY = 0.5


def _rate_limit():
    """Enforce minimum delay between DuckDuckGo API calls."""
    global _last_call_time
    with _rate_lock:
        now = time.time()
        elapsed = now - _last_call_time
        if elapsed < _MIN_DELAY:
            time.sleep(_MIN_DELAY - elapsed)
        _last_call_time = time.time()


def read_url(url: str) -> str:
    """Extract the content from the given url."""
    _rate_limit()
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"},
            timeout=15,
        )
        resp.raise_for_status()
        text = resp.text
        text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:4000]
    except Exception as e:
        return f"Error fetching URL: {e}"


def search_internet(
        query: str,
        max_results: int = 5,
) -> str:
    """Used to search the internet for information on a given topic using a query string."""
    _rate_limit()
    try:
        results = DDGS().text(query,
                              max_results=max_results,
                              backend="duckduckgo")
    except Exception:
        return "[]"
    normalized = [{"title": r["title"], "url": r["href"], "content": r["body"]} for r in results]
    return str(normalized)


# --- read_pdf -----------------------------------------------------------
# Shell out to ``pdftotext`` (and ``pdfinfo`` for orient mode) inside the
# active sandbox.  Three modes selected by which optional args are set:
# orient (no args), find (search=...), read (pages="N" or "N-M").

_PDF_MAGIC = b"%PDF-"
_ORIENT_PREVIEW_CHARS = 500
_SEARCH_MAX_HITS = 5
_SEARCH_CONTEXT_LINES = 3


def _parse_page_range(pages: str, total_pages: int | None = None) -> tuple[int, int]:
    """Parse a ``pages`` arg into a 1-based ``(first, last)`` inclusive range.

    Accepts a single page (``"5"``) or a single inclusive range
    (``"5-10"``).  Comma lists are intentionally rejected in v1 — the
    small model fumbles them more often than it uses them.

    Raises ``ValueError`` with a human-readable message that the tool
    surfaces as a tool-result string.
    """
    pages = pages.strip()
    if not pages:
        raise ValueError("pages: empty")
    if "," in pages:
        raise ValueError(
            "pages: comma lists not supported — use a single page (\"5\") "
            "or a range (\"5-10\")"
        )
    if "-" in pages:
        a, _, b = pages.partition("-")
        try:
            first, last = int(a), int(b)
        except ValueError:
            raise ValueError(f"pages: malformed range {pages!r}")
    else:
        try:
            first = last = int(pages)
        except ValueError:
            raise ValueError(f"pages: not an integer or range {pages!r}")
    if first < 1 or last < 1:
        raise ValueError("pages: page numbers are 1-based")
    if last < first:
        raise ValueError(f"pages: inverted range {pages!r}")
    if total_pages is not None and first > total_pages:
        raise ValueError(
            f"pages: first page {first} exceeds document length ({total_pages} pages)"
        )
    return first, last


def _run_in_sandbox_or_host(command: str) -> tuple[int, str]:
    """Run ``command`` (a shell string) in the active sandbox if any, else on the host.

    Returns ``(exit_code, combined_output)``.  Sandbox path uses the
    container's ``execute(...)``; host path uses ``subprocess.run`` with
    ``shell=True``.  The host fallback exists for unit tests and any
    eval setup that runs without a sandbox bound.
    """
    sandbox = get_active_sandbox()
    if sandbox is not None:
        resp = sandbox.execute(command)
        return (resp.exit_code if resp.exit_code is not None else 0), resp.output
    proc = subprocess.run(
        command, shell=True, capture_output=True, text=True,
    )
    return proc.returncode, (proc.stdout + proc.stderr)


def _check_pdf_magic(path: str) -> str | None:
    """Return an error message if the file at *path* isn't a PDF, else None.

    Verified host-side because the magic-byte check is just a 5-byte
    read — no need to round-trip through the sandbox.  The active
    sandbox's bind mount makes the same bytes visible on host.
    """
    sandbox = get_active_sandbox()
    if sandbox is not None:
        # In sandbox mode the model's path is container-relative; we let
        # pdftotext error if it's not a PDF rather than trying to resolve
        # the host bind path here.
        return None
    if not os.path.exists(path):
        return f"Error: file not found: {path}"
    try:
        with open(path, "rb") as f:
            head = f.read(5)
    except Exception as e:
        return f"Error reading {path}: {e}"
    if head != _PDF_MAGIC:
        return f"Error: not a PDF: {path}"
    return None


def _pdfinfo_pages(path: str) -> tuple[int, int | None]:
    """Run ``pdfinfo`` and return ``(exit_code, page_count)``.

    page_count is None if the output couldn't be parsed (encrypted,
    missing, etc.) so the caller can format a friendly message instead
    of returning a half-extracted blob.
    """
    code, out = _run_in_sandbox_or_host(f"pdfinfo {shlex.quote(path)}")
    if code != 0:
        return code, None
    for line in out.splitlines():
        if line.startswith("Pages:"):
            try:
                return code, int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return code, None


def _pdftotext(path: str, first: int | None = None, last: int | None = None) -> tuple[int, str]:
    """Run ``pdftotext`` and return ``(exit_code, text)``.

    With *first* and *last* set, restricts to that page range.  Without
    them, extracts the whole document.
    """
    parts = ["pdftotext"]
    if first is not None:
        parts.extend(["-f", str(first)])
    if last is not None:
        parts.extend(["-l", str(last)])
    parts.extend([shlex.quote(path), "-"])
    return _run_in_sandbox_or_host(" ".join(parts))


def _format_orient(path: str, page_count: int | None, first_page_text: str) -> str:
    size_hint = ""
    sandbox = get_active_sandbox()
    if sandbox is None:
        try:
            size_hint = f" | {os.path.getsize(path) // 1024} KB"
        except OSError:
            pass
    pages_label = f"{page_count} pages" if page_count is not None else "unknown pages"
    preview = first_page_text.strip()
    if len(preview) > _ORIENT_PREVIEW_CHARS:
        preview = preview[:_ORIENT_PREVIEW_CHARS].rstrip() + "..."
    if not preview:
        return (
            f"PDF: {path} | {pages_label}{size_hint}\n\n"
            "No extractable text on the first page — this PDF may be "
            "image-based.  OCR is not currently supported."
        )
    return f"PDF: {path} | {pages_label}{size_hint}\n\n=== PAGE 1 ===\n{preview}\n"


def _format_search(path: str, term: str, hits: list[tuple[int, list[str]]]) -> str:
    if not hits:
        return f"No matches for {term!r} in {path}."
    capped = hits[:_SEARCH_MAX_HITS]
    pages_list = [p for p, _ in capped]
    header = f"Found {term!r} on pages {pages_list} ({len(hits)} hits"
    if len(hits) > _SEARCH_MAX_HITS:
        header += f", showing first {_SEARCH_MAX_HITS}"
    header += ")"
    blocks = [header, ""]
    for page, lines in capped:
        blocks.append(f"=== PAGE {page} (around match) ===")
        blocks.extend(lines)
        blocks.append("")
    return "\n".join(blocks).rstrip() + "\n"


def _split_into_pages(text: str) -> list[str]:
    """Split ``pdftotext`` output on form-feed (\\f), one entry per page.

    pdftotext separates pages with ``\x0c``.  An empty trailing entry
    (caused by trailing form-feed) is dropped.
    """
    parts = text.split("\x0c")
    if parts and parts[-1] == "":
        parts.pop()
    return parts


def _gather_search_hits(text: str, term: str) -> list[tuple[int, list[str]]]:
    """Find lines containing *term*, return list of (page_number, context_lines).

    Page numbers are 1-based.  Context is up to ``_SEARCH_CONTEXT_LINES``
    lines centred on the matching line.
    """
    if not term:
        return []
    needle = term.lower()
    hits: list[tuple[int, list[str]]] = []
    for page_idx, page_text in enumerate(_split_into_pages(text), start=1):
        lines = page_text.splitlines()
        for i, line in enumerate(lines):
            if needle in line.lower():
                radius = _SEARCH_CONTEXT_LINES // 2
                start = max(0, i - radius)
                end = min(len(lines), i + radius + 1)
                hits.append((page_idx, lines[start:end]))
                break  # one hit per page is enough for orientation
    return hits


def read_pdf(path: str, search: str | None = None, pages: str | None = None) -> str:
    """Read text out of a PDF, three modes:

    - **Orient** (no other args): returns ``PDF: <path> | <N> pages |
      <KB> KB`` followed by the first page's text (capped to ~500
      chars).  Use this first on any unknown PDF.
    - **Find** (``search="term"``): returns matching page numbers with
      a few lines of context per hit.  Capped to 5 pages.  Faster
      than reading every page when you have a keyword to look for.
    - **Read** (``pages="5"`` or ``pages="5-10"``): full text of those
      pages with inline ``=== PAGE N ===`` markers.  Single page or a
      single inclusive range; comma lists are not supported.

    ``search`` and ``pages`` together are rejected — pick one mode per
    call.  Note: this differs from the line-based ``offset/limit``
    shape of ``read_file``; PDFs are page-natural so the arg is a
    page string.
    """
    # Argument validation runs before any I/O so a clearly-invalid call
    # surfaces a user-facing error even when the file doesn't exist.
    if search is not None and pages is not None:
        return (
            "Error: read_pdf accepts either search= or pages=, not both. "
            "Search first to find pages, then read those pages explicitly."
        )
    if search is not None and not search.strip():
        return "Error: search= must be non-empty."

    magic_err = _check_pdf_magic(path)
    if magic_err is not None:
        return magic_err

    # --- Read mode --------------------------------------------------
    if pages is not None:
        try:
            first, last = _parse_page_range(pages)
        except ValueError as e:
            return f"Error: {e}"
        code, out = _pdftotext(path, first=first, last=last)
        if code != 0:
            if "Incorrect password" in out or "encrypted" in out.lower():
                return f"Error: {path} is password-protected — cannot extract."
            return f"Error: pdftotext failed: {out.strip()[:200]}"
        page_texts = _split_into_pages(out)
        if not page_texts:
            return f"No text found in pages {pages} of {path}."
        blocks = []
        for offset, page_text in enumerate(page_texts):
            blocks.append(f"=== PAGE {first + offset} ===")
            blocks.append(page_text.strip() or "(empty page)")
            blocks.append("")
        return "\n".join(blocks).rstrip() + "\n"

    # --- Find mode --------------------------------------------------
    if search is not None:
        # We need the whole document text to grep through; pdftotext is
        # fast enough that re-running per page would be wasteful.
        code, out = _pdftotext(path)
        if code != 0:
            if "Incorrect password" in out or "encrypted" in out.lower():
                return f"Error: {path} is password-protected — cannot extract."
            return f"Error: pdftotext failed: {out.strip()[:200]}"
        hits = _gather_search_hits(out, search)
        return _format_search(path, search, hits)

    # --- Orient mode ------------------------------------------------
    # Try pdftotext first so encrypted PDFs surface a clean error
    # regardless of pdfinfo's behaviour (encrypted PDFs frequently
    # cause pdfinfo to fail too, and the test scripts that case).
    code, first_page_out = _pdftotext(path, first=1, last=1)
    if code != 0:
        if "Incorrect password" in first_page_out or "encrypted" in first_page_out.lower():
            return f"Error: {path} is password-protected — cannot extract."
        return f"Error: pdftotext failed: {first_page_out.strip()[:200]}"
    info_code, page_count = _pdfinfo_pages(path)
    # Page count is nice-to-have; if pdfinfo fails (rare for non-encrypted
    # PDFs that pdftotext just succeeded on) we render orient mode
    # without the count rather than failing the whole call.
    page_texts = _split_into_pages(first_page_out)
    first_page_text = page_texts[0] if page_texts else ""
    return _format_orient(path, page_count, first_page_text)
