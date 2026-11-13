"""Bounded subprocess for fetching and extracting untrusted web PDFs."""

import io
import resource
import sys

import requests


_MAX_INPUT = 25 * 1024 * 1024
_MAX_PAGES = 25
_MAX_CHARS = 200_000


def extract_pdf(data: bytes) -> str:
    """Return a bounded, page-marked text extract or raise for unreadable PDFs."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data), strict=False)
    if reader.is_encrypted:
        raise ValueError("encrypted PDF")
    out = []
    remaining = _MAX_CHARS
    count = min(len(reader.pages), _MAX_PAGES)
    for index in range(count):
        page_text = reader.pages[index].extract_text() or ""
        if not page_text:
            continue
        piece = f"\nPage {index + 1}:\n{page_text}"
        out.append(piece[:remaining])
        remaining -= min(len(piece), remaining)
        if not remaining:
            break
    if not out:
        raise ValueError("no extractable PDF text")
    if len(reader.pages) > count or not remaining:
        out.append("\n[PDF text truncated at the page or character limit]")
    return "".join(out)


def _fetch_pdf(url: str) -> bytes:
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"},
        timeout=(5, 5), stream=True,
    )
    with response:
        response.raise_for_status()
        length = response.headers.get("Content-Length", "")
        if str(length).isdigit() and int(length) > _MAX_INPUT:
            raise ValueError("oversized PDF")
        chunks = []
        size = 0
        for chunk in response.iter_content(chunk_size=65536):
            size += len(chunk)
            if size > _MAX_INPUT:
                raise ValueError("oversized PDF")
            chunks.append(chunk)
    return b"".join(chunks)


def main() -> int:
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (8, 8))
    try:
        url = sys.stdin.buffer.read(8193)
        if len(url) > 8192:
            return 2
        sys.stdout.write(extract_pdf(_fetch_pdf(url.decode("utf-8"))))
        return 0
    except requests.HTTPError:
        return 3
    except requests.Timeout:
        return 4
    except requests.RequestException:
        return 5
    except Exception:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
