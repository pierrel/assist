"""Per-file diff rendering for the inline thread page and the review page.

Pure functions — no FastAPI dependency, no module-level state.  Both
``threads.render_thread`` and ``review.render_review_page`` lean on
``render_file_diff`` and the shared ``_DIFF_CSS`` block.
"""
from __future__ import annotations

import html

from assist.domain_manager import Change


# The inline view on /thread/{tid} shows per-file diffs with a hard cap so
# a regenerated lockfile (or a 5k-line refactor) can't kill the browser.
# The full-fat view lives at /thread/{tid}/review and renders everything.
INLINE_FILE_LINE_CAP = 600
INLINE_FILE_BYTE_CAP = 64 * 1024
INLINE_TOTAL_BYTE_CAP = 256 * 1024

# Diff lines that aren't user-comment-worthy: file/index headers and the
# "no newline at end of file" marker.  Hunk headers and +/-/context rows
# are clickable on the review page.
_META_PREFIXES = (
    "diff --git ", "index ", "similarity index ", "dissimilarity index ",
    "rename from ", "rename to ", "copy from ", "copy to ",
    "new file mode", "deleted file mode", "old mode", "new mode",
    "--- ", "+++ ",
    "\\ ",  # "\ No newline at end of file"
)


def _classify_diff_line(line: str) -> str:
    """Return CSS class suffix for a unified-diff line.

    Returns one of: 'meta', 'hunk', 'add', 'del', 'ctx'.  Order of checks
    matters: '+++' / '---' headers must be caught as meta before falling
    into the +/- branches.
    """
    for p in _META_PREFIXES:
        if line.startswith(p):
            return "meta"
    if line.startswith("@@"):
        return "hunk"
    if line.startswith("+"):
        return "add"
    if line.startswith("-"):
        return "del"
    return "ctx"


def _is_clickable(kind: str) -> bool:
    """Per-line comments only attach to hunk headers and content rows."""
    return kind in ("hunk", "add", "del", "ctx")


def _diff_stats(diff: str) -> tuple[int, int]:
    """Return (additions, deletions) counting only +/- content lines.

    Skips the +++/--- file headers.
    """
    adds = dels = 0
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            adds += 1
        elif line.startswith("-"):
            dels += 1
    return adds, dels


def _is_binary_diff(diff: str) -> bool:
    """git diff emits one 'Binary files ... differ' line for binary files."""
    return any(
        line.startswith("Binary files ") and line.endswith(" differ")
        for line in diff.splitlines()
    )


def _rename_pair(diff: str) -> tuple[str, str] | None:
    """Return (old_path, new_path) if the diff carries rename headers, else None."""
    old = new = None
    for line in diff.splitlines():
        if line.startswith("rename from "):
            old = line[len("rename from "):].strip()
        elif line.startswith("rename to "):
            new = line[len("rename to "):].strip()
        if old and new:
            return old, new
    return None


def render_file_diff(
    change: Change, file_idx: int, *, full: bool, tid: str | None = None,
) -> str:
    """Render one file's diff as a collapsible <details> block.

    *full=False* (inline view): truncate when the file exceeds the inline
    cap and emit no per-line click affordances.  *full=True* (review
    page): render every row with stable data-* attributes for the click
    handler.

    *tid* is required for inline (full=False) renders so the truncation
    placeholder can link to ``/thread/{tid}/review``.  Passed in
    directly rather than templated post-hoc, since a diff line can
    contain the literal token ``{tid}`` and we'd silently corrupt it.
    """
    path_attr = html.escape(change.path, quote=True)
    file_id = f"file-{file_idx}"
    adds, dels = _diff_stats(change.diff)
    stats_html = (
        f'<span class="diff-stats">'
        f'<span class="add">+{adds}</span> '
        f'<span class="del">−{dels}</span>'
        f'</span>'
    )

    # Binary files: short header, no clickable rows.
    if _is_binary_diff(change.diff):
        body = '<div class="diff-binary">Binary file — no preview available.</div>'
        return (
            f'<details class="diff-file" id="{file_id}" {"open" if full else ""}>'
            f'<summary><span>{html.escape(change.path)}</span>{stats_html}'
            f'<span class="diff-binary-badge">binary</span></summary>'
            f'{body}</details>'
        )

    lines = change.diff.splitlines()
    n_lines = len(lines)
    n_bytes = len(change.diff.encode("utf-8", errors="replace"))

    if not full and (n_lines > INLINE_FILE_LINE_CAP or n_bytes > INLINE_FILE_BYTE_CAP):
        if tid is None:
            raise ValueError("render_file_diff(full=False) requires tid for the truncation link")
        body = (
            f'<div class="diff-truncated">Diff too large to preview here '
            f'({n_lines} lines, {n_bytes // 1024} KB). '
            f'<a href="/thread/{html.escape(tid, quote=True)}/review#{file_id}">Open in review</a> '
            f'to see the full diff.</div>'
        )
        return (
            f'<details class="diff-file" id="{file_id}">'
            f'<summary><span>{html.escape(change.path)}</span>{stats_html}'
            f'<span class="diff-truncated-badge">truncated</span></summary>'
            f'{body}</details>'
        )

    rows: list[str] = []
    for row_idx, line in enumerate(lines, start=1):
        kind = _classify_diff_line(line)
        cls = f"diff-row diff-{kind}"
        # Collapse trailing whitespace in display while keeping the raw
        # text in data-linetext for the submit-message context block.
        text_html = html.escape(line) if line else "&nbsp;"
        if full and _is_clickable(kind):
            data = (
                f' data-key="{path_attr}::{row_idx}"'
                f' data-file="{path_attr}"'
                f' data-row="{row_idx}"'
                f' data-linetext="{html.escape(line, quote=True)}"'
            )
        else:
            data = ""
        rows.append(f'<div class="{cls}"{data}>{text_html}</div>')

    body = f'<div class="diff-rows">{"".join(rows)}</div>'
    return (
        f'<details class="diff-file" id="{file_id}" {"open" if full else ""}>'
        f'<summary><span>{html.escape(change.path)}</span>{stats_html}</summary>'
        f'{body}</details>'
    )


_DIFF_CSS = """
.diff-file { margin: .6rem 0; border: 1px solid #d0d7de; border-radius: 6px; overflow: hidden; background: #fff; }
.diff-file > summary { padding: .5rem .75rem; background: #f6f8fa; cursor: pointer; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .85rem; display: flex; gap: .6rem; align-items: center; flex-wrap: wrap; list-style: none; }
.diff-file > summary::-webkit-details-marker { display: none; }
.diff-file > summary::before { content: "▶"; font-size: .7rem; color: #57606a; transition: transform .15s; }
.diff-file[open] > summary::before { transform: rotate(90deg); }
.diff-file[open] > summary { border-bottom: 1px solid #d0d7de; }
.diff-stats { font-size: .75rem; font-family: ui-monospace, monospace; }
.diff-stats .add { color: #1a7f37; }
.diff-stats .del { color: #cf222e; }
.diff-binary-badge, .diff-truncated-badge { font-size: .7rem; background: #ddf4ff; color: #0969da; border: 1px solid #b6e3ff; padding: .05rem .4rem; border-radius: 10px; }
.diff-truncated-badge { background: #fff8c5; color: #9a6700; border-color: #d4a72c; }
.diff-rows { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .8rem; line-height: 1.4; overflow-x: auto; }
.diff-row { padding: 0 .5rem; white-space: pre; min-height: 1.2em; border-left: 3px solid transparent; }
.diff-row.diff-add { background: #e6ffec; border-left-color: #abf2bc; }
.diff-row.diff-del { background: #ffebe9; border-left-color: #ffabab; }
.diff-row.diff-ctx { background: #fff; }
.diff-row.diff-hunk { background: #ddf4ff; color: #57606a; font-weight: 600; }
.diff-row.diff-meta { background: #f6f8fa; color: #57606a; font-size: .75rem; }
.diff-row[data-key] { cursor: pointer; }
.diff-row[data-key]:hover { outline: 1px solid #0969da; outline-offset: -1px; }
.diff-binary { padding: .6rem .75rem; color: #57606a; font-style: italic; font-size: .85rem; }
.diff-truncated { padding: .6rem .75rem; background: #fff8c5; color: #57606a; font-size: .85rem; }
.diff-truncated a { color: #0969da; text-decoration: underline; }
.diff-overflow { padding: .6rem .75rem; background: #fff8c5; border: 1px solid #d4a72c; border-radius: 6px; color: #57606a; font-size: .85rem; margin: .6rem 0; }
"""


def _render_inline_diffs(tid: str, changes: list[Change]) -> str:
    """Render the per-file collapsible diff stack for the thread page.

    Applies the inline truncation rules so a long diff can't kill the
    browser.  Each placeholder for an over-cap file links to the review
    page for the actual content.
    """
    if not changes:
        return ""
    rendered: list[str] = []
    total_bytes = 0
    truncated_after: int | None = None
    for idx, change in enumerate(changes):
        block = render_file_diff(change, idx, full=False, tid=tid)
        total_bytes += len(block)
        if total_bytes > INLINE_TOTAL_BYTE_CAP:
            truncated_after = idx
            break
        rendered.append(block)
    body = "\n".join(rendered)
    if truncated_after is not None:
        remaining = len(changes) - truncated_after
        body += (
            f'<div class="diff-overflow">'
            f'+ {remaining} more file{"s" if remaining != 1 else ""} '
            f'— <a href="/thread/{tid}/review">open the review page</a> '
            f'to see them all.'
            f'</div>'
        )
    return body
