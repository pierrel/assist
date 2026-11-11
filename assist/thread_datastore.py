"""Host-owned immutable text records and exact reference binding."""

from __future__ import annotations

import ast
import json
import re
import secrets
import sqlite3
import os
import stat
from urllib.parse import urlsplit

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.config import get_config
from markdown.extensions.tables import TableProcessor
from markdown.extensions.fenced_code import FencedBlockPreprocessor
from assist.middleware.output_sanitization import _sanitize


_BARE_REF = re.compile(r"(?<!\S)ref:([A-Za-z0-9_-]*)", re.I)
_STREAM_START = re.compile(r"\[|(?<!\S)ref:", re.I)
_LINK_TARGET = re.compile(r"\]\(ref:", re.I)
_LABEL_ATOM = r"(?:\\[^\n]|\[(?:\\[^\n]|[^\[\]\\\n])*\]|[^\[\]\\\n])"
_LABEL_OPEN = re.compile(r"\[(?:" + _LABEL_ATOM + r"){1,500}\]")
_STREAM_PREFIXES = {"r", "re", "ref"}
_RECORD_ID = re.compile(r"r[0-9a-f]{32}\Z")
_UNBOUND = "[source reference unresolved; link reachability was not checked]"
_MAX_TEXT_OUTPUT = 8 * 1024 * 1024
_INLINE_REF = re.compile(r"ref:([A-Za-z0-9_-]*)", re.I)
_TEXT_START = re.compile(r"https?://|\[|ref:([A-Za-z0-9_-]*)", re.I)
_URL_LITERAL = re.compile(r"https?://[^\s<>]+", re.I)
_LABEL_BOUNDARIES = re.compile(r"\\[^\n]|\[|\]|\n")
_SAVED_FILE_REFS = re.compile(r"https?://[^\s<>]+|ref:([A-Za-z0-9_-]*)", re.I)
BOUND_FILE_URLS_ARTIFACT = "assist_bound_file_urls"


def _code_spans(text: str) -> list[tuple[int, int, int]]:
    """Pair lexical code runs linearly, retaining content and closing bounds."""
    tokens = [token for token in TableProcessor.RE_CODE_PIPES.finditer(text)
              if token.group(2) or token.group(3)]
    future, closes = {}, {}
    for token in reversed(tokens):
        escaped = bool(token.group(2))
        size = len(token.group(2) or token.group(3)) - escaped
        closes[token.start()] = future.get(size - escaped) if size > escaped else None
        future[size] = (token.start(), token.end())
    code_end, spans = 0, []
    for token in tokens:
        if token.start() < code_end:
            continue
        if close := closes[token.start()]:
            spans.append((token.end(), *close))
            code_end = close[1]
    return spans


def _table_pipes(row: str) -> list[int]:
    """Find renderer-compatible pipes with linear code-span/escape handling."""
    spans, cursor, pipes = _code_spans(row), 0, []
    for token in TableProcessor.RE_CODE_PIPES.finditer(row):
        while cursor < len(spans) and token.start() >= spans[cursor][2]:
            cursor += 1
        if token.group(5) and not (cursor < len(spans) and spans[cursor][0] <= token.start()):
            pipes.append(token.start())
    return pipes


def _table_width(row: str, border: bool | None = None) -> tuple[int, bool]:
    row = row.strip(" ")
    pipes = _table_pipes(row)
    left, right = bool(pipes) and pipes[0] == 0, bool(pipes) and pipes[-1] == len(row) - 1
    if border is None:
        border = left or right
    return len(pipes) + 1 - (left + right if border else 0), border


def _table_lines(text: str):
    """Recognize header/separator blocks and explicit rows with linear lookahead."""
    start, table, fence, block_start = 0, False, None, True
    while start < len(text):
        end = text.find("\n", start)
        if end < 0:
            end = len(text)
        row = text[start:end].removesuffix("\r")
        stripped = row.strip(" ")
        marker = re.match(r"(`{3,}|~{3,})", row)
        opening = (not fence and marker and FencedBlockPreprocessor.FENCED_BLOCK_RE.match(
            row + "\n\n" + marker[1] + "\n"))
        if fence or opening:
            if fence and marker and marker[1] == fence and not row[marker.end():].strip(" "):
                fence = None
            elif opening:
                fence = marker[1]
            table, current, block_start = False, False, fence is None
        elif not stripped:
            table, current, block_start = False, False, True
        elif not table and row.startswith(("    ", "\t")):
            current, block_start = False, False
        else:
            if block_start and end < len(text):
                following_end = text.find("\n", end + 1)
                separator = text[end + 1:following_end if following_end >= 0 else len(text)].removesuffix("\r").strip(" ")
                if "|" in separator and set(separator) <= set("|:- "):
                    width, border = _table_width(row)
                    table = (width > 1 or border) and _table_width(separator, border)[0] == width
            current, block_start = table or stripped.startswith("|"), False
        yield end, [start + pipe for pipe in _table_pipes(row)] if current else []
        start = end + 1


def _generic_label_ends(text: str) -> dict[int, int]:
    """Match ordinary link-label brackets once without the source-label cap."""
    stack, ends = [], {}
    for token in _LABEL_BOUNDARIES.finditer(text):
        if token.group() == "\n":
            stack.clear()
        elif token.group() == "[":
            stack.append(token.start())
        elif token.group() == "]" and stack:
            start = stack.pop()
            if text[token.end():token.end() + 1] == "(":
                ends[start] = token.start()
    return ends


def _text_scanner(text: str, edit_context: bool = False):
    """Keep URLs opaque inside their destination, label or actual table cell."""
    lines = iter(_table_lines(text))
    line_end, pipes = next(lines, (len(text), []))
    pipe_index = 0
    label_end = -1
    label_ends = None
    code_spans, code_index = None, 0

    def scan(position: int):
        nonlocal line_end, pipes, pipe_index, label_end, label_ends, code_spans, code_index
        marker = _TEXT_START.search(text, position)
        if marker is None:
            return None
        while marker.start() > line_end:
            line_end, pipes = next(lines, (len(text), []))
            pipe_index = 0
        if not marker.group().lower().startswith(("http://", "https://")):
            if marker.group() == "[":
                close = _label_close(text, marker.start())
                if close is None:
                    if label_ends is None:
                        label_ends = _generic_label_ends(text)
                    close = label_ends.get(marker.start())
                if close is not None and text[close + 1:close + 2] == "(":
                    label_end = max(label_end, close)
            return marker
        start, end = marker.span()
        if code_spans is None:
            code_spans = _code_spans(text)
        while code_index < len(code_spans) and start >= code_spans[code_index][2]:
            code_index += 1
        if (code_index < len(code_spans)
                and code_spans[code_index][0] <= start < code_spans[code_index][1]):
            return _URL_LITERAL.match(text, start, code_spans[code_index][1]) or marker
        if start and text[start - 1] == "<":
            return _URL_LITERAL.match(text, start) or marker
        prefix = start
        while prefix > position and text[prefix - 1].isspace():
            prefix -= 1
        destination = text[max(0, prefix - 2):prefix] == "]("
        in_label = start < label_end
        if not destination and not pipes and not in_label:
            literal = _URL_LITERAL.match(text, start) or marker
            if edit_context:
                literal_value = literal.group()
                literal_pipes = _table_pipes(literal_value)
                if literal_pipes and any(token.start() > literal_pipes[0]
                        and literal_value[token.start() - 1] != "\\"
                        for token in _INLINE_REF.finditer(literal_value)):
                    raise ValueError("ambiguous URL/pipe edit fragment; include table header/context, use literal values, or use a full write")
            return literal
        while pipe_index < len(pipes) and pipes[pipe_index] < start:
            pipe_index += 1
        cell_end = pipes[pipe_index] if pipe_index < len(pipes) else len(text)
        depth, index = 0, end
        while index < (label_end if in_label else len(text)):
            char = text[index]
            if char.isspace() or char in "<>":
                break
            if char == "\\" and index + 1 < len(text) and text[index + 1] in "\\()|":
                index += 2
                continue
            if index == cell_end:
                break
            if destination and char == ")":
                if depth == 0:
                    break
                depth -= 1
            elif destination and char == "(":
                depth += 1
            index += 1
        return _URL_LITERAL.match(text, start, index) or marker

    return scan


def _check_text_value(value: str) -> None:
    position = 0
    scanner = _text_scanner(value)
    while token := scanner(position):
        if (token.group().lower().startswith("ref:")
                and (token.start() == 0 or value[token.start() - 1] != "\\")):
            raise ValueError("unescaped reference syntax is reserved; an explicit preceding backslash is retained, never decoded")
        position = token.end()
    if _sanitize(value) != value:
        raise ValueError("terminal control sequences are not supported in exact text")


def _source_url(value: object) -> str | None:
    if (not isinstance(value, str) or len(value) > 8192
            or any(c in "<>" or c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value)):
        return None
    try:
        parsed = urlsplit(value)
        return value if parsed.scheme in {"http", "https"} and parsed.hostname else None
    except ValueError:
        return None


class ThreadDataStore:
    """Append-only text records scoped to one host-owned thread directory."""

    def __init__(self, path: str):
        self.path = path

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, kind TEXT NOT NULL, value TEXT NOT NULL, title TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS imports (owner TEXT NOT NULL, record_id TEXT NOT NULL, PRIMARY KEY (owner, record_id))")
        return db

    def put_text(self, value: str, kind: str = "text", title: str = "") -> str:
        if (not isinstance(value, str) or len(value) > 64 * 1024
                or not isinstance(kind, str) or not 1 <= len(kind) <= 64
                or kind == "source" or not isinstance(title, str) or len(title) > 500):
            raise ValueError("invalid text record; source records require trusted retrieval")
        with self._connect() as db:
            record_id = "r" + secrets.token_hex(16)
            db.execute("INSERT INTO records VALUES (?, ?, ?, ?)",
                       (record_id, kind, value, title))
            return record_id

    def get(self, record_id: str) -> dict[str, str] | None:
        if not isinstance(record_id, str) or not _RECORD_ID.fullmatch(record_id):
            return None
        with self._connect() as db:
            row = db.execute("SELECT id, kind, value, title FROM records WHERE id = ?", (record_id,)).fetchone()
            return dict(zip(("id", "kind", "value", "title"), row)) if row else None

    def has_text_records(self) -> bool:
        with self._connect() as db:
            return db.execute("SELECT 1 FROM records WHERE kind != 'source' LIMIT 1").fetchone() is not None

    def import_from(self, child: "ThreadDataStore", owner: str, kind: str) -> None:
        """Copy one kind of child records and retain their durable import owner."""
        with child._connect() as source, self._connect() as target:
            for row in source.execute(
                    "SELECT id, kind, value, title FROM records WHERE kind = ?", (kind,)):
                target.execute("INSERT OR IGNORE INTO records VALUES (?, ?, ?, ?)", row)
                target.execute("INSERT OR IGNORE INTO imports VALUES (?, ?)", (owner, row[0]))

    def list_imported(self, owner: str, kind: str | None = None,
                      limit: int = 20) -> list[dict[str, str]]:
        with self._connect() as db:
            kind_clause = " AND records.kind = ?" if kind is not None else ""
            params = (owner, kind, limit) if kind is not None else (owner, limit)
            return [dict(id=row[0], kind=row[1], value=row[2], title=row[3])
                    for row in db.execute(
                        "SELECT records.id, records.kind, records.value, records.title "
                        "FROM records JOIN imports ON records.id = imports.record_id "
                        "WHERE imports.owner = ?" + kind_clause + " ORDER BY records.rowid DESC LIMIT ?", params)]

    def list(self, kind: str | None = None, limit: int = 20) -> list[dict[str, str]]:
        with self._connect() as db:
            where = " WHERE kind = ?" if kind is not None else ""
            params = (kind, limit) if kind is not None else (limit,)
            return [dict(id=row[0], kind=row[1], value=row[2], title=row[3])
                    for row in db.execute("SELECT id, kind, value, title FROM records" + where + " ORDER BY rowid DESC LIMIT ?", params)]

    def search(self, query: str, kind: str | None = None, limit: int = 20) -> list[dict[str, str]]:
        with self._connect() as db:
            kind_clause = "kind = ? AND " if kind is not None else ""
            params = (kind, query, query, limit) if kind is not None else (query, query, limit)
            return [dict(id=row[0], kind=row[1], value=row[2], title=row[3])
                    for row in db.execute("SELECT id, kind, value, title FROM records WHERE " + kind_clause + "(instr(value, ?) > 0 OR instr(title, ?) > 0) ORDER BY rowid DESC LIMIT ?", params)]


def register_source(store: ThreadDataStore, url: str, title: str = "") -> str | None:
    """Capture a validated retrieval URL as an immutable text record."""
    if _source_url(url) is None:
        return None
    with store._connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT id FROM records WHERE kind = 'source' AND value = ?", (url,)).fetchone()
        if row:
            return row[0]
        record_id = "r" + secrets.token_hex(16)
        db.execute("INSERT INTO records VALUES (?, 'source', ?, ?)",
                   (record_id, url, title[:500]))
        return record_id


def source_url(store: ThreadDataStore, record_id: str) -> str | None:
    row = store.get(record_id)
    return row["value"] if row and row["kind"] == "source" else None


def resolve_host_report(path: str, store: ThreadDataStore) -> bool:
    """Atomically bind saved source refs through a no-follow report dir handle."""
    directory = os.path.dirname(path)
    basename = os.path.basename(path)
    root = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = os.open(basename, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=root)
        with os.fdopen(fd, "rb") as source:
            old = os.fstat(source.fileno())
            if not stat.S_ISREG(old.st_mode):
                raise ValueError("research report must be a regular file")
            data = source.read(8 * 1024 * 1024 + 1)
        if len(data) > 8 * 1024 * 1024:
            raise ValueError("final research report exceeds 8 MiB source-binding limit")
        original = data.decode("utf-8")
        resolved = resolve_file_references(original, store,
                                      max_output_bytes=8 * 1024 * 1024)
        if resolved == original:
            return False
        temporary = ".source-" + secrets.token_hex(16)
        target_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                            old.st_mode & 0o777, dir_fd=root)
        try:
            with os.fdopen(target_fd, "wb") as target:
                target.write(resolved.encode("utf-8"))
                os.fchmod(target.fileno(), old.st_mode & 0o777)
            current = os.stat(basename, dir_fd=root, follow_symlinks=False)
            if (current.st_ino, current.st_mtime_ns, current.st_size) != (
                    old.st_ino, old.st_mtime_ns, old.st_size):
                raise ValueError("research report changed during source binding")
            os.replace(temporary, basename, src_dir_fd=root, dst_dir_fd=root)
            temporary = None
        finally:
            if temporary is not None:
                os.unlink(temporary, dir_fd=root)
        return True
    finally:
        os.close(root)


def _label_close(text: str, start: int = 0) -> int | None:
    """Find an explicit citation label in a bounded slice using the regex engine."""
    match = _LABEL_OPEN.match(text, start, min(len(text), start + 502))
    return match.end() - 1 if match else None


def _link_at(text: str, start: int, store: ThreadDataStore,
             bind_text: bool = False) -> tuple[int, str, str | None, bool] | None:
    if _LINK_TARGET.search(text, start + 1, min(len(text), start + 507)) is None:
        return None
    close = _label_close(text, start)
    if close is None or text[close + 1:close + 6].lower() != "(ref:":
        return None
    label = text[start + 1:close]
    label_used_text = False
    if bind_text:
        label, label_used_text = _resolve_once(label, store, text_only=True, bind_text=True)
    target_start = close + 6
    end = target_start
    while end < len(text) and text[end] not in "[)\n":
        end += 1
    if end < len(text) and text[end] == ")":
        url = source_url(store, text[target_start:end].lower())
        if not url:
            row = store.get(text[target_start:end].lower())
            if row and row["kind"] != "source":
                raise ValueError("generic text cannot be a source-link target; use an automatically registered source")
        if url:
            # Markdown decodes entities and escapes, parses code spans, and
            # splits table cells at pipes inside link targets.
            # Encode outward bound destinations so the rendered href matches the record.
            target = (re.sub(r"&(?=(?:#[0-9]+|#[xX][0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]+);)",
                             "&amp;", url).replace("\\", "&#92;")
                     .replace("`", "&#96;").replace("|", "&#124;"))
            return end + 1, f"[{label}](<{target}>)", url, label_used_text
        return end + 1, f"{label} {_UNBOUND}", None, label_used_text
    return end, f"{label} {_UNBOUND}", None, label_used_text


def _resolve_once(text: str, store: ThreadDataStore,
                  max_output_bytes: int | None = None, *,
                  bound_sources: list[str] | None = None,
                  text_only: bool = False, bind_text: bool = False,
                  edit_context: bool = False) -> tuple[str, bool]:
    """Substitute original tokens without scanning inserted text in this pass."""
    parts = []
    used = 0
    used_text = False

    def append(piece: str) -> None:
        nonlocal used
        if max_output_bytes is not None or bind_text:
            used += len(piece.encode("utf-8"))
            limit = max_output_bytes if max_output_bytes is not None else (
                _MAX_TEXT_OUTPUT if used_text else None)
            if limit is not None and used > limit:
                raise ValueError("resolved research report exceeds 8 MiB source-binding limit")
        parts.append(piece)

    position = 0
    scanner = _text_scanner(text, edit_context) if bind_text else lambda pos: _STREAM_START.search(text, pos)
    while marker := scanner(position):
        append(text[position:marker.start()])
        if marker.group().lower().startswith(("http://", "https://")):
            append(marker.group())
            position = marker.end()
        elif text[marker.start()] == "[":
            link = None if text_only else _link_at(text, marker.start(), store, bind_text)
            if link is None:
                append("[")
                position = marker.start() + 1
            else:
                position, replacement, bound_url, label_used_text = link
                used_text |= label_used_text
                append(replacement)
                if bound_sources is not None and bound_url is not None:
                    bound_sources.append(bound_url)
        else:
            token = _INLINE_REF.match(text, marker.start())
            assert token is not None
            row = store.get(token.group(1).lower())
            escaped = marker.start() > 0 and text[marker.start() - 1] == "\\"
            bound_url = None
            if bind_text and not escaped and row is None:
                raise ValueError("unknown or malformed inline reference; use the exact returned token or an explicit literal escape")
            if bind_text and row and row["kind"] != "source" and not escaped:
                _check_text_value(row["value"])
                used_text = True
                append(row["value"])
            elif not text_only and _BARE_REF.match(text, marker.start()):
                bound_url = row["value"] if row and row["kind"] == "source" else None
                append(bound_url or _UNBOUND)
            else:
                append(token.group())
            if bound_sources is not None and bound_url is not None:
                bound_sources.append(bound_url)
            position = token.end()
    append(text[position:])
    return "".join(parts), used_text


def resolve_references(text: str, store: ThreadDataStore,
                       max_output_bytes: int | None = None, *,
                       bound_sources: list[str] | None = None,
                       text_only: bool = False, edit_context: bool = False) -> str:
    """Bind exact inline text and source refs; leave ordinary URLs unchanged.

    File arguments use ``text_only`` to retain source drafts. Generic insertion
    must already be stable under another full pass. Escapes are retained, not
    decoded. Only source records contribute to ``bound_sources`` or link targets.
    """
    available = store.has_text_records()
    limit = (max_output_bytes if max_output_bytes is not None else
             _MAX_TEXT_OUTPUT if available and not text_only else None)
    resolved, used_text = _resolve_once(
        text, store, limit, bound_sources=bound_sources,
        text_only=text_only, bind_text=available, edit_context=edit_context)
    if used_text:
        check_limit = max_output_bytes if max_output_bytes is not None else _MAX_TEXT_OUTPUT
        complete = (_resolve_once(text, store, check_limit, bind_text=True, edit_context=edit_context)[0]
                    if text_only else resolved)
        if (text_only
                and _resolve_once(resolved, store, check_limit, bind_text=True, edit_context=edit_context)[0] != complete):
            raise ValueError("text insertion would change source-draft parsing; preserve literal boundaries")
        if _resolve_once(complete, store, check_limit, bind_text=True, edit_context=edit_context)[0] != complete:
            raise ValueError("text insertion would create another reference; preserve literal boundaries")
    return resolved


def resolve_file_references(text: str, store: ThreadDataStore,
                            max_output_bytes: int = _MAX_TEXT_OUTPUT, *,
                            bound_sources: list[str] | None = None) -> str:
    """Bind saved-file source refs only; generic values bind at write/edit time.

    Saved URL bytes are opaque regardless of later Markdown context. Known raw
    generic drafts outside those URLs are unsupported and fail before replacing
    report bytes or projecting a read result. Explicit escapes remain literal.
    """
    if store.has_text_records():
        for token in _SAVED_FILE_REFS.finditer(text):
            if (token.group(1) is not None
                    and (token.start() == 0 or text[token.start() - 1] != "\\")):
                row = store.get(token.group(1).lower())
                if row and row["kind"] != "source":
                    raise ValueError("unbound generic reference in saved file; bind values with write_file/edit_file before saving")
    return _resolve_once(text, store, max_output_bytes, bound_sources=bound_sources)[0]


def _display_references(text: str, store: ThreadDataStore) -> str:
    try:
        return resolve_references(text, store)
    except ValueError as error:
        return f"[text reference unresolved: {error}]"


class ReferenceStream:
    """Stream sources incrementally; check whole messages when text records exist."""

    def __init__(self, store: ThreadDataStore):
        self.store = store
        self.pending = ""
        self.last_emitted_char = ""
        self.discard_invalid_target = False
        self.discard_bare_id = False
        self.text_mode = None
        self.text_parts = []
        self.text_bytes = 0

    def push(self, text: str, *, final: bool = False) -> str:
        if self.text_mode is None and (text or self.pending):
            self.text_mode = self.store.has_text_records()
        if self.text_mode:
            self.text_bytes += len(text.encode("utf-8"))
            if self.text_bytes > _MAX_TEXT_OUTPUT:
                raise ValueError("reference message exceeds 8 MiB")
            self.text_parts.append(text)
            if not final:
                return ""
            result = _display_references("".join(self.text_parts), self.store)
            self.text_parts = []
            self.text_bytes = 0
            return result
        if self.discard_bare_id:
            boundary = re.search(r"[^A-Za-z0-9_-]", text)
            if boundary is None:
                return ""
            text = text[boundary.start():]
            self.discard_bare_id = False
        if self.discard_invalid_target:
            boundary = re.search(r"[\[\)\n]", text)
            if boundary is None:
                return ""
            text = text[boundary.end():] if boundary.group() == ")" else text[boundary.start():]
            self.discard_invalid_target = False
        self.pending += text
        output = []
        while self.pending:
            marker = _STREAM_START.search(self.pending)
            if marker is None:
                held = 0 if final else max(
                    (size for size in range(1, min(8, len(self.pending)) + 1)
                     if self.pending[-size:].lower() in _STREAM_PREFIXES), default=0)
                output.append(self.pending[:len(self.pending) - held])
                self.pending = self.pending[len(self.pending) - held:]
                break
            output.append(self.pending[:marker.start()])
            self.pending = self.pending[marker.start():]
            if self.pending.startswith("["):
                close_label = _label_close(self.pending)
                if close_label is None or close_label + 1 == len(self.pending):
                    if not final and len(self.pending) <= 502:
                        break
                    output.append(self.pending[0])
                    self.pending = self.pending[1:]
                    continue
                if self.pending[close_label + 1] != "(":
                    output.append("[")
                    self.pending = self.pending[1:]
                    continue
                target = self.pending[close_label + 2:]
                if not "ref:".startswith(target.lower()) and not target.lower().startswith("ref:"):
                    output.append("[")
                    self.pending = self.pending[1:]
                    continue
                end = self.pending.find(")", close_label + 2)
                if end < 0:
                    if not final and len(self.pending) <= 2550:
                        break
                    boundary = re.search(r"[\[\n]", target)
                    if boundary is None:
                        self.discard_invalid_target = not final
                        output.append(resolve_references(self.pending, self.store))
                        self.pending = ""
                    else:
                        offset = close_label + 2 + boundary.start()
                        output.append(resolve_references(self.pending[:offset], self.store))
                        self.pending = self.pending[offset:]
                    continue
                piece = self.pending[:end + 1]
                self.pending = self.pending[end + 1:]
            else:
                previous = next((part[-1] for part in reversed(output) if part),
                                self.last_emitted_char)
                if previous and not previous.isspace():
                    output.append(self.pending[0])
                    self.pending = self.pending[1:]
                    continue
                boundary = re.search(r"[^A-Za-z0-9_-]", self.pending[4:])
                boundary_at = boundary.start() + 4 if boundary is not None else None
                if boundary is None:
                    if not final and len(self.pending) <= 260:
                        break
                    if not final:
                        output.append(_UNBOUND)
                        self.pending = ""
                        self.discard_bare_id = True
                        break
                    piece, self.pending = self.pending, ""
                else:
                    piece = self.pending[:boundary_at]
                    self.pending = self.pending[boundary_at:]
            output.append(resolve_references(piece, self.store))
        emitted = "".join(output)
        if emitted:
            self.last_emitted_char = emitted[-1]
        return emitted

    def reset(self) -> None:
        self.pending = ""
        self.last_emitted_char = ""
        self.discard_invalid_target = False
        self.discard_bare_id = False
        self.text_mode = None
        self.text_parts = []
        self.text_bytes = 0


def _current_store() -> ThreadDataStore | None:
    try:
        path = ((get_config() or {}).get("configurable") or {}).get("thread_datastore_path")
    except RuntimeError:
        return None
    return ThreadDataStore(path) if isinstance(path, str) else None


def _read_records(kind: str | None, limit: int, query: str | None = None) -> str:
    store = _current_store()
    if store is None:
        return "Thread data is unavailable in this conversation."
    if kind is not None and (not isinstance(kind, str) or not 1 <= len(kind) <= 64):
        return "Kind must contain 1 to 64 characters."
    if not isinstance(limit, int) or not 1 <= limit <= 20:
        return "Limit must be between 1 and 20."
    if query is not None and (not isinstance(query, str) or not 1 <= len(query) <= 200):
        return "Search query must contain 1 to 200 characters."
    rows = store.list(kind, limit) if query is None else store.search(query, kind, limit)
    for row in rows:
        if len(row["value"]) > 2000:
            row["value"] = row["value"][:2000]
            row["truncated"] = True
    return json.dumps(rows, ensure_ascii=False)


def list_thread_data(kind: str | None = None, limit: int = 20) -> str:
    """List recent thread text records by kind. Get a record to read a truncated value in full."""
    return _read_records(kind, limit)


def put_thread_data(value: str, title: str = "") -> str:
    """Keep important exact text from user/tool context as a new immutable record.

    Use for values whose spelling matters across reuse: identifiers, names,
    quotations or exact passages. Copy the returned reference inline in replies
    or file text, including sentences/table cells; do not reconstruct its ID.
    Save generic references through write_file/edit_file: values bind before
    effects. Saved-file reads/finalization bind sources only; raw generic drafts
    outside literal URLs fail instead of expanding. Do not write unbound
    generic references with execute.
    For edit_file with references, BOTH exact old_string and new_string must
    include their surrounding newline bytes. Read the saved file first.
    For URL/pipe table edits, include the table header/context; an ambiguous
    fragment is rejected. Literal values can avoid reference syntax in edits.
    Source URLs from retrieval already register automatically; use source_link
    for citations. This tool never verifies facts or creates source authority.
    Values are at most 65536 characters. Unescaped ref: is reserved; explicit
    backslashes shield literal tokens and remain in output without decoding.
    """
    store = _current_store()
    if store is None:
        return "Thread data is unavailable in this conversation."
    try:
        if not isinstance(value, str) or len(value) > 65536:
            raise ValueError("value must be text with at most 65536 characters")
        _check_text_value(value)
        record_id = store.put_text(value, title=title)
    except ValueError as error:
        return f"Invalid exact text: {error}"
    return json.dumps({"id": record_id, "kind": "text", "title": title,
                       "reference": f"ref:{record_id}"}, ensure_ascii=False)


def search_thread_data(query: str, kind: str | None = None, limit: int = 20) -> str:
    """Find thread text records by literal substring. Get a record for its full value."""
    return _read_records(kind, limit, query)


def get_thread_data(record_id: str) -> str:
    """Read one thread text record by its ID."""
    store = _current_store()
    if store is None:
        return "Thread data is unavailable in this conversation."
    row = store.get(record_id)
    return json.dumps(row, ensure_ascii=False) if row else "Thread data record not found."


class ReferenceBindingMiddleware(AgentMiddleware):
    """Bind generic file text eagerly; capture sources and bind outward refs."""

    def __init__(self, *, bind_file_refs: bool = False):
        self.bind_file_refs = bind_file_refs

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        prepared = self._bind_arguments(request)
        if isinstance(prepared, ToolMessage):
            return prepared
        return self._bind_result(prepared, handler(prepared))

    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        prepared = self._bind_arguments(request)
        if isinstance(prepared, ToolMessage):
            return prepared
        return self._bind_result(prepared, await handler(prepared))

    def _bind_arguments(self, request):
        name = request.tool_call.get("name")
        if name not in {"write_file", "edit_file"}:
            return request
        store = _current_store()
        if store is None:
            return request
        args = request.tool_call.get("args") or {}
        keys = ("content",) if name == "write_file" else ("old_string", "new_string")
        replacements = {}
        try:
            for key in keys:
                value = args.get(key)
                if isinstance(value, str):
                    resolved = resolve_references(value, store, text_only=True, edit_context=name == "edit_file")
                    if resolved != value:
                        replacements[key] = resolved
            if name == "edit_file" and replacements:
                if not all(isinstance(args.get(key), str)
                           and args[key].startswith("\n") and args[key].endswith("\n")
                           for key in keys):
                    raise ValueError("reference-bearing edits need the exact surrounding newline in BOTH old_string and new_string")
        except ValueError as error:
            return ToolMessage(content=f"Text reference error: {error}", status="error",
                               name=name, tool_call_id=request.tool_call["id"])
        return (request.override(tool_call={**request.tool_call, "args": {**args, **replacements}})
                if replacements else request)

    def _bind_result(self, request: ToolCallRequest, result):
        store = _current_store()
        if store is None or not isinstance(result, ToolMessage) or result.status == "error":
            return result
        name = request.tool_call.get("name")
        if name == "search_internet" and isinstance(result.content, str):
            try:
                rows = ast.literal_eval(result.content)
            except (ValueError, SyntaxError):
                return result
            if not isinstance(rows, list):
                return result
            bound = []
            for row in rows:
                if not isinstance(row, dict):
                    return result
                record_id = register_source(store, row.get("url"), str(row.get("title") or ""))
                item = {key: value for key, value in row.items()
                        if key not in {"source_id", "source_link"}}
                if record_id:
                    item.update(source_id=f"ref:{record_id}",
                                source_link=f"[Source](ref:{record_id})")
                bound.append(item)
            return result.model_copy(update={"content": str(bound)})
        if name == "read_url" and isinstance(result.content, str) and not result.content.startswith("Error fetching URL:"):
            url = (request.tool_call.get("args") or {}).get("url")
            record_id = register_source(store, url)
            if record_id:
                return result.model_copy(update={"content":
                                                 f"Source link: [Source](ref:{record_id})\n{result.content}"})
        if self.bind_file_refs and name == "read_file" and isinstance(result.content, str):
            bound_sources: list[str] = []
            try:
                resolved = resolve_file_references(result.content, store,
                                                  bound_sources=bound_sources)
            except ValueError as error:
                return result.model_copy(update={"content": f"File reference error: {error}", "status": "error"})
            if resolved != result.content:
                from assist.middleware.url_provenance import urls_in_text
                original_urls = urls_in_text(result.content)
                artifact = result.artifact if isinstance(result.artifact, dict) else {}
                artifact = {**artifact, BOUND_FILE_URLS_ARTIFACT: tuple(dict.fromkeys(
                    (*original_urls, *bound_sources)))}
                if bound_sources:
                    resolved += "\n\nExact source URLs for fact-checking:\n" + "\n".join(
                        dict.fromkeys(bound_sources))
                return result.model_copy(update={"content": resolved,
                                                 "artifact": artifact})
        return result

    def after_model(self, state, runtime):
        store = _current_store()
        messages = state.get("messages") or []
        if store is None or not messages or not isinstance(messages[-1], AIMessage):
            return None
        last = messages[-1]
        if not isinstance(last.content, str):
            return None
        resolved = _display_references(last.content, store)
        if resolved == last.content:
            return None
        return {"messages": [last.model_copy(update={"content": resolved})]}
