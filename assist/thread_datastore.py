"""Host-owned thread records and source-reference binding."""

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


_BARE_REF = re.compile(r"(?<!\S)ref:([A-Za-z0-9_-]*)", re.I)
_STREAM_START = re.compile(r"\[|(?<!\S)ref:", re.I)
_LINK_TARGET = re.compile(r"\]\(ref:", re.I)
_LABEL_ATOM = r"(?:\\[^\n]|\[(?:\\[^\n]|[^\[\]\\\n])*\]|[^\[\]\\\n])"
_LABEL_OPEN = re.compile(r"\[(?:" + _LABEL_ATOM + r"){1,500}\]")
_STREAM_PREFIXES = {"r", "re", "ref"}
_RECORD_ID = re.compile(r"r[0-9a-f]{32}\Z")
_UNBOUND = "[source link unavailable: ask me to verify it again]"
BOUND_FILE_URLS_ARTIFACT = "assist_bound_file_urls"


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
    """Atomically bind one local research report through a no-follow dir handle."""
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
        resolved = resolve_references(original, store,
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


def _link_at(text: str, start: int, store: ThreadDataStore) -> tuple[int, str, str | None] | None:
    if _LINK_TARGET.search(text, start + 1, min(len(text), start + 507)) is None:
        return None
    close = _label_close(text, start)
    if close is None or text[close + 1:close + 6].lower() != "(ref:":
        return None
    label = text[start + 1:close]
    target_start = close + 6
    end = target_start
    while end < len(text) and text[end] not in "[)\n":
        end += 1
    if end < len(text) and text[end] == ")":
        url = source_url(store, text[target_start:end].lower())
        if url:
            # Markdown decodes entities and escapes, parses code spans, and
            # splits table cells at pipes inside link targets.
            # Encode outward bound destinations so the rendered href matches the record.
            target = (re.sub(r"&(?=(?:#[0-9]+|#[xX][0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]+);)",
                             "&amp;", url).replace("\\", "&#92;")
                     .replace("`", "&#96;").replace("|", "&#124;"))
            return end + 1, f"[{label}](<{target}>)", url
        return end + 1, f"{label} {_UNBOUND}", None
    return end, f"{label} {_UNBOUND}", None


def resolve_references(text: str, store: ThreadDataStore,
                       max_output_bytes: int | None = None,
                       *, bound_sources: list[str] | None = None) -> str:
    """Resolve explicit source refs; preserve ordinary text and URLs.

    ``max_output_bytes`` bounds UTF-8 output and raises on overflow.
    ``bound_sources`` collects exact host URLs for an internal fact-check read;
    outward Markdown keeps escaped targets for exact rendered destinations.
    """
    parts = []
    used = 0

    def append(piece: str) -> None:
        nonlocal used
        if max_output_bytes is not None:
            used += len(piece.encode("utf-8"))
            if used > max_output_bytes:
                raise ValueError("resolved research report exceeds 8 MiB source-binding limit")
        parts.append(piece)

    position = 0
    while marker := _STREAM_START.search(text, position):
        append(text[position:marker.start()])
        if text[marker.start()] == "[":
            link = _link_at(text, marker.start(), store)
            if link is None:
                append("[")
                position = marker.start() + 1
            else:
                position, replacement, bound_url = link
                append(replacement)
                if bound_sources is not None and bound_url is not None:
                    bound_sources.append(bound_url)
        else:
            bare = _BARE_REF.match(text, marker.start())
            assert bare is not None
            bound_url = source_url(store, bare.group(1).lower())
            append(bound_url or _UNBOUND)
            if bound_sources is not None and bound_url is not None:
                bound_sources.append(bound_url)
            position = bare.end()
    append(text[position:])
    return "".join(parts)


class ReferenceStream:
    """Hold incomplete explicit references until their source target is known."""

    def __init__(self, store: ThreadDataStore):
        self.store = store
        self.pending = ""
        self.last_emitted_char = ""
        self.discard_invalid_target = False
        self.discard_bare_id = False

    def push(self, text: str, *, final: bool = False) -> str:
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
    """Capture trusted retrieval URLs and resolve explicit model references."""

    def __init__(self, *, bind_file_refs: bool = False):
        self.bind_file_refs = bind_file_refs

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        return self._bind_result(request, handler(request))

    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        return self._bind_result(request, await handler(request))

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
            resolved = resolve_references(result.content, store,
                                          bound_sources=bound_sources)
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
        resolved = resolve_references(last.content, store)
        if resolved == last.content:
            return None
        return {"messages": [last.model_copy(update={"content": resolved})]}

