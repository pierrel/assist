"""Private SMS history, independent of threads and their Git workspaces."""
from __future__ import annotations

import os
from pathlib import Path
import re
import sqlite3
import stat
from datetime import datetime, timezone
import uuid
from contextlib import contextmanager


ID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
STATUSES = frozenset({"received", "unknown", "sent", "not_sent"})
MAX_TEXT_BYTES = 16 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SmsArchive:
    """Store immutable message identity and monotonic observed outcomes."""

    def __init__(self, directory: str | Path, *, thread_root: str | Path):
        raw = Path(directory)
        if not raw.is_absolute() or raw == Path(raw.anchor) or ".." in raw.parts:
            raise ValueError("SMS archive requires a private absolute directory")
        for part in (raw, *raw.parents):
            if part.is_symlink():
                raise ValueError("SMS archive path must not contain symlinks")
            marker = part / ".git"
            if (marker.is_file() or marker.is_symlink()
                    or (marker / "HEAD").is_file() and (marker / "objects").is_dir()
                    or (part / "HEAD").is_file() and (part / "objects").is_dir()):
                raise ValueError("SMS archive must be outside every Git repository")
        self.directory = raw.resolve()
        if self.directory.is_relative_to(Path(thread_root).resolve()):
            raise ValueError("SMS archive must be outside the thread root")
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.directory.stat()
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("SMS archive directory must be owned by the service and mode 0700")
        self.path = self.directory / "messages.sqlite3"
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            info = self.path.lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                    or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600):
                raise ValueError("SMS archive database must be a private regular file")
        else:
            os.close(descriptor)
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY, direction TEXT NOT NULL, number TEXT NOT NULL,
                    text TEXT NOT NULL, status TEXT NOT NULL, occurred_at TEXT NOT NULL,
                    recorded_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    thread_id TEXT, prompt TEXT, run_id TEXT, admitted INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS sms_correspondent ON messages(number, occurred_at);
                CREATE INDEX IF NOT EXISTS sms_time ON messages(occurred_at);
            """)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=2)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA synchronous=FULL")
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _validate(record_id: str, direction: str, number: str, text: str, status: str) -> None:
        if not isinstance(record_id, str) or not ID_RE.fullmatch(record_id):
            raise ValueError("invalid SMS record id")
        if direction not in {"incoming", "outgoing"} or status not in STATUSES:
            raise ValueError("invalid SMS direction or status")
        if (direction == "incoming") != (status == "received"):
            raise ValueError("SMS status does not match direction")
        if not isinstance(number, str) or not re.fullmatch(r"[+0-9A-Za-z]{1,20}", number):
            raise ValueError("invalid SMS correspondent")
        if not isinstance(text, str) or "\0" in text or len(text.encode("utf-8")) > MAX_TEXT_BYTES:
            raise ValueError("invalid SMS body")

    def record(self, record_id: str, direction: str, number: str, text: str,
               status: str, *, occurred_at: str | None = None,
               handling: tuple[str, str] | None = None) -> tuple[dict, bool]:
        """Commit a receipt; retries cannot change its content or regress its outcome.

        A new outgoing ``unknown`` record is the write-before-effect claim.
        Only its creator may attempt the send. Incoming handling is fixed on
        first receipt and carries a persisted Run id before acknowledgement.
        """
        self._validate(record_id, direction, number, text, status)
        if occurred_at is not None:
            parsed = datetime.fromisoformat(occurred_at)
            if parsed.utcoffset() is None:
                raise ValueError("SMS timestamp requires a time zone")
            occurred_at = parsed.astimezone(timezone.utc).isoformat()
        now = utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute("SELECT * FROM messages WHERE id=?", (record_id,)).fetchone()
            if previous is not None:
                row = dict(previous)
                if (row["direction"], row["number"], row["text"]) != (direction, number, text):
                    raise ValueError("SMS id conflicts with its immutable message")
                if occurred_at is not None and occurred_at != row["occurred_at"]:
                    raise ValueError("SMS id conflicts with its observed timestamp")
                if row["status"] in {"sent", "not_sent"} and status in {"sent", "not_sent"} and row["status"] != status:
                    raise ValueError("SMS id conflicts with a known terminal outcome")
                if row["status"] == "unknown" and status in {"sent", "not_sent"}:
                    connection.execute("UPDATE messages SET status=?,updated_at=? WHERE id=?", (status, now, record_id))
                created = False
            else:
                thread_id, prompt = handling if handling else (None, None)
                run_id = uuid.uuid4().hex if handling else None
                connection.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?,?,0)",
                                   (record_id, direction, number, text, status, occurred_at or now,
                                    now, now, thread_id, prompt, run_id))
                created = True
            row = dict(connection.execute("SELECT * FROM messages WHERE id=?", (record_id,)).fetchone())
        return row, created

    def mark_admitted(self, record_id: str) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE messages SET admitted=1 WHERE id=?", (record_id,))

    @staticmethod
    def _public(row) -> dict:
        value = {key: row[key] for key in ("id", "direction", "number", "text", "status", "occurred_at", "recorded_at", "updated_at")}
        if row["direction"] == "incoming":
            value["handling"] = ("archive_only" if row["thread_id"] is None
                                 else "admitted" if row["admitted"] else "pending")
        return value

    def read(self, record_id: str) -> dict | None:
        if not isinstance(record_id, str) or not ID_RE.fullmatch(record_id):
            raise ValueError("invalid SMS record id")
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM messages WHERE id=?", (record_id,)).fetchone()
        return self._public(row) if row else None

    def search(self, *, number: str | None = None, text: str | None = None,
               direction: str | None = None, status: str | None = None,
               after: str | None = None, before: str | None = None, limit: int = 20) -> list[dict]:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 20:
            raise ValueError("SMS search limit must be 1 to 20")
        if direction is not None and direction not in {"incoming", "outgoing"}:
            raise ValueError("invalid SMS direction")
        if status is not None and status not in STATUSES:
            raise ValueError("invalid SMS status")
        conditions, parameters = [], []
        for column, value in (("number", number), ("direction", direction), ("status", status)):
            if value is not None:
                if not isinstance(value, str) or len(value) > 128:
                    raise ValueError("invalid SMS search filter")
                conditions.append(column + "=?")
                parameters.append(value)
        for operator, value in ((">=", after), ("<", before)):
            if value is not None:
                parsed = datetime.fromisoformat(value)
                if parsed.utcoffset() is None:
                    raise ValueError("SMS search time requires a time zone")
                conditions.append("occurred_at" + operator + "?")
                parameters.append(parsed.astimezone(timezone.utc).isoformat())
        if text is not None:
            if not isinstance(text, str) or len(text) > 256:
                raise ValueError("SMS text search is limited to 256 characters")
            conditions.append("instr(text,?)>0")
            parameters.append(text)
        query = "SELECT * FROM messages"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY occurred_at DESC,id DESC LIMIT ?"
        parameters.append(limit)
        with self._connect() as connection:
            remaining = 2000
            def stop():
                nonlocal remaining
                remaining -= 1
                return remaining <= 0
            connection.set_progress_handler(stop, 1000)
            rows = connection.execute(query, parameters).fetchall()
        values = []
        for row in rows:
            value = self._public(row)
            value["text"] = value["text"][:240]
            values.append(value)
        return values


def configured_archive() -> SmsArchive:
    directory = os.getenv("ASSIST_SMS_ARCHIVE_DIR")
    if not directory:
        raise ValueError("SMS archive is not configured (ASSIST_SMS_ARCHIVE_DIR)")
    return SmsArchive(directory, thread_root=os.getenv("ASSIST_THREADS_DIR", "/tmp/assist_threads"))
