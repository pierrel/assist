"""Encrypted repository credentials with an unlock cached outside sandbox mounts."""

from __future__ import annotations

import base64
import json
import os
import re
import stat
import threading
import weakref
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id


FILE = ".assist-secrets.json"
KEY_FILE = "secret-unlock.json"
REQUEST_FILE = "secret-request.json"
MAX_FILE = 128 * 1024
MAX_ENTRIES = 64
_ID = re.compile(r"[0-9a-f]{32}\Z")
_CHECK = b"assist-secret-store-v1"
_KDF_SLOTS = threading.BoundedSemaphore(2)
_REQUEST_MAP_LOCK = threading.Lock()
_REQUEST_LOCKS = weakref.WeakValueDictionary()


def _request_lock(directory: Path):
    with _REQUEST_MAP_LOCK:
        key = str(directory)
        lock = _REQUEST_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _REQUEST_LOCKS[key] = lock
        return lock


class SecretStoreError(ValueError):
    """A fixed error category safe to show without submitted values."""


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _utf8(value: str) -> bytes:
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise SecretStoreError("Invalid secret text") from error


def _unb64(value: object, length: int | None = None) -> bytes:
    if not isinstance(value, str) or len(value) > MAX_FILE:
        raise SecretStoreError("Secret store is invalid")
    try:
        raw = base64.b64decode(value, validate=True)
    except ValueError as error:
        raise SecretStoreError("Secret store is invalid") from error
    if length is not None and len(raw) != length:
        raise SecretStoreError("Secret store is invalid")
    return raw


def login_page(url: str) -> str:
    """Public exact HTTPS origin and path, excluding query and fragment."""
    if not isinstance(url, str) or len(url) > 2048:
        raise SecretStoreError("Invalid login page")
    _utf8(url)
    try:
        parsed = urlsplit(url)
        port = parsed.port
        host = parsed.hostname
    except ValueError as error:
        raise SecretStoreError("Invalid login page") from error
    if (parsed.scheme != "https" or not host or parsed.username is not None
            or parsed.password is not None or (port is not None and not 1 <= port <= 65535)
            or not url.isprintable() or any(c.isspace() for c in url)):
        raise SecretStoreError("Invalid login page")
    try:
        host = host.encode("idna").decode("ascii").lower()
    except UnicodeError as error:
        raise SecretStoreError("Invalid login page") from error
    if not re.fullmatch(r"[a-z0-9.-]{1,253}", host):
        raise SecretStoreError("Invalid login page")
    origin = f"https://{host}" + (f":{port}" if port not in (None, 443) else "")
    return origin + (parsed.path or "/")


def _aad(entry: dict) -> bytes:
    return json.dumps({key: entry[key] for key in ("id", "label", "login_page")},
                      sort_keys=True, separators=(",", ":")).encode()


def _read_regular(directory: Path, name: str, limit: int) -> bytes | None:
    root = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=root)
        except FileNotFoundError:
            return None
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit or info.st_nlink != 1:
                raise SecretStoreError("Secret store is invalid")
            data = os.read(fd, limit + 1)
            if len(data) > limit:
                raise SecretStoreError("Secret store is invalid")
            return data
        finally:
            os.close(fd)
    finally:
        os.close(root)


def _replace(directory: Path, name: str, data: bytes) -> None:
    root = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    temporary = ".secret-" + uuid4().hex
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=root)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, name, src_dir_fd=root, dst_dir_fd=root)
        os.fsync(root)
    finally:
        try:
            os.unlink(temporary, dir_fd=root)
        except FileNotFoundError:
            pass
        os.close(root)


class SecretStore:
    def __init__(self, repository: str | Path, thread_dir: str | Path):
        self.repository = Path(repository)
        self.thread_dir = Path(thread_dir)

    def pending_request(self) -> dict | None:
        data = _read_regular(self.thread_dir, REQUEST_FILE, 4096)
        if data is None:
            return None
        try:
            value = json.loads(data)
            if (not isinstance(value, dict)
                    or set(value) != {"id", "label", "login_page"}
                    or not isinstance(value["id"], str)
                    or not _ID.fullmatch(value["id"])
                    or not isinstance(value["label"], str)
                    or not 1 <= len(value["label"]) <= 80
                    or not value["label"].isprintable()
                    or login_page(value["login_page"]) != value["login_page"]):
                raise ValueError
            _utf8(value["label"])
            return value
        except (ValueError, KeyError, TypeError) as error:
            raise SecretStoreError("Private entry request is invalid") from error

    def request_input(self, label: str, page: str) -> dict:
        if (not isinstance(label, str) or not 1 <= len(label) <= 80
                or not label.isprintable()):
            raise SecretStoreError("Invalid private entry request")
        _utf8(label)
        request = {"id": uuid4().hex, "label": label,
                   "login_page": login_page(page)}
        encoded = json.dumps(request, ensure_ascii=False).encode()
        if len(encoded) > 4096:
            raise SecretStoreError("Invalid private entry request")
        with _request_lock(self.thread_dir):
            _replace(self.thread_dir, REQUEST_FILE, encoded)
        return request

    @contextmanager
    def selected_request(self, request_id: str | None, label: str, page: str):
        """Bind an agent-requested private form to its current public destination."""
        if request_id is None:
            yield
            return
        if not isinstance(request_id, str) or not _ID.fullmatch(request_id):
            raise SecretStoreError("Private entry request changed; refresh")
        with _request_lock(self.thread_dir):
            pending = self.pending_request()
            if (pending is None or pending["id"] != request_id
                    or pending["label"] != label
                    or pending["login_page"] != login_page(page)):
                raise SecretStoreError("Private entry request changed; refresh")
            yield
            (self.thread_dir / REQUEST_FILE).unlink()

    def _catalog(self) -> dict | None:
        data = _read_regular(self.repository, FILE, MAX_FILE)
        if data is None:
            return None
        try:
            value = json.loads(data)
            if (not isinstance(value, dict) or set(value) != {"version", "salt", "check", "entries"}
                    or value["version"] != 1 or not isinstance(value["entries"], list)
                    or len(value["entries"]) > MAX_ENTRIES):
                raise ValueError
            _unb64(value["salt"], 16)
            check = value["check"]
            if not isinstance(check, dict) or set(check) != {"nonce", "data"}:
                raise ValueError
            _unb64(check["nonce"], 12)
            _unb64(check["data"])
            for entry in value["entries"]:
                if (not isinstance(entry, dict)
                        or set(entry) != {"id", "label", "login_page", "nonce", "data"}
                        or not isinstance(entry["id"], str) or not _ID.fullmatch(entry["id"])
                        or not isinstance(entry["label"], str) or len(entry["label"]) > 80
                        or not isinstance(entry["login_page"], str)
                        or login_page(entry["login_page"]) != entry["login_page"]):
                    raise ValueError
                _utf8(entry["label"])
                _unb64(entry["nonce"], 12)
                _unb64(entry["data"])
            if len({entry["id"] for entry in value["entries"]}) != len(value["entries"]):
                raise ValueError
            return value
        except (ValueError, KeyError, TypeError) as error:
            raise SecretStoreError("Secret store is invalid") from error

    def _write(self, catalog: dict) -> None:
        encoded = json.dumps(catalog, separators=(",", ":"), ensure_ascii=False).encode()
        if len(encoded) > MAX_FILE:
            raise SecretStoreError("Secret store is full")
        _replace(self.repository, FILE, encoded)

    @staticmethod
    def _derive(passphrase: str, salt: bytes) -> bytes:
        if not isinstance(passphrase, str):
            raise SecretStoreError("Unlock phrase must be 12 to 1024 bytes")
        encoded = _utf8(passphrase)
        if not 12 <= len(encoded) <= 1024:
            raise SecretStoreError("Unlock phrase must be 12 to 1024 bytes")
        if not _KDF_SLOTS.acquire(blocking=False):
            raise SecretStoreError("Secret unlock busy")
        try:
            return Argon2id(salt=salt, length=32, iterations=2, lanes=1,
                            memory_cost=64 * 1024).derive(encoded)
        finally:
            _KDF_SLOTS.release()

    @staticmethod
    def _verify(catalog: dict, key: bytes) -> None:
        try:
            check = catalog["check"]
            if AESGCM(key).decrypt(_unb64(check["nonce"], 12),
                                   _unb64(check["data"]), _CHECK) != _CHECK:
                raise InvalidTag
        except (InvalidTag, ValueError) as error:
            raise SecretStoreError("Unlock phrase is incorrect") from error

    def _cached(self) -> dict | None:
        data = _read_regular(self.thread_dir, KEY_FILE, 4096)
        if data is None:
            return None
        try:
            info = os.stat(self.thread_dir / KEY_FILE, follow_symlinks=False)
            value = json.loads(data)
            if (info.st_uid != os.getuid() or info.st_mode & 0o077
                    or not isinstance(value, dict)
                    or set(value) != {"version", "salt", "key", "seen"}
                    or value["version"] != 1 or not isinstance(value["seen"], list)
                    or len(value["seen"]) > MAX_ENTRIES
                    or any(not isinstance(ident, str) or not _ID.fullmatch(ident)
                           for ident in value["seen"])
                    or len(set(value["seen"])) != len(value["seen"])):
                raise ValueError
            _unb64(value["salt"], 16)
            _unb64(value["key"], 32)
            return value
        except (ValueError, KeyError, TypeError) as error:
            raise SecretStoreError("Thread secret unlock is invalid") from error

    def _cache(self, catalog: dict, key: bytes, seen: list[str]) -> None:
        cached = self._cached()
        current = {item["id"] for item in catalog["entries"]}
        if cached is not None:
            if (cached["salt"] != catalog["salt"]
                    or _unb64(cached["key"], 32) != key
                    or not set(cached["seen"]) <= current):
                raise SecretStoreError("Secret store changed; repair required")
            seen = list(dict.fromkeys([*cached["seen"], *seen]))
        if not set(seen) <= current:
            raise SecretStoreError("Secret store changed; repair required")
        if cached is not None and seen == cached["seen"]:
            return
        _replace(self.thread_dir, KEY_FILE,
                 json.dumps({"version": 1, "salt": catalog["salt"],
                             "key": _b64(key), "seen": seen}).encode())

    def unlock(self, passphrase: str, *, create: bool = False) -> None:
        catalog = self._catalog()
        if catalog is None:
            if not create:
                raise SecretStoreError("No secret store in this repository")
            if self._cached() is not None:
                raise SecretStoreError("Secret store changed; repair required")
            salt = os.urandom(16)
            key = self._derive(passphrase, salt)
            nonce = os.urandom(12)
            catalog = {"version": 1, "salt": _b64(salt),
                       "check": {"nonce": _b64(nonce),
                                 "data": _b64(AESGCM(key).encrypt(nonce, _CHECK, _CHECK))},
                       "entries": []}
            self._write(catalog)
        else:
            key = self._derive(passphrase, _unb64(catalog["salt"], 16))
            self._verify(catalog, key)
            for entry in catalog["entries"]:
                self._decrypt(entry, key)
        self._cache(catalog, key, [item["id"] for item in catalog["entries"]])

    def _key_for(self, catalog: dict | None) -> bytes:
        if catalog is None:
            if self._cached() is not None:
                raise SecretStoreError("Secret store changed; repair required")
            raise SecretStoreError("No secret store in this repository")
        cached = self._cached()
        if cached is None:
            raise SecretStoreError("Unlock this repository in this thread")
        if (cached["salt"] != catalog["salt"]
                or not set(cached["seen"]) <= {item["id"] for item in catalog["entries"]}):
            raise SecretStoreError("Secret store changed; repair required")
        key = _unb64(cached["key"], 32)
        self._verify(catalog, key)
        return key

    def key(self) -> bytes:
        return self._key_for(self._catalog())

    def list_public(self) -> list[dict]:
        catalog = self._catalog()
        if catalog is None:
            return []
        return [{key: item[key] for key in ("id", "label", "login_page")}
                for item in catalog["entries"]]

    @staticmethod
    def validate_entry(label: str, page: str, username: str, password: str) -> str:
        if (not isinstance(label, str) or not 1 <= len(label) <= 80
                or not isinstance(username, str) or not isinstance(password, str)):
            raise SecretStoreError("Invalid secret entry")
        if len(_utf8(username)) > 320 or not 1 <= len(_utf8(password)) <= 1024:
            raise SecretStoreError("Invalid secret entry")
        _utf8(label)
        return login_page(page)

    def put(self, label: str, page: str, username: str, password: str) -> str:
        page = self.validate_entry(label, page, username, password)
        catalog = self._catalog()
        key = self._key_for(catalog)
        if len(catalog["entries"]) >= MAX_ENTRIES:
            raise SecretStoreError("Secret store is full")
        entry = {"id": uuid4().hex, "label": label, "login_page": page,
                 "nonce": _b64(os.urandom(12))}
        payload = json.dumps({"username": username, "password": password},
                             separators=(",", ":")).encode()
        entry["data"] = _b64(AESGCM(key).encrypt(
            _unb64(entry["nonce"], 12), payload, _aad(entry)))
        catalog["entries"].append(entry)
        self._write(catalog)
        self._cache(catalog, key, [entry["id"]])
        return entry["id"]

    @staticmethod
    def _decrypt(entry: dict, key: bytes) -> dict:
        try:
            payload = json.loads(AESGCM(key).decrypt(
                _unb64(entry["nonce"], 12), _unb64(entry["data"]), _aad(entry)))
            if (not isinstance(payload, dict) or set(payload) != {"username", "password"}
                    or not isinstance(payload["username"], str)
                    or not isinstance(payload["password"], str)):
                raise ValueError
            return payload
        except (InvalidTag, ValueError, TypeError) as error:
            raise SecretStoreError("Secret record is invalid") from error

    def get(self, secret_id: str) -> dict:
        if not isinstance(secret_id, str) or not _ID.fullmatch(secret_id):
            raise SecretStoreError("Unknown secret")
        catalog = self._catalog()
        key = self._key_for(catalog)
        entry = next((item for item in catalog["entries"] if item["id"] == secret_id), None)
        if entry is None:
            raise SecretStoreError("Unknown secret")
        payload = self._decrypt(entry, key)
        self._cache(catalog, key, [secret_id])
        return {"id": entry["id"], "label": entry["label"],
                "login_page": entry["login_page"], **payload}

    def private_values(self) -> list[str]:
        catalog = self._catalog()
        key = self._key_for(catalog)
        payloads = [self._decrypt(entry, key) for entry in catalog["entries"]]
        self._cache(catalog, key, [entry["id"] for entry in catalog["entries"]])
        return [value for payload in payloads
                for value in (payload["username"], payload["password"]) if value]
