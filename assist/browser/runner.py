"""Typed Playwright worker reachable only by a Unix socket inside its sidecar."""
from __future__ import annotations

import json
import os
import re
import signal
import socket
import stat
import sys
import time
from urllib.parse import urljoin, urlsplit
from uuid import uuid4

from playwright.sync_api import sync_playwright

SOCKET = "/run/browser.sock"
MAX_COMMAND = 32768
MAX_RESULT = 65536
SENSITIVE = re.compile(r"password|passcode|one.time|otp|card|cvv|payment", re.I)


class BrowserInputError(Exception):
    pass


def http_url(url: str) -> str:
    if not isinstance(url, str) or len(url) > 4096:
        raise BrowserInputError("invalid URL")
    try:
        parsed = urlsplit(url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or (parsed.port is not None and not 1 <= parsed.port <= 65535)):
            raise ValueError
    except ValueError as error:
        raise BrowserInputError("only HTTP(S) URLs without userinfo are supported") from error
    return url


def host_port(url: str) -> str:
    try:
        parsed = urlsplit(url)
        return f"{parsed.hostname}:{parsed.port or (443 if parsed.scheme == 'https' else 80)}"
    except ValueError:
        return "<invalid-host>"


def recv_limited(conn, limit):
    chunks = []
    size = 0
    while True:
        chunk = conn.recv(min(4096, limit + 1 - size))
        if not chunk:
            return b"".join(chunks)
        size += len(chunk)
        if size > limit:
            raise BrowserInputError("browser message exceeds limit")
        chunks.append(chunk)


class BrowserWorker:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.pages = {}
        self.page_ids = {}
        self.web_pages = set()
        self.snapshots = {}
        self.targets = {}
        self.downloads = {}
        self.errors = []
        self.failed_hosts = set()

    def _start(self):
        if self.context is not None:
            return
        try:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(
                headless=True, chromium_sandbox=True, downloads_path="/downloads",
                proxy={"server": "http://assist-egress-proxy:8888", "bypass": ""})
            self.context = self.browser.new_context(
                accept_downloads=True, service_workers="block")
            self.context.set_default_timeout(10000)
            self.context.on("page", self._register_page)
        except Exception:
            for resource, method in ((self.context, "close"),
                                     (self.browser, "close"),
                                     (self.playwright, "stop")):
                if resource is not None:
                    try:
                        getattr(resource, method)()
                    except Exception:
                        pass
            self.context = self.browser = self.playwright = None
            raise

    def _prune_pages(self):
        for ident, page in list(self.pages.items()):
            if page.is_closed():
                del self.pages[ident]
                self.page_ids.pop(page, None)
                self.web_pages.discard(page)
                self.snapshots.pop(ident, None)
                self.targets.pop(ident, None)

    def _register_page(self, page):
        self._prune_pages()
        if page in self.page_ids:
            return self.page_ids[page]
        if len(self.pages) >= 5:
            page.close()
            self._error("<popup>", "document", "page_limit")
            return None
        page_id = uuid4().hex[:12]
        self.pages[page_id] = page
        self.page_ids[page] = page_id
        page.on("download", self._register_download)
        page.on("requestfailed", self._request_failed)
        page.on("response", self._response)
        page.on("framenavigated", lambda frame: self._check_navigation(page, frame))
        if page.url != "about:blank":
            self._check_page_url(page)
        return page_id

    def _check_page_url(self, page):
        try:
            http_url(page.url)
        except BrowserInputError:
            if not page.is_closed():
                page.close()
            self._prune_pages()
            raise BrowserInputError("non-HTTP page navigation is unsupported") from None
        self.web_pages.add(page)

    def _check_navigation(self, page, frame):
        if frame == page.main_frame and (page.url != "about:blank"
                                         or page in self.web_pages):
            try:
                self._check_page_url(page)
            except BrowserInputError:
                self._error("<page>", "document", "non_http_navigation")

    def _register_download(self, download):
        if len(self.downloads) < 8:
            self.downloads[uuid4().hex[:12]] = download
        else:
            self._error("<download>", "download", "download_limit")
            download.cancel()

    def _request_failed(self, request):
        host = host_port(request.url)
        self.failed_hosts.add(host)
        self._error(host, request.resource_type, "request_failed")

    def _response(self, response):
        if response.status >= 400:
            host = host_port(response.url)
            self.failed_hosts.add(host)
            self._error(host, response.request.resource_type,
                        f"http_{response.status}")

    def _error(self, host, resource, reason):
        self.errors.append({"host": host[:255], "resource": resource[:30],
                            "reason": reason[:40]})
        self.errors = self.errors[-20:]

    def _page(self, page_id):
        page = self.pages.get(page_id)
        if page is None or page.is_closed():
            raise BrowserInputError("unknown or closed page ID")
        return page

    @staticmethod
    def _target_state(handle, base_url):
        tag = handle.evaluate("element => element.tagName.toLowerCase()")
        role = handle.get_attribute("role") or {
            "a": "link", "button": "button", "input": "textbox",
            "textarea": "textbox"}.get(tag, "button")
        name = (handle.get_attribute("aria-label")
                or handle.get_attribute("placeholder")
                or handle.evaluate("element => element.labels?.[0]?.innerText || ''")
                or handle.inner_text() or "")[:120]
        raw_href = handle.get_attribute("href")
        href = urljoin(base_url, raw_href) if raw_href is not None else None
        fill_attrs = tuple(handle.get_attribute(attr) for attr in (
            "type", "name", "id", "autocomplete", "placeholder", "aria-label"))
        return role, name, href, raw_href, fill_attrs

    @staticmethod
    def _result_size(result):
        return len(json.dumps({"result": result}, ensure_ascii=False).encode())

    def _observe(self, page_id):
        page = self._page(page_id)
        if page.url == "about:blank" and page not in self.web_pages:
            return {
                "page_id": page_id, "snapshot_id": uuid4().hex[:12],
                "url": "about:blank", "snapshot": "", "targets": [],
                "pages": [{"page_id": ident, "url": item.url[:256]}
                          for ident, item in self.pages.items() if not item.is_closed()],
                "downloads": [], "network_errors": self.errors[-12:],
            }
        self._check_page_url(page)
        try:
            page.wait_for_load_state("domcontentloaded", timeout=3000)
        except Exception:
            pass
        observation_errors = []
        try:
            snapshot = page.locator("body").aria_snapshot(timeout=3000)[:16000]
        except Exception:
            snapshot = ""
            observation_errors.append("snapshot_unavailable")
        snapshot_id = uuid4().hex[:12]
        targets = {}
        described = []
        try:
            locator = page.locator(
                "a,button,input,textarea,[role=button],[role=link]")
            candidates = min(locator.count(), 60)
        except Exception:
            candidates = 0
            observation_errors.append("targets_unavailable")
        for index in range(candidates):
            try:
                handle = locator.nth(index).element_handle()
                if handle is None:
                    continue
                if not handle.is_visible():
                    continue
                role, name, href, raw_href, fill_attrs = self._target_state(
                    handle, page.url)
                ref = uuid4().hex[:10]
                targets[ref] = (handle, role, name, href, raw_href, fill_attrs)
                described.append({"ref": ref, "role": role, "name": name,
                                  **({"href": href[:1024]} if href and role == "link" else {})})
            except Exception:
                continue
            if len(described) >= 40:
                break
        self._page(page_id)
        if candidates and not described:
            observation_errors.append("targets_unavailable")
        result = {
            "page_id": page_id, "snapshot_id": snapshot_id,
            "url": page.url[:512], "snapshot": "",
            "targets": [],
            "pages": [{"page_id": ident, "url": item.url[:256]}
                      for ident, item in self.pages.items() if not item.is_closed()],
            "downloads": [{"download_id": ident,
                           "name": value.suggested_filename[:180]}
                          for ident, value in self.downloads.items()],
            "network_errors": self.errors[-12:],
        }
        if observation_errors:
            result["observation_errors"] = observation_errors
        while snapshot and self._result_size({**result, "snapshot": snapshot}) > 32768:
            snapshot = snapshot[:len(snapshot) // 2]
            result["truncated"] = True
        result["snapshot"] = snapshot
        kept = {}
        for target in described:
            result["targets"].append(target)
            if self._result_size(result) > MAX_RESULT - 512:
                result["targets"].pop()
                result["truncated"] = True
                break
            kept[target["ref"]] = targets[target["ref"]]
        self._check_page_url(page)
        self.snapshots[page_id] = snapshot_id
        self.targets[page_id] = kept
        return result

    def open(self, url, reuse_page_id=None):
        http_url(url)
        self._start()
        self._prune_pages()
        if reuse_page_id is not None:
            if not isinstance(reuse_page_id, str) or not reuse_page_id:
                raise BrowserInputError("invalid page ID to reuse")
            page_id = reuse_page_id
            page = self._page(page_id)
            self.snapshots.pop(page_id, None)
            self.targets.pop(page_id, None)
        else:
            if len(self.pages) >= 5:
                raise BrowserInputError("browser page limit reached")
            page = self.context.new_page()
            page_id = self._register_page(page)
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=10000)
        except Exception:
            host = host_port(url)
            self.failed_hosts.add(host)
            self._error(host, "document", "navigation_failed")
            try:
                http_url(page.url)
            except BrowserInputError:
                if not page.is_closed():
                    page.close()
                self._prune_pages()
                return {"page_id": page_id, "snapshot_id": uuid4().hex[:12],
                        "url": url[:512], "snapshot": "", "targets": [],
                        "pages": [], "downloads": [],
                        "network_errors": self.errors[-12:],
                        "observation_errors": ["navigation_failed"]}
        return self._observe(page_id)

    def close(self, page_id):
        page = self._page(page_id)
        page.close()
        self._prune_pages()
        live = [{"page_id": ident, "url": item.url[:256]}
                for ident, item in self.pages.items()]
        return {"closed_page_id": page_id, "pages": live,
                "active_page_id": live[-1]["page_id"] if live else None}

    def observe(self, page_id):
        return self._observe(page_id)

    def act(self, page_id, snapshot_id, action, target):
        page = self._page(page_id)
        if self.snapshots.get(page_id) != snapshot_id:
            raise BrowserInputError("stale page observation")
        if action not in {"click", "fill"} or not isinstance(target, dict):
            raise BrowserInputError("unsupported browser action")
        ref, href = target.get("ref"), target.get("href")
        role, name = target.get("role"), target.get("name")
        if ref:
            selected = self.targets.get(page_id, {}).get(ref)
            if selected is None:
                raise BrowserInputError("unknown or stale target reference")
            locator, role, name, observed_href, raw_href, fill_attrs = selected
        elif href:
            matches = [item for item in self.targets.get(page_id, {}).values()
                       if item[1] == "link" and item[3] == href]
            if len(matches) != 1:
                raise BrowserInputError("href is absent or ambiguous; use an observed ref")
            locator, role, name, observed_href, raw_href, fill_attrs = matches[0]
        elif isinstance(role, str) and isinstance(name, str):
            matches = [item for item in self.targets.get(page_id, {}).values()
                       if item[1:3] == (role, name)]
            if len(matches) != 1:
                raise BrowserInputError("role/name target is absent or ambiguous")
            locator, role, name, observed_href, raw_href, fill_attrs = matches[0]
        else:
            raise BrowserInputError("give an observed ref, exact href, or role and name")
        if not locator.is_visible():
            raise BrowserInputError("observed target is no longer visible")
        live_role, live_name, live_href, live_raw_href, live_attrs = self._target_state(
            locator, page.url)
        if (live_role, live_name) != (role, name):
            raise BrowserInputError("observed target changed")
        if (live_href, live_raw_href) != (observed_href, raw_href):
            raise BrowserInputError("observed link changed")
        if action == "fill" and live_attrs != fill_attrs:
            raise BrowserInputError("observed field changed")
        if action == "click" and live_href is not None:
            http_url(live_href)
        if action == "fill":
            if role not in {"textbox", "searchbox"}:
                raise BrowserInputError("only nonsecret text fields can be filled")
            text = target.get("text")
            if not isinstance(text, str) or len(text) > 2000:
                raise BrowserInputError("fill text exceeds the limit")
            attributes = " ".join((locator.get_attribute(attr) or "") for attr in (
                "type", "name", "id", "autocomplete", "placeholder", "aria-label"))
            if SENSITIVE.search(attributes + " " + (name or "")):
                raise BrowserInputError("secret, payment and one-time-code fields are unsupported")
            locator.fill(text, timeout=10000)
        else:
            before = set(self.pages)
            locator.click(timeout=10000)
            new_pages = [ident for ident in self.pages if ident not in before]
            if new_pages:
                return self._observe(new_pages[-1])
        return self._observe(page_id)

    def wait(self, page_id, role, name):
        page = self._page(page_id)
        if not isinstance(role, str) or not isinstance(name, str):
            raise BrowserInputError("wait requires a role and exact name")
        page.get_by_role(role, name=name, exact=True).wait_for(
            state="visible", timeout=10000)
        return self._observe(page_id)

    def download_info(self, download_id):
        download = self.downloads.get(download_id)
        if download is None:
            raise BrowserInputError("unknown download ID")
        path = download.path()
        if not path or not os.path.realpath(path).startswith("/downloads/"):
            raise BrowserInputError("download is unavailable")
        stat = os.stat(path)
        if not os.path.isfile(path) or stat.st_size > 20 * 1024 * 1024:
            raise BrowserInputError("download exceeds the 20 MiB transfer limit")
        return {"path": str(path), "name": download.suggested_filename[:180],
                "size": stat.st_size}

    def probe(self, host, port):
        if (not isinstance(host, str) or not isinstance(port, int)
                or not 1 <= port <= 65535 or re.search(r"\s", host)):
            raise BrowserInputError("invalid probe target")
        host = host.lower()
        if f"{host}:{port}" not in self.failed_hosts:
            raise BrowserInputError("only an observed failed host can be probed")
        with socket.create_connection(("assist-egress-proxy", 8888), timeout=3) as conn:
            conn.settimeout(3)
            conn.sendall(f"CONNECT {host}:{port} HTTP/1.1\r\n\r\n".encode())
            head = b""
            while b"\r\n\r\n" not in head and len(head) < 4096:
                chunk = conn.recv(min(1024, 4096 - len(head)))
                if not chunk:
                    break
                head += chunk
        if b"\r\n\r\n" not in head:
            raise BrowserInputError("incomplete proxy probe response")
        head = head.decode("latin-1", errors="replace")
        match = re.search(r"^X-Assist-Egress-Result: ([a-z_]+)\r?$", head, re.M)
        first_line = head.partition("\r\n")[0].split(" ")
        if len(first_line) < 2 or not first_line[1].isdigit():
            raise BrowserInputError("invalid proxy probe response")
        status = first_line[1]
        return {"host": host, "port": port, "status": status,
                "reason": match.group(1) if match else "unknown"}

    def handle(self, command):
        if not isinstance(command, dict) or command.get("session") != os.environ.get(
                "BROWSER_SESSION_TOKEN"):
            raise BrowserInputError("invalid browser session")
        operation = command.get("operation")
        args = command.get("args") or {}
        if not isinstance(args, dict):
            raise BrowserInputError("invalid browser arguments")
        methods = {"open": self.open, "close": self.close,
                   "observe": self.observe, "act": self.act,
                   "wait": self.wait, "download_info": self.download_info,
                   "probe": self.probe}
        if operation not in methods:
            raise BrowserInputError("unsupported browser operation")
        return methods[operation](**args)


def serve():
    with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as stream:
        if stream.read().strip() != os.environ["BROWSER_BOOT_ID"]:
            os._exit(124)
    deadline_ns = int(os.environ["BROWSER_DEADLINE_NS"])
    ttl = (deadline_ns - time.clock_gettime_ns(time.CLOCK_BOOTTIME)) / 1_000_000_000
    if ttl <= 0 or ttl > 240:
        os._exit(124)
    signal.signal(signal.SIGALRM, lambda _signum, _frame: os._exit(124))
    signal.setitimer(signal.ITIMER_REAL, ttl)
    worker = BrowserWorker()
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(SOCKET)
        os.chmod(SOCKET, 0o600)
        server.listen(1)
        while True:
            conn, _ = server.accept()
            with conn:
                try:
                    payload = recv_limited(conn, MAX_COMMAND)
                    answer = {"result": worker.handle(json.loads(payload))}
                except BrowserInputError as error:
                    answer = {"error": str(error)}
                except Exception:
                    # Playwright errors can contain URL paths and form content.
                    answer = {"error": "browser operation failed"}
                encoded = json.dumps(answer, ensure_ascii=False).encode()
                if len(encoded) > MAX_RESULT:
                    encoded = b'{"error":"browser observation exceeds limit"}'
                conn.sendall(encoded)


def call():
    payload = sys.stdin.buffer.read(MAX_COMMAND + 1)
    if len(payload) > MAX_COMMAND:
        raise SystemExit(2)
    with socket.socket(socket.AF_UNIX) as conn:
        conn.settimeout(18)
        conn.connect(SOCKET)
        conn.sendall(payload)
        conn.shutdown(socket.SHUT_WR)
        result = recv_limited(conn, MAX_RESULT)
    if not result:
        raise SystemExit(2)
    sys.stdout.buffer.write(result)


def export(path):
    """Stream one completed tmpfs download; Docker cp cannot read this mount."""
    if (not path.startswith("/downloads/") or os.path.realpath(path) != path
            or "/" in path.removeprefix("/downloads/")):
        raise SystemExit(2)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            size = os.fstat(stream.fileno())
            if not stat.S_ISREG(size.st_mode) or size.st_size > 20 * 1024 * 1024:
                raise SystemExit(2)
            while chunk := stream.read(65536):
                sys.stdout.buffer.write(chunk)
    except OSError:
        raise SystemExit(2) from None


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "serve":
        serve()
    elif len(sys.argv) == 2 and sys.argv[1] == "call":
        call()
    elif len(sys.argv) == 3 and sys.argv[1] == "export":
        export(sys.argv[2])
    else:
        raise SystemExit(2)
