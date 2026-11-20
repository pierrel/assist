"""Typed Playwright worker reachable only by a private Unix socket."""
from __future__ import annotations

import json
import base64
import math
import os
import re
import socket
import stat
import subprocess
import tempfile
import time
import sys
from time import monotonic
from urllib.parse import quote, urljoin, urlsplit
from uuid import uuid4

from playwright.sync_api import sync_playwright
if __package__:
    from .redaction import MAX_PRIVATE_COMMAND, contains_hidden, replace_hidden
else:
    from browser_redaction import MAX_PRIVATE_COMMAND, contains_hidden, replace_hidden

RUNTIME_DIR = os.environ.get("BROWSER_RUNTIME_DIR", "/run")
SOCKET = os.path.join(RUNTIME_DIR, "browser.sock")
DOWNLOAD_DIR = os.environ.get("BROWSER_DOWNLOAD_DIR", "/downloads")
PROXY_HOST = os.environ.get("BROWSER_PROXY_HOST", "assist-egress-proxy")
PROXY_PORT = int(os.environ.get("BROWSER_PROXY_PORT", "8888"))
MAX_COMMAND = MAX_PRIVATE_COMMAND
MAX_RESULT = 65536
MAX_VIEWER_RESULT = 512 * 1024
MAX_STORAGE = 48 * 1024
MAX_RAW_TEXT = 128 * 1024
MAX_CURRENT_PASSWORDS = 64 * 1024
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


def origin_target(url: str) -> tuple[str, str, int]:
    """Reduce an HTTP URL to a policy target without retaining path or query."""
    http_url(url)
    parsed = urlsplit(url)
    host = parsed.hostname.encode("idna").decode("ascii").lower()
    if not re.fullmatch(r"[a-z0-9.-]{1,253}", host):
        raise BrowserInputError("unsupported browser hostname")
    default_port = 443 if parsed.scheme == "https" else 80
    port = parsed.port or default_port
    origin = f"{parsed.scheme}://{host}"
    if port != default_port:
        origin += f":{port}"
    return origin, host, port


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
        self.requests = {}
        self.request_partial = set()
        self.initial_storage = None
        self.chromium = None
        self.revoked = False
        self._private_values = ()
        self._current_before = ()
        self._close_unobserved = False
        self.page_namespace = os.environ.get("BROWSER_GENERATION", uuid4().hex)[:16]

    def start_engine(self):
        """Attach to our blank native browser before the host seals its port."""
        if self.browser is not None or self.revoked:
            raise BrowserInputError("browser engine cannot restart")
        environment = {name: os.environ[name] for name in ("PATH", "HOME", "TMPDIR")}
        try:
            deadline = monotonic() + 8
            with tempfile.TemporaryFile(dir=RUNTIME_DIR) as startup_log:
                self.chromium = subprocess.Popen([
                    "/usr/bin/chromium", "--ozone-platform=headless", "--use-gl=angle",
                    "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--no-first-run",
                    "--no-default-browser-check", "--disable-background-networking",
                    "--disable-component-update", "--disable-sync", "--disable-extensions",
                    "--remote-debugging-address=127.0.0.1", "--remote-debugging-port=9222",
                    f"--user-data-dir={RUNTIME_DIR}/chromium", "--no-startup-window",
                    f"--proxy-server=http://{PROXY_HOST}:{PROXY_PORT}",
                    "--proxy-bypass-list=<-loopback>"], env=environment,
                    stdout=subprocess.DEVNULL, stderr=startup_log)
                while True:
                    startup_log.seek(0)
                    match = re.search(rb"DevTools listening on (ws://127\.0\.0\.1:9222/devtools/browser/[^\s]+)",
                                      startup_log.read(32768))
                    if self.chromium.poll() is not None or monotonic() >= deadline:
                        raise BrowserInputError("native browser startup failed")
                    if match:
                        break
                    time.sleep(0.02)
                self.playwright = sync_playwright().start()
                self.browser = self.playwright.chromium.connect_over_cdp(
                    match[1].decode("ascii"), timeout=5000)
        except Exception:
            self.revoke()
            raise

    def _start(self):
        if self.revoked or self.browser is None:
            raise BrowserInputError("browser engine is unavailable")
        if self.context is not None:
            return
        try:
            self.context = self.browser.new_context(
                accept_downloads=True, service_workers="block",
                **({"storage_state": self.initial_storage}
                   if self.initial_storage is not None else {}))
            self.context.set_default_timeout(10000)
            self.context.on("page", self._register_page)
        except Exception:
            self.revoke()
            raise

    def _prune_pages(self):
        for ident, page in list(self.pages.items()):
            if page.is_closed():
                del self.pages[ident]
                self.page_ids.pop(page, None)
                self.web_pages.discard(page)
                self.snapshots.pop(ident, None)
                self.targets.pop(ident, None)
                self.requests.pop(ident, None)
                self.request_partial.discard(ident)

    def _register_page(self, page):
        self._prune_pages()
        if page in self.page_ids:
            return self.page_ids[page]
        if len(self.pages) >= 5:
            page.close()
            self._error("<popup>", "document", "page_limit")
            return None
        page_id = f"{self.page_namespace}-{uuid4().hex[:12]}"
        self.pages[page_id] = page
        self.page_ids[page] = page_id
        self.requests[page_id] = {}
        page.on("request", lambda request: self._record_request(page_id, request))
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
        failure = request.failure
        reason = failure if isinstance(failure, str) and re.fullmatch(r"net::[A-Z0-9_]+", failure) else "request_failed"
        self._error(host, request.resource_type, reason)

    def _response(self, response):
        if response.status >= 400:
            host = host_port(response.url)
            self._error(host, response.request.resource_type,
                        f"http_{response.status}")

    def _record_request(self, page_id, request):
        try:
            origin, _, _ = origin_target(request.url)
        except (BrowserInputError, UnicodeError):
            return
        observed = self.requests.get(page_id)
        if observed is None:
            return
        if origin not in observed and len(observed) >= 32:
            self.request_partial.add(page_id)
            return
        record = observed.setdefault(origin, {"types": set(), "redirect": False})
        if len(record["types"]) < 3:
            record["types"].add(str(request.resource_type)[:30])
        if request.redirected_from is not None:
            record["redirect"] = True

    def _error(self, host, resource, reason):
        self.errors.append({"host": host, "resource": resource,
                            "reason": reason})
        self.errors = self.errors[-20:]

    def _page(self, page_id):
        page = self.pages.get(page_id)
        if page is None or page.is_closed():
            raise BrowserInputError("unknown or closed page ID")
        return page

    @staticmethod
    def _target_state(handle, base_url):
        tag = handle.evaluate("element => element.tagName.toLowerCase()")
        input_button = tag == "input" and (handle.get_attribute("type") or "").lower() in {
            "submit", "button", "reset"}
        role = handle.get_attribute("role") or {
            "a": "link", "button": "button", "input": "button" if input_button else "textbox",
            "textarea": "textbox"}.get(tag, "button")
        button_value = handle.evaluate("""element => {
            const value = element.value || '';
            return value.length <= 2000 ? value : null;
        }""") if input_button else None
        if input_button and button_value is None:
            raise BrowserInputError("input-button value exceeds the limit")
        name = (handle.get_attribute("aria-label")
                or handle.get_attribute("placeholder")
                or handle.evaluate("element => element.labels?.[0]?.innerText || ''")
                or button_value
                or handle.inner_text() or "")
        raw_href = handle.get_attribute("href")
        href = urljoin(base_url, raw_href) if raw_href is not None else None
        fill_attrs = tuple(handle.get_attribute(attr) for attr in (
            "type", "name", "id", "autocomplete", "placeholder", "aria-label"))
        return role, name, href, raw_href, fill_attrs, button_value

    @staticmethod
    def _result_size(result):
        return len(json.dumps({"result": result}, ensure_ascii=False).encode())

    def _observe(self, page_id):
        page = self._page(page_id)
        if page.url == "about:blank" and page not in self.web_pages:
            return self._finish({
                "page_id": page_id, "snapshot_id": uuid4().hex[:12],
                "url": "about:blank", "snapshot": "", "targets": [],
                "pages": [{"page_id": ident, "url": item.url}
                          for ident, item in self.pages.items() if not item.is_closed()],
                "downloads": [], "network_errors": self.errors[-12:],
            })
        self._check_page_url(page)
        try:
            page.wait_for_load_state("domcontentloaded", timeout=3000)
        except Exception:
            pass
        observation_errors = []
        try:
            snapshot = page.locator("body").aria_snapshot(timeout=3000)
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
                role, name, href, raw_href, fill_attrs, button_value = self._target_state(
                    handle, page.url)
                ref = uuid4().hex[:10]
                targets[ref] = (handle, role, name, href, raw_href, fill_attrs, button_value)
                described.append({"ref": ref, "role": role, "name": name,
                                  **({"href": href} if href and role == "link" else {})})
            except Exception:
                continue
            if len(described) >= 40:
                break
        self._page(page_id)
        if candidates and not described:
            observation_errors.append("targets_unavailable")
        result = {
            "page_id": page_id, "snapshot_id": snapshot_id,
            "url": page.url, "snapshot": snapshot,
            "targets": [],
            "pages": [{"page_id": ident, "url": item.url}
                      for ident, item in self.pages.items() if not item.is_closed()],
            "downloads": [{"download_id": ident,
                           "name": value.suggested_filename}
                          for ident, value in self.downloads.items()],
            "network_errors": self.errors[-12:],
        }
        if len(page.url) <= 8192:
            result["_source_url"] = page.url
        if observation_errors:
            result["observation_errors"] = observation_errors
        for target in described:
            result["targets"].append(target)
        self._check_page_url(page)
        self.snapshots[page_id] = snapshot_id
        self.targets[page_id] = targets
        return self._finish(result)

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
            self._error(host, "document", "navigation_failed")
            try:
                http_url(page.url)
            except BrowserInputError:
                if not page.is_closed():
                    page.close()
                self._prune_pages()
                return self._finish({"page_id": page_id, "snapshot_id": uuid4().hex[:12],
                        "url": url, "snapshot": "", "targets": [],
                        "pages": [], "downloads": [],
                        "network_errors": self.errors[-12:],
                        "observation_errors": ["navigation_failed"]})
        return self._observe(page_id)

    def close(self, page_id):
        page = self._page(page_id)
        page.close()
        self._prune_pages()
        live = [{"page_id": ident, "url": item.url}
                for ident, item in self.pages.items()]
        if self._close_unobserved:
            return {"closed_page_id": page_id}
        try:
            return self._finish({"closed_page_id": page_id, "pages": live,
                                 "active_page_id": live[-1]["page_id"] if live else None})
        except BrowserInputError:
            return {"closed_page_id": page_id}

    def observe(self, page_id):
        return self._observe(page_id)

    def viewer(self, action, page_id=None, **args):
        """Fixed viewer controls and JPEG frames through the host viewer route."""
        self._prune_pages()
        if action == "tabs":
            return self._finish({"pages": [{"page_id": ident, "url": page.url}
                              for ident, page in self.pages.items()
                              if page in self.web_pages and page.url.startswith(("http://", "https://"))]})
        page = self._page(page_id)
        self._check_page_url(page)
        if action == "frame":
            try:
                masks = self._screenshot_masks(page)
                for quality in (50, 25, 10):
                    data = page.screenshot(type="jpeg", quality=quality, timeout=5000,
                                           mask=masks, mask_color="#ff00ff")
                    if len(data) <= 350 * 1024:
                        break
                else:
                    raise BrowserInputError("page image exceeds viewer limit")
            except Exception:
                raise BrowserInputError("viewer frame redaction unavailable") from None
            size = page.viewport_size
            return {"image": base64.b64encode(data).decode("ascii"), **size}
        if action == "viewport":
            width, height = args.get("width"), args.get("height")
            if (type(width) is not int or type(height) is not int
                    or not 320 <= width <= 1024 or not 240 <= height <= 1024):
                raise BrowserInputError("invalid viewer dimensions")
            page.set_viewport_size({"width": width, "height": height})
        elif action == "click":
            x, y = args.get("x"), args.get("y")
            size = page.viewport_size
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in (x, y)) or not (
                    0 <= x <= size["width"] and 0 <= y <= size["height"]):
                raise BrowserInputError("invalid viewer position")
            page.mouse.click(x, y)
        elif action == "text":
            text = args.get("text")
            if not isinstance(text, str) or len(text) > 2000:
                raise BrowserInputError("viewer text exceeds limit")
            page.keyboard.insert_text(text)
        elif action == "key":
            key = args.get("key")
            if key not in {"Enter", "Tab", "Backspace", "Delete", "ArrowLeft", "ArrowRight",
                           "ArrowUp", "ArrowDown", "Home", "End", "Control+a"}:
                raise BrowserInputError("unsupported viewer key")
            page.keyboard.press(key)
        elif action == "scroll":
            delta = args.get("delta")
            if type(delta) not in (int, float) or not math.isfinite(delta) or abs(delta) > 1500:
                raise BrowserInputError("invalid viewer scroll")
            page.mouse.wheel(0, delta)
        else:
            raise BrowserInputError("unsupported viewer action")
        # Human interaction invalidates the agent's old semantic target binding.
        self.snapshots.pop(page_id, None)
        self.targets.pop(page_id, None)
        return {"applied": True}

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
            locator, role, name, observed_href, raw_href, fill_attrs, button_value = selected
        elif href:
            matches = [item for item in self.targets.get(page_id, {}).values()
                       if item[1] == "link" and item[3] == href]
            if len(matches) != 1:
                raise BrowserInputError("href is absent or ambiguous; use an observed ref")
            locator, role, name, observed_href, raw_href, fill_attrs, button_value = matches[0]
        elif isinstance(role, str) and isinstance(name, str):
            matches = [item for item in self.targets.get(page_id, {}).values()
                       if item[1:3] == (role, name)]
            if len(matches) != 1:
                raise BrowserInputError("role/name target is absent or ambiguous")
            locator, role, name, observed_href, raw_href, fill_attrs, button_value = matches[0]
        else:
            raise BrowserInputError("give an observed ref, exact href, or role and name")
        if not locator.is_visible():
            raise BrowserInputError("observed target is no longer visible")
        live_role, live_name, live_href, live_raw_href, live_attrs, live_value = self._target_state(
            locator, page.url)
        if (live_role, live_name, live_value) != (role, name, button_value):
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

    def fill_secret(self, page_id, snapshot_id, login_page, username_ref,
                    password_ref, username, password):
        """Fill bound login fields without returning a page observation."""
        page = self._page(page_id)
        if self.snapshots.get(page_id) != snapshot_id:
            raise BrowserInputError("stale page observation")
        def exact_page():
            parsed = urlsplit(page.url)
            if parsed.scheme != "https" or not parsed.hostname:
                return False
            host = parsed.hostname.encode("idna").decode("ascii").lower()
            actual = f"https://{host}" + (
                f":{parsed.port}" if parsed.port not in (None, 443) else "")
            # Chromium percent-encodes Unicode paths without decoding reserved separators.
            expected = quote(login_page, safe="/%:@!$&'()*+,;=-._~")
            return actual + (parsed.path or "/") == expected

        if not exact_page():
            raise BrowserInputError("credential page changed")
        if (not isinstance(password, str) or not isinstance(username, str)
                or not isinstance(password_ref, str) or not password_ref
                or username and not isinstance(username_ref, str)):
            raise BrowserInputError("invalid credential fields")

        def field(ref, allowed):
            selected = self.targets.get(page_id, {}).get(ref)
            if selected is None:
                raise BrowserInputError("unknown or stale credential field")
            locator, role, name, href, raw_href, attrs, button_value = selected
            if role != "textbox" or attrs[0] not in allowed or href is not None:
                raise BrowserInputError("credential field changed")
            if not locator.is_visible():
                raise BrowserInputError("credential field changed")
            live = self._target_state(locator, page.url)
            if live != selected[1:]:
                raise BrowserInputError("credential field changed")
            return locator

        password_field = field(password_ref, {"password"})
        password_state = self._target_state(password_field, page.url)
        username_field = field(username_ref, {"text", "email", None}) if username else None
        self.snapshots.pop(page_id, None)
        self.targets.pop(page_id, None)
        if username_field is not None:
            username_field.fill(username, timeout=10000)
            if not exact_page():
                raise BrowserInputError("credential page changed")
            if (not password_field.is_visible()
                    or self._target_state(password_field, page.url) != password_state):
                raise BrowserInputError("credential field changed")
        password_field.fill(password, timeout=10000)
        return {"filled": True}

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
        if not path or not os.path.realpath(path).startswith(RUNTIME_DIR + "/"):
            raise BrowserInputError("download is unavailable")
        info = os.stat(path)
        if not os.path.isfile(path) or info.st_size > 20 * 1024 * 1024:
            raise BrowserInputError("download exceeds the 20 MiB transfer limit")
        # CDP attachment uses Playwright's private artifact directory. Export
        # only a completed, size-checked download through the existing boundary.
        destination = os.path.join(DOWNLOAD_DIR, download_id)
        download.save_as(destination)
        hidden = tuple(sorted(set(self._private_values + self._current_before
                                  + tuple(self._password_values())),
                              key=len, reverse=True))
        if hidden:
            with open(destination, "rb") as stream:
                content = stream.read(20 * 1024 * 1024 + 1)
            if (len(content) > 20 * 1024 * 1024
                    or any(value.encode("utf-8") in content for value in hidden)
                    or (b"%" in content or b"+" in content)
                    and contains_hidden(content.decode("latin-1"), hidden)):
                os.unlink(destination)
                raise BrowserInputError("download contains a private value")
        return self._finish({"path": destination, "name": download.suggested_filename,
                             "size": info.st_size}, trusted_path=True)

    @staticmethod
    def _policy(host, port, deadline):
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise BrowserInputError("browser preflight budget exhausted")
        try:
            with socket.create_connection((PROXY_HOST, PROXY_PORT),
                                          timeout=min(1, remaining)) as conn:
                conn.settimeout(min(1, remaining))
                conn.sendall(f"POLICY {host}:{port} HTTP/1.1\r\n\r\n".encode())
                head = b""
                while b"\r\n\r\n" not in head and len(head) < 1024:
                    chunk = conn.recv(min(512, 1024 - len(head)))
                    if not chunk:
                        break
                    head += chunk
        except OSError as error:
            raise BrowserInputError("browser preflight proxy unavailable") from error
        if b"\r\n\r\n" not in head:
            raise BrowserInputError("incomplete browser preflight response")
        text = head.decode("latin-1")
        status = text.partition("\r\n")[0].split(" ")
        reason = re.search(r"^X-Assist-Egress-Result: ([a-z_]+)\r?$", text, re.M)
        if (len(status) < 2 or status[1] not in {"200", "403"}
                or reason is None or (status[1] == "200") !=
                (reason.group(1) == "allowed")):
            raise BrowserInputError("invalid browser preflight response")
        return {"allowed": status[1] == "200", "reason": reason.group(1)}

    def preflight(self, url, page_id=None):
        origin, host, port = origin_target(url)
        deadline = monotonic() + 8
        top = {"origin": origin, **self._policy(host, port, deadline)}
        result = {"top_level": top, "dependencies": [], "partial": True}
        if page_id is None or not top["allowed"]:
            return self._finish(result)
        page = self._page(page_id)
        try:
            page_origin = origin_target(page.url)[0]
        except BrowserInputError:
            return self._finish(result)
        result["observed_page_origin"] = page_origin
        if page_origin != origin:
            return self._finish(result)
        for observed_origin, record in self.requests.get(page_id, {}).items():
            if observed_origin == origin:
                continue
            try:
                _, observed_host, observed_port = origin_target(observed_origin)
                decision = self._policy(observed_host, observed_port, deadline)
            except BrowserInputError:
                result["truncated"] = True
                break
            result["dependencies"].append({
                "origin": observed_origin,
                "resource_types": sorted(record["types"]),
                "redirect_observed": record["redirect"],
                "source": "page_observation_untrusted", **decision})
        if page_id in self.request_partial:
            result["truncated"] = True
        return self._finish(result)

    def _password_values(self):
        values = []
        size = 0
        for page in self.pages.values():
            if page.is_closed():
                continue
            if page.url == "about:blank" and page not in self.web_pages:
                continue
            try:
                found = page.locator('input[type="password"]').evaluate_all(
                    "nodes => nodes.length <= 60 ? nodes.map(node => node.value).filter(Boolean) : null")
            except Exception as error:
                raise BrowserInputError("password fields unavailable") from error
            if (not isinstance(found, list)
                    or any(not isinstance(value, str) for value in found)):
                raise BrowserInputError("password fields unavailable")
            for value in found:
                size += len(value.encode("utf-8"))
                if size > MAX_CURRENT_PASSWORDS:
                    raise BrowserInputError("password fields exceed limit")
                values.append(value)
        return values

    def _screenshot_masks(self, page):
        """Native password masks for human frames; model text uses the shared inventory."""
        return [frame.locator('input[type="password"]') for frame in page.frames]

    def _hidden_values(self):
        return tuple(sorted(set(self._private_values + self._current_before
                                + tuple(self._password_values())), key=len, reverse=True))

    def _finish(self, result, *, trusted_path=False):
        hidden = self._hidden_values()
        if isinstance(result, dict) and "_source_url" in result:
            if contains_hidden(result["_source_url"], hidden):
                result.pop("_source_url")
        path = result.pop("path") if trusted_path else None
        masked = self._masked_result(result, hidden)
        if trusted_path:
            masked["path"] = path
        return self._limit_result(masked)

    @staticmethod
    def _masked_result(result, hidden):
        if isinstance(result, str):
            if len(result.encode("utf-8")) > MAX_RAW_TEXT:
                raise BrowserInputError("browser observation exceeds limit")
            return replace_hidden(result, hidden, "[password hidden]")
        if isinstance(result, list):
            return [BrowserWorker._masked_result(item, hidden) for item in result]
        if isinstance(result, dict):
            return {key: (item if key in {"page_id", "snapshot_id", "ref",
                                          "download_id", "active_page_id",
                                          "closed_page_id"}
                          else BrowserWorker._masked_result(item, hidden))
                    for key, item in result.items()}
        return result

    def _limit_result(self, result):
        def clipped(value, limit):
            if len(value) <= limit:
                return value
            marker = "[password hidden]"
            start = value.rfind(marker, 0, limit + len(marker) - 1)
            if 0 <= start < limit and start + len(marker) > limit:
                limit = start + len(marker)
            return value[:limit]

        if not isinstance(result, dict):
            return result
        if "url" in result:
            result["url"] = clipped(result["url"], 512)
        for page in result.get("pages", []):
            page["url"] = clipped(page["url"], 256)
        for download in result.get("downloads", []):
            download["name"] = clipped(download["name"], 180)
        if "name" in result and "path" in result:
            result["name"] = clipped(result["name"], 180)
        for error in result.get("network_errors", []):
            for key, limit in (("host", 255), ("resource", 30), ("reason", 40)):
                error[key] = clipped(error[key], limit)
        if "snapshot" not in result:
            return result
        for target in result.get("targets", []):
            target["name"] = clipped(target["name"], 120)
            if "href" in target:
                target["href"] = clipped(target["href"], 1024)
        snapshot = result["snapshot"]
        if len(snapshot) > 16000:
            snapshot = clipped(snapshot, 16000)
            result["truncated"] = True
        while snapshot and self._result_size({**result, "snapshot": snapshot,
                                              "targets": []}) > 32768:
            shorter = clipped(snapshot, len(snapshot) // 2)
            if shorter == snapshot:
                snapshot = ""
                break
            snapshot = shorter
            result["truncated"] = True
        result["snapshot"] = snapshot
        page_id = result.get("page_id")
        kept = self.targets.get(page_id, {})
        while (result.get("targets")
               and self._result_size(result) > MAX_RESULT - 512):
            target = result["targets"].pop()
            kept.pop(target["ref"], None)
            result["truncated"] = True
        if self._result_size(result) > MAX_RESULT - 512:
            raise BrowserInputError("browser observation exceeds limit")
        return result

    def handle(self, command):
        if not isinstance(command, dict) or command.get("session") != os.environ.get(
                "BROWSER_SESSION_TOKEN"):
            raise BrowserInputError("invalid browser session")
        operation = command.get("operation")
        args = command.get("args") or {}
        if not isinstance(args, dict):
            raise BrowserInputError("invalid browser arguments")
        private_values = command.get("private_values", [])
        if (not isinstance(private_values, list) or len(private_values) > 128
                or any(not isinstance(value, str) or len(value.encode("utf-8")) > 1024
                       for value in private_values)):
            raise BrowserInputError("invalid private browser metadata")
        methods = {"ready": self.ready, "open": self.open, "close": self.close,
                   "observe": self.observe, "act": self.act,
                   "fill_secret": self.fill_secret,
                   "wait": self.wait, "download_info": self.download_info,
                   "preflight": self.preflight, "load_storage": self.load_storage,
                   "storage_state": self.storage_state, "revoke": self.revoke,
                   "viewer": self.viewer}
        if operation not in methods:
            raise BrowserInputError("unsupported browser operation")
        if operation == "close":
            try:
                before = self._password_values()
            except BrowserInputError:
                before = []
                self._close_unobserved = True
        else:
            before = self._password_values() if operation in {
                "open", "observe", "act", "wait", "download_info", "preflight", "viewer"} else []
        self._private_values = tuple(private_values)
        self._current_before = tuple(before)
        try:
            return methods[operation](**args)
        finally:
            self._private_values = ()
            self._current_before = ()
            self._close_unobserved = False

    def ready(self):
        with open("/proc/self/status", encoding="ascii") as stream:
            status = {}
            for line in stream:
                name, separator, value = line.partition(":\t")
                if separator:
                    status[name] = value.strip()
        return {"pid": os.getpid(), "uid": status["Uid"].split(),
                "gid": status["Gid"].split(),
                "caps": [status[name] for name in (
                    "CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")],
                "no_new_privs": status["NoNewPrivs"],
                "seccomp": status["Seccomp"]}

    def load_storage(self, state):
        """Load an opaque bounded Playwright snapshot before opening a context."""
        if self.context is not None or not isinstance(state, dict):
            raise BrowserInputError("browser storage is already active or invalid")
        encoded = json.dumps(state, separators=(",", ":")).encode()
        if len(encoded) > MAX_STORAGE:
            raise BrowserInputError("browser storage exceeds limit")
        self.initial_storage = state
        return {"loaded": True}

    def storage_state(self):
        """Export bounded auth state to the host, never to the model."""
        if self.context is None:
            return None
        state = self.context.storage_state(indexed_db=False)
        if len(json.dumps(state, separators=(",", ":")).encode()) > MAX_STORAGE:
            raise BrowserInputError("browser storage exceeds limit")
        return state


    def revoke(self):
        """Close the engine permanently; only a new turn may launch another."""
        self.revoked = True
        try:
            for resource, method in ((self.context, "close"),
                                     (self.browser, "close"),
                                     (self.playwright, "stop")):
                if resource is not None:
                    try:
                        getattr(resource, method)()
                    except Exception:
                        pass
        finally:
            for process in (self.chromium,):
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=1)
            self.context = self.browser = self.playwright = None
            self.chromium = None
            self.pages.clear()
            self.snapshots.clear()
        return {"revoked": True}


def serve():
    if os.getpid() != 1:
        os._exit(124)
    worker = BrowserWorker()
    os.makedirs(DOWNLOAD_DIR, mode=0o700, exist_ok=True)
    worker.start_engine()
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(SOCKET)
        os.chmod(SOCKET, 0o600)
        server.listen(1)
        while True:
            conn, _ = server.accept()
            with conn:
                limit = MAX_RESULT
                try:
                    payload = recv_limited(conn, MAX_COMMAND)
                    command = json.loads(payload)
                    if isinstance(command, dict) and command.get("operation") == "viewer":
                        limit = MAX_VIEWER_RESULT
                    answer = {"result": worker.handle(command)}
                except BrowserInputError as error:
                    answer = {"error": str(error)}
                except Exception:
                    # Playwright errors can contain URL paths and form content.
                    answer = {"error": "browser operation failed"}
                encoded = json.dumps(answer, ensure_ascii=False).encode()
                if len(encoded) > limit:
                    encoded = b'{"error":"browser observation exceeds limit"}'
                conn.sendall(encoded)
                if worker.revoked:
                    # PID 1 exit kills every native descendant; the outer
                    # shell turn remains outside this namespace.
                    os._exit(0)


def call():
    payload = sys.stdin.buffer.read(MAX_COMMAND + 1)
    if len(payload) > MAX_COMMAND:
        raise SystemExit(2)
    with socket.socket(socket.AF_UNIX) as conn:
        conn.settimeout(18)
        conn.connect(SOCKET)
        conn.sendall(payload)
        conn.shutdown(socket.SHUT_WR)
        command = json.loads(payload)
        result = recv_limited(conn, MAX_VIEWER_RESULT if command.get("operation") == "viewer" else MAX_RESULT)
    if not result:
        raise SystemExit(2)
    sys.stdout.buffer.write(result)


def export(path):
    """Stream one completed tmpfs download; Docker cp cannot read this mount."""
    prefix = DOWNLOAD_DIR + "/"
    if (not path.startswith(prefix) or os.path.realpath(path) != path
            or "/" in path.removeprefix(prefix)):
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
