"""Observed browser targets remain bound to their live DOM semantics."""
import json
from types import SimpleNamespace

import pytest

from assist.browser import runner
from assist.browser.runner import BrowserInputError, BrowserWorker, http_url


class _Page:
    url = "https://example.com/"

    def __init__(self, closed=False):
        self.closed = closed

    def is_closed(self):
        return self.closed

    def on(self, *_args):
        pass


class _Link:
    def __init__(self, href, name="Manual", tag="a", kind=None):
        self.href = href
        self.name = name
        self.tag = tag
        self.kind = kind
        self.clicked = False
        self.filled = False

    def is_visible(self):
        return True

    def get_attribute(self, name):
        return {"href": self.href, "aria-label": self.name,
                "type": self.kind}.get(name)

    def evaluate(self, _expression):
        return self.tag

    def inner_text(self):
        return self.name

    def click(self, **_kwargs):
        self.clicked = True

    def fill(self, *_args, **_kwargs):
        self.filled = True


def _observed(worker, element):
    worker.pages["page"] = _Page()
    worker.snapshots["page"] = "snapshot"
    worker.targets["page"] = {"observed": (
        element, *worker._target_state(element, _Page.url))}


@pytest.mark.parametrize("selector", [
    {"ref": "observed"},
    {"href": "https://example.com/original"},
    {"role": "link", "name": "Manual"},
])
def test_changed_observed_link_fails_for_every_target_selector(selector):
    worker = BrowserWorker()
    link = _Link("/changed")
    _observed(worker, link)
    worker.targets["page"]["observed"] = (
        link, "link", "Manual", "https://example.com/original", "/original",
        worker.targets["page"]["observed"][-1])

    with pytest.raises(BrowserInputError, match="observed link changed"):
        worker.act("page", "snapshot", "click", selector)
    assert not link.clicked


def test_changed_button_name_cannot_be_clicked():
    worker = BrowserWorker()
    button = _Link(None, "Show report", tag="button")
    _observed(worker, button)
    button.name = "Delete account"
    with pytest.raises(BrowserInputError, match="target changed"):
        worker.act("page", "snapshot", "click", {"ref": "observed"})
    assert not button.clicked


def test_changed_field_attributes_cannot_be_filled():
    worker = BrowserWorker()
    field = _Link(None, "Search", tag="input", kind="text")
    _observed(worker, field)
    field.kind = "password"
    with pytest.raises(BrowserInputError, match="field changed"):
        worker.act("page", "snapshot", "fill",
                   {"ref": "observed", "text": "secret"})
    assert not field.filled


def test_absent_href_cannot_be_added_to_observed_link():
    worker = BrowserWorker()
    link = _Link(None)
    _observed(worker, link)
    link.href = "/delete"
    with pytest.raises(BrowserInputError, match="link changed"):
        worker.act("page", "snapshot", "click", {"ref": "observed"})
    assert not link.clicked


@pytest.mark.parametrize("url", ["http://@example.com/", "http://example.com:0/"])
def test_invalid_userinfo_and_port_zero(url):
    with pytest.raises(BrowserInputError):
        http_url(url)


def test_http_403_response_is_probeable_with_fragmented_proxy_headers(monkeypatch):
    worker = BrowserWorker()
    worker._response(SimpleNamespace(
        status=403, url="http://denied.example/",
        request=SimpleNamespace(resource_type="document")))
    assert "denied.example:80" in worker.failed_hosts

    class _Socket:
        parts = [b"HTTP/1.1 403 Forbidden\r\nX-Assist-Egress-",
                 b"Result: host_not_approved\r\nContent-Length: 0\r\n\r\n"]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def settimeout(self, *_args):
            pass

        def sendall(self, *_args):
            pass

        def recv(self, _size):
            return self.parts.pop(0)

    monkeypatch.setattr(runner.socket, "create_connection", lambda *_args, **_kwargs: _Socket())
    assert worker.probe("denied.example", 80) == {
        "host": "denied.example", "port": 80,
        "status": "403", "reason": "host_not_approved"}


def test_observation_budget_preserves_targets():
    class _Body:
        def aria_snapshot(self, **_kwargs):
            return "S" * 16000

    class _Indexed:
        def __init__(self, index):
            self.index = index

        def element_handle(self):
            return _Link("/" + str(self.index) + "x" * 3000,
                         name="\0" * 120)

    class _Locator:
        def count(self):
            return 60

        def nth(self, index):
            return _Indexed(index)

    class _RichPage(_Page):
        def wait_for_load_state(self, **_kwargs):
            pass

        def locator(self, selector):
            return _Body() if selector == "body" else _Locator()

    worker = BrowserWorker()
    worker.pages["page"] = _RichPage()
    for _ in range(12):
        worker._error("e" * 255, "document", "request_failed")
    observed = worker.observe("page")
    assert observed["targets"]
    assert observed["truncated"] is True
    assert len(json.dumps({"result": observed}).encode()) < runner.MAX_RESULT
    assert len(worker.targets["page"]) == len(observed["targets"])


def test_closed_pages_do_not_consume_concurrent_page_quota():
    worker = BrowserWorker()
    for index in range(5):
        worker.pages[str(index)] = _Page(closed=True)
    fresh = _Page()
    assert worker._register_page(fresh)
    assert len(worker.pages) == 1


def test_partial_playwright_launch_is_closed(monkeypatch):
    closed = []

    class _Chromium:
        def launch(self, **_kwargs):
            raise RuntimeError("launch failed")

    class _Driver:
        chromium = _Chromium()

        def stop(self):
            closed.append("driver")

    monkeypatch.setattr(runner, "sync_playwright",
                        lambda: SimpleNamespace(start=lambda: _Driver()))
    worker = BrowserWorker()
    with pytest.raises(RuntimeError, match="launch failed"):
        worker._start()
    assert closed == ["driver"]
    assert worker.playwright is worker.browser is worker.context is None


def test_failed_context_cleanup_reaches_driver_even_if_browser_close_raises(
        monkeypatch):
    closed = []

    class _Browser:
        def new_context(self, **_kwargs):
            raise RuntimeError("context failed")

        def close(self):
            closed.append("browser")
            raise RuntimeError("close failed")

    class _Driver:
        chromium = SimpleNamespace(launch=lambda **_kwargs: _Browser())

        def stop(self):
            closed.append("driver")

    monkeypatch.setattr(runner, "sync_playwright",
                        lambda: SimpleNamespace(start=lambda: _Driver()))
    worker = BrowserWorker()
    with pytest.raises(RuntimeError, match="context failed"):
        worker._start()
    assert closed == ["browser", "driver"]
    assert worker.playwright is worker.browser is worker.context is None


def test_ninth_download_is_cancelled_and_reported():
    worker = BrowserWorker()
    worker.downloads = {str(index): object() for index in range(8)}
    cancelled = []
    worker._register_download(SimpleNamespace(cancel=lambda: cancelled.append(True)))
    assert cancelled == [True]
    assert worker.errors[-1]["reason"] == "download_limit"
