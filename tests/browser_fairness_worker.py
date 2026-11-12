"""Synthetic network-isolated Chromium worker for the scheduling regression.

This fixture replaces production browser startup, not its DOM/action code. It
uses a local HTTP form and a background writer to prove Docker freezing.
"""
import importlib.util
import os
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit


class Page(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/result"):
            query = parse_qs(urlsplit(self.path).query)["query"][0]
            with open("/workspace/submissions", "a") as output:
                output.write(query + "\n")
            body = ("<h1>Found " + query + "</h1>").encode()
        else:
            body = (b'<form action="/result"><input name="query" placeholder="Artist">'
                    b'<input type="submit" value="SEARCH"></form>')
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def writer():
    count = 0
    while True:
        Path("/workspace/counter").write_text(str(count))
        count += 1
        time.sleep(0.01)


spec = importlib.util.spec_from_file_location("runner", "/candidate/assist/browser/runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def start(worker):
    if worker.context is None:
        worker.playwright = runner.sync_playwright().start()
        worker.browser = worker.playwright.chromium.launch(
            executable_path="/usr/bin/chromium", headless=True,
            args=["--no-sandbox"])
        worker.context = worker.browser.new_context()
        worker.context.on("page", worker._register_page)


runner.BrowserWorker._start = start
# The fixture owns startup for the eager production worker too.
runner.BrowserWorker.start_engine = lambda _worker: None
threading.Thread(target=ThreadingHTTPServer(("127.0.0.1", 8000), Page).serve_forever,
                 daemon=True).start()
threading.Thread(target=writer, daemon=True).start()
runner.serve()
