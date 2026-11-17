"""Real HTTP/Unix relay admission and cancellation without a model."""
import http.server
import threading

import httpx
import pytest

from assist.hermes_provider_relay import HermesProviderRelay
from assist.thread_queue import ThreadAffinityQueue


@pytest.mark.parametrize("response_started", [False, True])
def test_cross_thread_admission_and_stop_drains_active_provider(tmp_path, response_started):
    reached = threading.Event()
    release = threading.Event()
    exited = threading.Event()

    class Provider(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            if response_started:
                self.send_response(200)
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(b"first")
                self.wfile.flush()
            reached.set()
            release.wait(10)

    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    queue = ThreadAffinityQueue(hold_timeout_s=30, wait_timeout_s=2)
    relay = None
    try:
        with queue.acquire("a") as handle:
            relay = HermesProviderRelay(
                tmp_path, f"http://127.0.0.1:{upstream.server_port}/v1", "fixture",
                "fixture", 131072,
                lambda: queue.current_handle() is handle and not handle.expired)
            relay.start()

            def request():
                try:
                    with httpx.Client(transport=httpx.HTTPTransport(uds=str(relay.socket_path)),
                                      timeout=10, trust_env=False) as client:
                        client.post("http://relay/v1/chat/completions", json={"model": "fixture"})
                finally:
                    exited.set()

            worker = threading.Thread(target=request, daemon=True)
            worker.start()
            assert reached.wait(3), "HTTP worker must see the globally owned queue handle"
            assert relay.connections
            relay.close()
            assert not relay.connections
            assert exited.wait(2), "Stop must wake a blocked provider reader"
        with queue.acquire("b"):
            assert not relay.connections
    finally:
        release.set()
        if relay is not None:
            relay.close()
        upstream.shutdown()
        upstream.server_close()


def test_provider_fence_denies_request_without_native_hook(tmp_path):
    relay = HermesProviderRelay(tmp_path, "http://127.0.0.1:1/v1", "fixture",
                                "fixture", 131072, lambda: False)
    relay.start()
    try:
        with httpx.Client(transport=httpx.HTTPTransport(uds=str(relay.socket_path)),
                          timeout=3, trust_env=False) as client:
            response = client.post("http://relay/v1/chat/completions", json={"model": "fixture"})
        assert response.status_code == 409
        assert not relay.connections
    finally:
        relay.close()
