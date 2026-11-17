"""A retained Hermes generation's private provider and parking boundary.

Native hooks coordinate safe points. Only this host relay grants provider
access; a hook failure cannot grant a request while the generation is parked.
"""
from __future__ import annotations

import http.client
from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import socket
import socketserver
import threading
from urllib.parse import urlsplit


_MAX_BODY = 2 * 1024 * 1024


class HermesRelayError(RuntimeError):
    pass


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def __init__(self, *args):
        self.slots = threading.BoundedSemaphore(32)
        super().__init__(*args)

    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()


class HermesProviderRelay:
    """Forward native model bodies to one host-selected local endpoint."""

    def __init__(self, directory: Path, upstream_url: str, api_key: str,
                 model: str, context_length: int, admitted):
        endpoint = urlsplit(upstream_url)
        if (endpoint.scheme not in {"http", "https"}
                or endpoint.hostname not in {"127.0.0.1", "::1", "localhost"}
                or endpoint.username or endpoint.password or endpoint.query
                or endpoint.fragment or endpoint.path.rstrip("/") != "/v1"):
            raise HermesRelayError("Hermes requires the configured local model endpoint")
        self.endpoint, self.api_key, self.model = endpoint, api_key, model
        self.context_length, self.admitted = context_length, admitted
        self.condition = threading.Condition()
        self.operations: dict[str, str] = {}
        self.connections: dict[http.client.HTTPConnection, socket.socket | None] = {}
        self.pausing = self.parked = self.stopped = False
        self.changed = threading.Event()
        self.result: dict | None = None
        self.started = False
        self.socket_path = directory / "provider.sock"
        handler = type("HermesProviderHandler", (_Handler,), {"relay": self})
        self.server = _Server(str(self.socket_path), handler)
        os.chmod(self.socket_path, 0o600)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       name="hermes-provider-relay", daemon=True)

    def start(self):
        self.thread.start()
        self.started = True

    def boundary(self, value: dict):
        operation, kind = value.get("operation"), value.get("kind")
        if not isinstance(operation, str) or not operation or len(operation) > 256:
            raise HermesRelayError("Invalid native operation")
        if kind not in {"model", "auxiliary", "tool"}:
            raise HermesRelayError("Invalid native operation kind")
        with self.condition:
            # An active model/tool may need nested auxiliary calls to finish.
            # New outer operations wait, so that finite active set can drain.
            while self.pausing and not self.stopped and not (
                    kind == "auxiliary" and any(
                        active != "auxiliary" for active in self.operations.values())):
                self.condition.wait()
            if self.stopped:
                raise HermesRelayError("Hermes generation has stopped")
            if operation in self.operations:
                raise HermesRelayError("Duplicate native operation")
            # Joined delegation waits for children; counting that wait would
            # prevent those children from parking before the parent returns.
            if not (kind == "tool" and value.get("tool") == "delegate_task"):
                self.operations[operation] = kind
            self.changed.set()

    def settle(self, operation):
        with self.condition:
            self.operations.pop(operation, None)
            self.changed.set()
            self.condition.notify_all()

    def request_pause(self) -> bool:
        """Close cooperative admission; report when native calls have drained."""
        with self.condition:
            self.pausing = True
            return not self.operations and not self.connections

    def confirm_parked(self):
        with self.condition:
            if not self.pausing or self.operations or self.connections:
                raise HermesRelayError("Hermes generation is not quiescent")
            self.parked = True

    def resume(self):
        with self.condition:
            if self.stopped or not self.parked:
                raise HermesRelayError("Hermes generation cannot resume")
            self.pausing = self.parked = False
            self.condition.notify_all()

    def stop_admission(self):
        with self.condition:
            self.stopped = True
            sockets = tuple(self.connections.values())
            self.condition.notify_all()
            self.changed.set()
        for transport in sockets:
            if transport is not None:
                try:
                    transport.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            # The forwarding thread owns close; shutdown wakes its active read.

    def close(self):
        self.stop_admission()
        if self.started:
            self.server.shutdown()
        self.server.server_close()
        if self.started:
            self.thread.join(timeout=5)
        with self.condition:
            if not self.condition.wait_for(lambda: not self.connections, timeout=5):
                raise HermesRelayError("Hermes provider drain is unconfirmed")
        self.socket_path.unlink(missing_ok=True)

    def forward(self, handler, body: bytes):
        value = json.loads(body)
        if not isinstance(value, dict) or value.get("model") != self.model:
            raise HermesRelayError("Native request selected an unavailable model")
        admitted = self.admitted()
        endpoint = self.endpoint
        connection_type = (http.client.HTTPSConnection if endpoint.scheme == "https"
                           else http.client.HTTPConnection)
        connection = connection_type(endpoint.hostname, endpoint.port, timeout=5)
        with self.condition:
            if (not admitted or self.stopped or self.parked
                    or self.pausing and not self.operations):
                raise HermesRelayError("Hermes does not own model admission")
            self.connections[connection] = None
            try:
                # Connect/send is bounded. Cancellation cannot close an unstarted
                # connection and then let this thread reopen it afterward.
                connection.request("POST", "/v1/chat/completions", body=body, headers={
                    "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
                # HTTPConnection detaches its socket for a Connection: close
                # response. Retain the transport until its reader exits.
                self.connections[connection] = connection.sock
            except BaseException:
                self.connections.pop(connection, None)
                connection.close()
                raise
        try:
            # Native RPC deadlines and the owning Run bound execution. An extra
            # relay idle deadline would truncate valid native local-model waits.
            connection.sock.settimeout(None)
            response = connection.getresponse()
            handler.send_response(response.status)
            handler.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
            handler.send_header("Connection", "close")
            handler.end_headers()
            transferred = 0
            while chunk := response.read1(65536):
                transferred += len(chunk)
                if transferred > _MAX_BODY:
                    raise HermesRelayError("Native provider response exceeds its bound")
                handler.wfile.write(chunk)
                handler.wfile.flush()
        finally:
            connection.close()
            with self.condition:
                self.connections.pop(connection, None)
                self.changed.set()
                self.condition.notify_all()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path != "/v1/models":
            self.send_error(404)
            return
        self._json({"data": [{"id": self.relay.model, "object": "model",
                              "context_length": self.relay.context_length}]})

    def _json(self, value):
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.connection.settimeout(5)
        try:
            size = int(self.headers.get("Content-Length", "-1"))
            if not 0 <= size <= _MAX_BODY or self.headers.get("Transfer-Encoding"):
                raise HermesRelayError("Invalid request size")
            body = self.rfile.read(size)
            if len(body) != size:
                raise HermesRelayError("Incomplete native request")
            if self.path == "/v1/chat/completions":
                self.relay.forward(self, body)
                return
            value = json.loads(body)
            if not isinstance(value, dict):
                raise HermesRelayError("Invalid native control request")
            if self.path == "/boundary":
                self.relay.boundary(value)
            elif self.path == "/settled":
                self.relay.settle(value.get("operation"))
            elif self.path == "/result":
                if not isinstance(value.get("reply"), str) or type(value.get("failed")) is not bool:
                    raise HermesRelayError("Invalid native result")
                with self.relay.condition:
                    if self.relay.result is not None or self.relay.stopped:
                        raise HermesRelayError("Native result is no longer admitted")
                    self.relay.result = value
                    self.relay.changed.set()
            else:
                self.send_error(404)
                return
            self._json({"accepted": True})
        except (HermesRelayError, ValueError, OSError):
            self.send_error(409, "Hermes request was not admitted")
