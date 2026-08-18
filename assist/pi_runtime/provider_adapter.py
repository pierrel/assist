"""Expose one private Unix provider relay to a networkless Pi worker process."""
from __future__ import annotations

import os
import signal
import socket
import sys
import time


_SOCKET_PATH = "/run/pi/provider.sock"
_PORT = 18765
_MAX_REQUEST_BYTES = 2 * 1024 * 1024
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_HEADERS_BYTES = 32 * 1024
_REQUEST_TIMEOUT_SECONDS = 5


class _WorkerExited(RuntimeError):
    """The Pi worker exited while the adapter was relaying its request."""

    def __init__(self, status: int) -> None:
        self.status = status


def _check_child(child: int | None) -> None:
    if child is None:
        return
    finished, status = os.waitpid(child, os.WNOHANG)
    if finished:
        raise _WorkerExited(os.waitstatus_to_exitcode(status))


def _receive(connection: socket.socket, child: int | None,
             timeout: float | None) -> bytes:
    """Receive with short polls so PID 1 observes a dead worker promptly."""
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        _check_child(child)
        remaining = 0.2 if deadline is None else min(0.2, deadline - time.monotonic())
        if remaining <= 0:
            raise RuntimeError("Pi provider request timed out")
        connection.settimeout(remaining)
        try:
            return connection.recv(65536)
        except TimeoutError:
            continue


def _read_request(client: socket.socket, child: int | None = None) -> bytes:
    """Read one complete, bounded HTTP request before contacting the host relay."""
    deadline = time.monotonic() + _REQUEST_TIMEOUT_SECONDS
    raw = bytearray()
    body_length: int | None = None
    expected_length: int | None = None
    while expected_length is None or len(raw) < expected_length:
        chunk = _receive(client, child, max(0, deadline - time.monotonic()))
        if not chunk:
            raise RuntimeError("Pi provider request ended early")
        raw.extend(chunk)
        if body_length is None:
            marker = raw.find(b"\r\n\r\n")
            if marker < 0:
                if len(raw) > _MAX_HEADERS_BYTES:
                    raise RuntimeError("Pi provider headers exceed their bound")
                continue
            if marker + 4 > _MAX_HEADERS_BYTES:
                raise RuntimeError("Pi provider headers exceed their bound")
            header_block = bytes(raw[:marker]).decode("ascii", "strict")
            headers = header_block.split("\r\n")
            content_lengths = [line.split(":", 1)[1].strip() for line in headers[1:]
                               if line.lower().startswith("content-length:")]
            if len(content_lengths) != 1 or not content_lengths[0].isdecimal():
                raise RuntimeError("Pi provider request has invalid Content-Length")
            body_length = int(content_lengths[0])
            if body_length > _MAX_REQUEST_BYTES:
                raise RuntimeError("Pi provider request exceeds its bound")
            expected_length = marker + 4 + body_length
            if expected_length > _MAX_HEADERS_BYTES + _MAX_REQUEST_BYTES:
                raise RuntimeError("Pi provider request exceeds its bound")
        if len(raw) > _MAX_HEADERS_BYTES + _MAX_REQUEST_BYTES:
            raise RuntimeError("Pi provider request exceeds its bound")
    if len(raw) != expected_length:
        raise RuntimeError("Pi provider request has trailing bytes")
    return bytes(raw)


def _relay(client: socket.socket, child: int | None = None) -> None:
    """Relay one bounded request and one bounded response over the private socket."""
    request = _read_request(client, child)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as upstream:
        _check_child(child)
        upstream.settimeout(_REQUEST_TIMEOUT_SECONDS)
        upstream.connect(_SOCKET_PATH)
        upstream.sendall(request)
        transferred = 0
        # The host relay enforces the same idle-byte timeout as Deep Agents.
        # Do not add a shorter adapter-wide deadline around a healthy stream.
        while chunk := _receive(upstream, child, None):
            transferred += len(chunk)
            if transferred > _MAX_HEADERS_BYTES + _MAX_RESPONSE_BYTES:
                raise RuntimeError("Pi provider response exceeds its bound")
            client.sendall(chunk)


def _make_listener(port: int = _PORT) -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen(1)
    listener.settimeout(0.2)
    return listener


def _configured_port() -> int:
    """Permit an isolated test port; production starts with an empty environment."""
    value = os.environ.get("ASSIST_PI_ADAPTER_PORT", str(_PORT))
    if not value.isdecimal() or not 1024 <= int(value) <= 65535:
        raise RuntimeError("Pi provider adapter port is invalid")
    return int(value)


def _serve(listener: socket.socket, child: int) -> int:
    """Keep the listener as PID 1 until the Pi worker exits."""
    while True:
        try:
            _check_child(child)
        except _WorkerExited as exited:
            return exited.status
        try:
            client, _ = listener.accept()
        except TimeoutError:
            continue
        with client:
            try:
                _relay(client, child)
            except _WorkerExited as exited:
                return exited.status
            except (OSError, RuntimeError, UnicodeError):
                # The host relay is the authority and returns the useful error to Pi.
                # A malformed or interrupted local client must not end the worker turn.
                continue


def _stop_child(child: int) -> None:
    try:
        os.kill(child, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        os.waitpid(child, 0)
    except ChildProcessError:
        pass


def main() -> int:
    command = sys.argv[1:]
    if not command:
        raise RuntimeError("Pi worker command is required")
    listener = _make_listener(_configured_port())
    child = os.fork()
    if child == 0:
        listener.close()
        os.execvp(command[0], command)
    try:
        return _serve(listener, child)
    except BaseException:
        _stop_child(child)
        raise
    finally:
        listener.close()


if __name__ == "__main__":
    raise SystemExit(main())
