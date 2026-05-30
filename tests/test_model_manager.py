"""Tests for assist.model_manager — focused on the LLM httpx client config.

The wider responsibilities of model_manager (endpoint discovery, cache
invalidation, /props fallback) are exercised by integration tests via the
real probe.  This file pins the TCP keepalive / httpx wiring that closes
the 2026-05-29 CLOSE_WAIT-wedge class.  Background:
docs/2026-05-30-llm-client-tcp-keepalive.org.
"""
import socket
import threading

import httpx
import pytest

from assist.model_manager import (
    _TCP_KEEPALIVE_SOCKET_OPTIONS,
    _build_http_client,
    _build_request_timeout,
)


def test_tcp_keepalive_socket_options_constant():
    # Pin the exact (level, optname, value) tuples.  The detection-time
    # math in the module docstring depends on these specific values:
    #
    #     30 (idle) + 2 * 10 (probes) = 50s before kernel surfaces a
    #     dead peer as ECONNRESET.
    #
    # Changing any of these without re-doing the math is a regression.
    assert _TCP_KEEPALIVE_SOCKET_OPTIONS == [
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
        (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30),
        (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10),
        (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3),
    ]


def test_build_http_client_returns_httpx_client():
    client = _build_http_client()
    try:
        assert isinstance(client, httpx.Client)
    finally:
        client.close()


def test_build_http_client_propagates_socket_options_to_transport():
    # httpx 0.28 stores the socket_options on the underlying
    # httpcore ConnectionPool.  This is internal, so the assertion
    # may need updating on httpx upgrade — but it's the cheapest
    # check that the options aren't being dropped between
    # HTTPTransport's kwarg and the pool that will set them on a
    # real socket.  The behavioral test below catches an upgrade
    # that silently breaks the wiring.
    client = _build_http_client()
    try:
        transport = client._transport
        assert isinstance(transport, httpx.HTTPTransport)
        # Compare ignoring tuple-vs-list — httpcore may normalize.
        observed = list(transport._pool._socket_options)
        assert observed == _TCP_KEEPALIVE_SOCKET_OPTIONS
    finally:
        client.close()


def test_build_http_client_applies_per_phase_timeout():
    # The client's default Timeout should mirror _build_request_timeout's
    # per-phase values, so per-call timeouts have a consistent fallback.
    client = _build_http_client()
    try:
        expected = _build_request_timeout()
        actual = client.timeout
        assert actual.connect == expected.connect
        assert actual.read == expected.read
        assert actual.write == expected.write
        assert actual.pool == expected.pool
    finally:
        client.close()


def test_keepalive_options_actually_applied_to_kernel_socket(monkeypatch):
    """The kernel-level proof: when the httpx client opens a connection,
    the four keepalive ``setsockopt`` calls actually reach a socket.

    Catches the case where httpx's internals change and silently drop
    ``socket_options`` — the introspection test above wouldn't catch
    that.  Records every ``setsockopt`` call against ``socket.socket``
    during one HTTP GET, then asserts each keepalive tuple is present.
    """
    options_set: list[tuple[int, int, int]] = []
    real_setsockopt = socket.socket.setsockopt

    def recording_setsockopt(self, level, optname, value):
        # Record the int form so comparison against the constant works
        # even if value is passed as bytes (we use int values everywhere).
        if isinstance(value, int):
            options_set.append((level, optname, value))
        return real_setsockopt(self, level, optname, value)

    monkeypatch.setattr(socket.socket, "setsockopt", recording_setsockopt)

    # Tiny TCP server: accept one connection, send a minimal HTTP/1.1
    # response, close.  Bound to an ephemeral port on localhost.
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def serve_one() -> None:
        try:
            conn, _ = server.accept()
            try:
                conn.recv(4096)
                conn.send(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Length: 0\r\n"
                    b"Connection: close\r\n"
                    b"\r\n"
                )
            finally:
                conn.close()
        except Exception:
            pass

    serve_thread = threading.Thread(target=serve_one, daemon=True)
    serve_thread.start()

    client = _build_http_client()
    try:
        client.get(f"http://127.0.0.1:{port}/")
    finally:
        client.close()
        server.close()
        serve_thread.join(timeout=2.0)

    # Each keepalive option must have been applied to *some* socket
    # during the connection — the client's outbound socket.  We don't
    # care which socket object specifically; the kernel-level proof is
    # that the call happened with the expected (level, optname, value).
    for expected in _TCP_KEEPALIVE_SOCKET_OPTIONS:
        assert expected in options_set, (
            f"setsockopt{expected} was not applied to any socket during "
            f"the httpx client's connection — options_set={options_set}"
        )
