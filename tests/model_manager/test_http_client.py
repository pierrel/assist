"""TCP keepalive and httpx client wiring for the ChatOpenAI used by every
LLM call in assist.  Closes the 2026-05-29 CLOSE_WAIT-wedge class.

Background: docs/2026-05-30-llm-client-tcp-keepalive.org.

Located under ``tests/model_manager/`` to match the existing layout
(see ``test_dynamic_model.py`` for the endpoint-discovery tests).
"""
import socket
import threading

import httpx
import pytest

from assist.model_manager import (
    _TCP_KEEPALIVE_SOCKET_OPTIONS,
    _AsyncHttpClient,
    _HttpClient,
    _build_http_async_client,
    _build_http_client,
    _build_request_timeout,
)


def test_tcp_keepalive_socket_options_constant():
    # Pin the exact (level, optname, value) tuples.  The detection-time
    # math in the module docstring depends on these specific values:
    #
    #     30 (idle) + 3 * 10 (probes) = 60s before kernel surfaces a
    #     dead peer as ECONNRESET.
    #
    # Changing any of these without re-doing the math is a regression.
    assert _TCP_KEEPALIVE_SOCKET_OPTIONS == [
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
        (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30),
        (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10),
        (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3),
    ]


def test_build_http_client_returns_keepalive_client():
    # Sync client is an _HttpClient (httpx.Client subclass with __del__)
    # and carries the per-phase Timeout.  The actual kernel-level proof
    # that keepalive options reach the socket lives in
    # `test_keepalive_options_actually_applied_to_kernel_socket` below.
    client = _build_http_client()
    try:
        assert isinstance(client, _HttpClient)
        assert isinstance(client, httpx.Client)
        expected = _build_request_timeout()
        assert client.timeout.connect == expected.connect
        assert client.timeout.read == expected.read
        assert client.timeout.write == expected.write
        assert client.timeout.pool == expected.pool
    finally:
        client.close()


def test_build_http_async_client_returns_keepalive_async_client():
    # Async client must also have keepalive — deepagents' subagent
    # dispatch is `await subagent.ainvoke(...)`, which routes through
    # ChatOpenAI's `http_async_client`.  Without coverage here, the
    # subagent LLM calls (`context`/`research`/`critique`) would still
    # be exposed to the 2026-05-29 wedge.
    aclient = _build_http_async_client()
    assert isinstance(aclient, _AsyncHttpClient)
    assert isinstance(aclient, httpx.AsyncClient)
    expected = _build_request_timeout()
    assert aclient.timeout.connect == expected.connect
    assert aclient.timeout.read == expected.read
    assert aclient.timeout.write == expected.write
    assert aclient.timeout.pool == expected.pool


def test_keepalive_options_actually_applied_to_kernel_socket(monkeypatch):
    """The kernel-level proof: when the httpx client opens a connection,
    the four keepalive ``setsockopt`` calls actually reach a socket.

    Records every ``setsockopt`` call against ``socket.socket`` during
    one HTTP GET, then asserts each keepalive tuple is present.  This
    is what catches an httpx upgrade that silently drops
    ``socket_options`` between the client and the kernel — pure
    introspection of ``client._transport._pool._socket_options`` would
    pass even when the transport stopped applying them.
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

    serve_error: list[Exception] = []

    def serve_one() -> None:
        # Narrow except: OSError covers accept/recv/send IO failures,
        # which we want to surface (not silently swallow) so the test
        # fails with the real cause rather than a misleading "options
        # not applied" assertion further down.
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
        except OSError as exc:
            serve_error.append(exc)

    serve_thread = threading.Thread(target=serve_one, daemon=True)
    serve_thread.start()

    client = _build_http_client()
    try:
        client.get(f"http://127.0.0.1:{port}/")
    finally:
        client.close()
        server.close()
        serve_thread.join(timeout=2.0)

    assert not serve_error, f"server-side IO error: {serve_error[0]!r}"

    # Each keepalive option must have been applied to *some* socket
    # during the connection — the client's outbound socket.  We don't
    # care which socket object specifically; the kernel-level proof is
    # that the call happened with the expected (level, optname, value).
    # `recording_setsockopt` runs from any thread that touches a socket
    # during the test (atomic list.append under the GIL keeps this safe).
    for expected in _TCP_KEEPALIVE_SOCKET_OPTIONS:
        assert expected in options_set, (
            f"setsockopt{expected} was not applied to any socket during "
            f"the httpx client's connection — options_set={options_set}"
        )
