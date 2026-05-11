"""Egress allowlist proxy for the assist sandbox.

Exact-match hostname allowlist.  Refuses anything not on the list with
HTTP 403.  Supports both CONNECT (HTTPS tunnel) and HTTP-via-proxy
(absolute-URL request line) so HTTPS traffic to pypi and HTTP traffic
to host.docker.internal:8000 (the local model endpoint) both flow
through the same gate.

Why custom Python instead of tinyproxy / squid:
  - The codebase prefers small, audit-friendly pieces (see the C git
    shim that replaced a bash version).  ~150 lines of stdlib Python
    is shorter than the tinyproxy config that gets the regex semantics
    right.
  - Exact-string match is the security property we want.  A regex
    filter without explicit anchoring lets "evil-pypi.org" match
    "pypi.org" (the dot is a regex metachar) — silent allow.  Owning
    the comparison eliminates that class of bug.
  - No `Filter` regex, no MITM, no CA cert provisioning.  CONNECT is
    plaintext on the request line; we filter on the hostname there.

Allowlist source:
  EGRESS_ALLOWLIST env var (comma-separated) — set by
  SandboxManager._ensure_egress_proxy_running at container-create time.
  No file fallback; the env var is the wire protocol from host to proxy.
"""
import os
import select
import socket
import sys
import threading
from urllib.parse import urlparse

LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8888"))
PIPE_TIMEOUT = 600  # seconds of idle on a tunnel before tearing down


def load_allowlist() -> frozenset[str]:
    raw = os.environ.get("EGRESS_ALLOWLIST", "")
    return frozenset(h.strip().lower() for h in raw.split(",") if h.strip())


ALLOWLIST = load_allowlist()


def log(msg: str) -> None:
    print(f"egress-proxy: {msg}", flush=True)


def deny(client: socket.socket, host: str, reason: str) -> None:
    log(f"DENY {host} ({reason})")
    try:
        client.sendall(
            b"HTTP/1.1 403 Forbidden\r\n"
            b"Content-Length: 0\r\n"
            b"Connection: close\r\n\r\n"
        )
    except OSError:
        pass


def pipe(a: socket.socket, b: socket.socket) -> None:
    """Bidirectional byte pump with idle timeout."""
    try:
        while True:
            r, _, _ = select.select([a, b], [], [], PIPE_TIMEOUT)
            if not r:
                return
            for s in r:
                try:
                    data = s.recv(8192)
                except OSError:
                    return
                if not data:
                    return
                other = b if s is a else a
                try:
                    other.sendall(data)
                except OSError:
                    return
    except Exception as e:
        log(f"pipe error: {e}")


def read_request_head(client: socket.socket) -> tuple[str, bytes]:
    """Read until \\r\\n\\r\\n; return (head, leftover_body_bytes)."""
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = client.recv(4096)
        if not chunk:
            return "", b""
        buf += chunk
        if len(buf) > 65536:
            raise ValueError("request head exceeds 64KB")
    head, _, rest = buf.partition(b"\r\n\r\n")
    return head.decode("latin-1"), rest


def handle(client: socket.socket, addr) -> None:
    upstream = None
    try:
        client.settimeout(30)
        try:
            head, body = read_request_head(client)
        except ValueError as e:
            deny(client, "<oversize>", str(e))
            return
        if not head:
            return
        request_line, _, header_block = head.partition("\r\n")
        parts = request_line.split(" ")
        if len(parts) != 3:
            deny(client, "<malformed>", f"bad request line: {request_line!r}")
            return
        method, target, _ = parts

        if method == "CONNECT":
            host, _, port_str = target.partition(":")
            host = host.lower()  # DNS hostnames are case-insensitive (RFC 4343)
            try:
                port = int(port_str) if port_str else 443
            except ValueError:
                deny(client, target, "bad port")
                return
            if host not in ALLOWLIST:
                deny(client, host, "not in allowlist")
                return
            try:
                upstream = socket.create_connection((host, port), timeout=10)
            except OSError as e:
                deny(client, host, f"upstream connect failed: {e}")
                return
            log(f"ALLOW CONNECT {host}:{port}")
            client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            client.settimeout(None)
            upstream.settimeout(None)
            pipe(client, upstream)
            return

        # HTTP via proxy: request line is "METHOD http://host[:port]/path HTTP/1.1"
        if not target.startswith(("http://", "https://")):
            deny(client, target, "non-absolute URL on non-CONNECT method")
            return
        u = urlparse(target)
        host = (u.hostname or "").lower()
        port = u.port or (443 if u.scheme == "https" else 80)
        if host not in ALLOWLIST:
            deny(client, host, f"not in allowlist (HTTP {method})")
            return
        path = u.path or "/"
        if u.query:
            path += "?" + u.query
        kept_headers = []
        for line in header_block.split("\r\n"):
            if not line:
                continue
            name = line.split(":", 1)[0].strip().lower()
            if name in ("proxy-connection", "connection"):
                continue
            kept_headers.append(line)
        new_request = f"{method} {path} HTTP/1.1\r\n"
        if not any(h.split(":", 1)[0].strip().lower() == "host" for h in kept_headers):
            new_request += f"Host: {u.netloc}\r\n"
        new_request += "Connection: close\r\n"
        for h in kept_headers:
            new_request += h + "\r\n"
        new_request += "\r\n"
        try:
            upstream = socket.create_connection((host, port), timeout=10)
        except OSError as e:
            deny(client, host, f"upstream connect failed: {e}")
            return
        log(f"ALLOW {method} {host}:{port}{path}")
        upstream.sendall(new_request.encode("latin-1") + body)
        client.settimeout(None)
        upstream.settimeout(None)
        pipe(client, upstream)
    except Exception as e:
        log(f"handler error from {addr}: {e}")
    finally:
        try:
            client.close()
        except OSError:
            pass
        if upstream is not None:
            try:
                upstream.close()
            except OSError:
                pass


def main() -> int:
    if not ALLOWLIST:
        log("ERROR: empty allowlist (EGRESS_ALLOWLIST env unset and " +
            f"{ALLOWLIST_PATH} missing).  Refusing to start fail-open.")
        return 2
    log(f"allowlist ({len(ALLOWLIST)} entries): {sorted(ALLOWLIST)}")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", LISTEN_PORT))
    srv.listen(64)
    log(f"listening on 0.0.0.0:{LISTEN_PORT}")
    while True:
        client, addr = srv.accept()
        threading.Thread(target=handle, args=(client, addr), daemon=True).start()


if __name__ == "__main__":
    sys.exit(main() or 0)
