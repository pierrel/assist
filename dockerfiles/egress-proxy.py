"""Egress allowlist proxy for the assist sandbox.

Exact-match hostname allowlist.  Refuses anything not on the list with
HTTP 403.  Supports both CONNECT (HTTPS tunnel) and HTTP-via-proxy
(absolute-URL request line) so HTTPS traffic to pypi and HTTP traffic
to host.docker.internal:8000 (the local model endpoint) both flow
through the same gate.

Why custom Python instead of tinyproxy / squid:
  - The codebase prefers small, audit-friendly pieces (see the C git
    shim that replaced a bash version).  One small stdlib-Python file
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

User-approved grants (docs/2026-07-21-egress-approval-hitl.org):
  On a base-allowlist MISS only, /approvals (a read-only host mount) is
  re-read: approved-hosts.json maps grant keys to {host, port, origin_tid,
  expires_at} and client-map.json maps sandbox egress-network IPs to thread
  ids. A connection is granted iff an entry matches the exact (host, port),
  its origin_tid equals the CONNECTING CLIENT's thread (grants are
  thread-scoped), and its duration marker is live — an aware-UTC future
  expires_at, or the literal "revoked-only". Every parse problem, unknown
  client IP, or odd marker is a skip/deny: the files can only ever ADD
  narrowly, never remove or re-match base entries, and the proxy runs
  unchanged (base list only) when the mount is absent.

  Approved (non-base) hosts additionally pass a RESOLVED-ADDRESS guard:
  resolve, reject private/loopback/link-local/metadata space, and connect
  to the vetted IP — a user-approved hostname is attacker-influenceable
  (DNS rebinding), unlike the operator-curated base entries, so it must
  never be able to point into the host or LAN. The 403 body text arrives
  via EGRESS_DENY_BODY (centralized in assist/egress/guidance.py — no
  guidance prose lives here).
"""
import ipaddress
import json
import os
import select
import socket
import sys
import threading
from datetime import datetime, timezone
from urllib.parse import urlparse

LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8888"))
PIPE_TIMEOUT = 600  # seconds of idle on a tunnel before tearing down
APPROVALS_DIR = os.environ.get("APPROVALS_DIR", "/approvals")
APPROVALS_MAX_BYTES = 65536
REVOKED_ONLY = "revoked-only"


def load_allowlist() -> frozenset[str]:
    raw = os.environ.get("EGRESS_ALLOWLIST", "")
    return frozenset(h.strip().lower() for h in raw.split(",") if h.strip())


ALLOWLIST = load_allowlist()
DENY_BODY = os.environ.get("EGRESS_DENY_BODY", "").encode("utf-8")


def log(msg: str) -> None:
    print(f"egress-proxy: {msg}", flush=True)


def deny(client: socket.socket, host: str, reason: str) -> None:
    log(f"DENY {host} ({reason})")
    try:
        client.sendall(
            b"HTTP/1.1 403 Forbidden\r\n"
            b"Content-Type: text/plain\r\n"
            + f"Content-Length: {len(DENY_BODY)}\r\n".encode()
            + b"Connection: close\r\n\r\n" + DENY_BODY
        )
    except OSError:
        pass


def _read_small_json(name: str) -> dict:
    """Fail-closed read of one /approvals file: any problem — missing,
    oversized, unparseable, wrong shape — is an empty dict (base list only)."""
    path = os.path.join(APPROVALS_DIR, name)
    try:
        if os.path.getsize(path) > APPROVALS_MAX_BYTES:
            log(f"ignoring oversized {name}")
            return {}
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _grant_live(expires_at) -> bool:
    if expires_at == REVOKED_ONLY:
        return True
    try:
        exp = datetime.fromisoformat(str(expires_at))
        return exp.tzinfo is not None and exp > datetime.now(timezone.utc)
    except Exception:
        return False


def approved_target(host: str, port: int, client_ip: str) -> bool:
    """True iff a live, thread-matching grant covers (host, port) for the
    thread that owns the connecting sandbox. Direct key lookup (the
    projection is keyed ``tid:host:port``), but the ENTRY's own fields are
    the authority — the key is never trusted on its own. Fail-closed on
    every parse problem."""
    tid = _read_small_json("client-map.json").get(client_ip)
    if not tid:
        return False
    entry = _read_small_json("approved-hosts.json").get(f"{tid}:{host}:{port}")
    if not isinstance(entry, dict):
        return False
    try:
        return (str(entry["host"]).lower() == host
                and int(entry["port"]) == port
                and str(entry["origin_tid"]) == str(tid)
                and _grant_live(entry.get("expires_at")))
    except Exception:
        return False


def vet_resolved(host: str, port: int) -> str | None:
    """Resolve an APPROVED host and return an address safe to connect to, or
    None. Rejects any private/loopback/link-local/metadata result: a
    user-approved hostname is attacker-influenceable via DNS, so it must
    never reach the host gateway, the LAN, or cloud-metadata space.
    Connecting to the vetted IP (not re-resolving) closes the check/connect
    TOCTOU."""
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except OSError:
        return None
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        # is_global is the one-predicate tight form: everything not globally
        # routable — RFC1918, loopback, link-local/metadata, ULA, v4-mapped
        # private, AND CGNAT 100.64/10 (which is_private misses) — is refused.
        if ip.is_global:
            return addr
    return None


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
            approved_ip = None
            if host not in ALLOWLIST:
                if not approved_target(host, port, addr[0]):
                    deny(client, host, "not in allowlist")
                    return
                approved_ip = vet_resolved(host, port)
                if approved_ip is None:
                    deny(client, host,
                         "approved host resolves to private/reserved space")
                    return
            try:
                upstream = socket.create_connection(
                    (approved_ip or host, port), timeout=10)
            except OSError as e:
                deny(client, host, f"upstream connect failed: {e}")
                return
            if approved_ip:
                log(f"ALLOW-APPROVED CONNECT {host}:{port} via {approved_ip}")
            else:
                log(f"ALLOW CONNECT {host}:{port}")
            client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            client.settimeout(None)
            upstream.settimeout(None)
            pipe(client, upstream)
            return

        # HTTP via proxy: request line is "METHOD http://host[:port]/path HTTP/1.1".
        # We do NOT accept https:// here — an HTTPS absolute-URI on a
        # non-CONNECT method would mean the client expects us to do
        # the TLS handshake to upstream, which we don't (CONNECT is the
        # right verb for that).  Refuse instead of silently sending
        # plaintext HTTP to port 443 and confusing everyone.
        if not target.startswith("http://"):
            deny(client, target,
                 f"non-absolute or non-http:// URL on {method} (use CONNECT for HTTPS)")
            return
        u = urlparse(target)
        host = (u.hostname or "").lower()
        port = u.port or 80
        approved_ip = None
        if host not in ALLOWLIST:
            if not approved_target(host, port, addr[0]):
                deny(client, host, f"not in allowlist (HTTP {method})")
                return
            approved_ip = vet_resolved(host, port)
            if approved_ip is None:
                deny(client, host,
                     "approved host resolves to private/reserved space")
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
            upstream = socket.create_connection(
                (approved_ip or host, port), timeout=10)
        except OSError as e:
            deny(client, host, f"upstream connect failed: {e}")
            return
        if approved_ip:
            log(f"ALLOW-APPROVED {method} {host}:{port}{path} via {approved_ip}")
        else:
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
        log("ERROR: empty allowlist (EGRESS_ALLOWLIST env unset or empty).  "
            "Refusing to start fail-open.")
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
