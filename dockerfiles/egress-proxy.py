"""Egress allowlist proxy for Assist shell and isolated browser clients.

Exact-match hostname allowlist.  Refuses anything not on the list with
HTTP 403.  Supports both CONNECT (HTTPS tunnel) and HTTP-via-proxy
(absolute-URL request line) so HTTPS traffic to pypi and HTTP traffic
to host.docker.internal:8000 (the local model endpoint) both flow
through the same gate.

Why custom Python instead of tinyproxy / squid:
  - The codebase prefers audit-friendly pieces (see the C git shim that
    replaced a bash version). Exact policy stays here instead of being
    expressed through a proxy's broad regex configuration.
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
  On a base-allowlist MISS only, /approvals (an optional read-only host
  mount) is re-read: approved-hosts.json maps grant keys to {host, port,
  origin_tid, expires_at}. The separate /client-map read-only mount binds
  network IPs to thread and container generation. A grant requires an exact
  (host, port), the connecting client's thread, and a live duration marker.
  Missing or invalid attribution cannot use a grant. Without /approvals,
  ordinary shell clients retain their base-list-only behavior.

  Approved (non-base) shell hosts additionally pass a RESOLVED-ADDRESS guard:
  resolve, reject private/loopback/link-local/metadata space, and connect
  to the vetted IP — a user-approved hostname is attacker-influenceable
  (DNS rebinding), unlike the operator-curated base entries, so it must
  never be able to point into the host or LAN. Browser public mode checks
  the resolved address even for base hosts. Browser internal mode is limited
  to its one exact operator-base hostname and a non-global address; it never
  follows approvals. Both modes require source CIDR and current attribution.
  The 403 body arrives via EGRESS_DENY_BODY in assist/egress/guidance.py.

Host throttling:
  Every allowed remote hostname receives new connections on a fixed 2s, 4s,
  8s, 16s, then 60s backoff across all sandbox turns. Excess requests receive
  a local HTTP 429 before reaching the host. The local model bridge
  ``host.docker.internal`` is excluded: it is not remote egress and normal
  model execution opens many connections through that name.  CONNECT remains
  opaque TLS forwarding, so this controls new connections rather than claiming
  to inspect encrypted response rate-limit headers.
"""
import base64
import hashlib
import ipaddress
import json
import math
import os
import re
import select
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8888"))
PIPE_TIMEOUT = 600  # seconds of idle on a tunnel before tearing down
APPROVALS_DIR = os.environ.get("APPROVALS_DIR", "/approvals")
CLIENT_MAP_DIR = os.environ.get("CLIENT_MAP_DIR", "/client-map")
APPROVALS_MAX_BYTES = 65536
REVOKED_ONLY = "revoked-only"
# Repeated connections to one host start quickly, then reach the one-minute
# ceiling on the fifth request.  This is a fixed proxy boundary, not a
# per-tool/model knob, so concurrent sandbox clients cannot evade it.
HOST_BACKOFF_INTERVALS_S = (2.0, 4.0, 8.0, 16.0, 60.0)
HOST_IDLE_RESET_S = 5 * 60.0
_THROTTLE_PRUNE_THRESHOLD = 256
# The proxy carries ordinary model requests to the local host through this
# name.  It is not remote egress, and throttling it would serialize every
# model call in a normal turn.
_UNTHROTTLED_HOSTS = frozenset({"host.docker.internal"})
_EXECUTION_TOKEN_RE = re.compile(r"assist-exec-[0-9a-f]{32}")


def _network(name: str):
    raw = os.environ.get(name)
    return ipaddress.ip_network(raw, strict=True) if raw else None


SANDBOX_CIDR = _network("EGRESS_SANDBOX_CIDR")
BROWSER_CIDR = _network("EGRESS_BROWSER_CIDR")


def source_kind(client_ip: str) -> str | None:
    """Unknown or overlapping proxy ingress gets no policy fallback."""
    try:
        ip = ipaddress.ip_address(client_ip)
    except ValueError:
        return None
    ordinary = SANDBOX_CIDR is not None and ip in SANDBOX_CIDR
    browser = BROWSER_CIDR is not None and ip in BROWSER_CIDR
    if ordinary == browser:
        return None
    return "browser" if browser else "sandbox"


def load_allowlist() -> frozenset[str]:
    raw = os.environ.get("EGRESS_ALLOWLIST", "")
    return frozenset(h.strip().lower() for h in raw.split(",") if h.strip())


ALLOWLIST = load_allowlist()
DENY_BODY = os.environ.get("EGRESS_DENY_BODY", "").encode("utf-8")
THROTTLE_BODY = os.environ.get("EGRESS_THROTTLE_BODY", "").encode("utf-8")


def log(msg: str) -> None:
    print(f"egress-proxy: {msg}", flush=True)


class HostThrottle:
    """Admit one remote-host connection with bounded exponential backoff."""

    def __init__(self, intervals_s=HOST_BACKOFF_INTERVALS_S,
                 clock=time.monotonic) -> None:
        self._intervals_s = intervals_s
        self._clock = clock
        self._lock = threading.Lock()
        self._states: dict[str, tuple[float, float, int]] = {}

    def admit(self, host: str) -> None:
        """Reserve one connection or raise without creating a waiting thread."""
        now = self._clock()
        with self._lock:
            next_allowed, last_request, count = self._states.get(
                host, (now, float("-inf"), 0))
            if now - last_request >= HOST_IDLE_RESET_S:
                next_allowed, count = now, 0
            retry_after = max(1, math.ceil(next_allowed - now))
            if next_allowed > now:
                self._states[host] = (next_allowed, now, count)
                raise HostThrottleBusy(retry_after)
            interval_s = self._intervals_s[min(count, len(self._intervals_s) - 1)]
            self._states[host] = (now + interval_s, now, count + 1)
            self._prune(now)

    def _prune(self, now: float) -> None:
        if len(self._states) <= _THROTTLE_PRUNE_THRESHOLD:
            return
        self._states = {
            host: state for host, state in self._states.items()
            if now - state[1] < HOST_IDLE_RESET_S
        }


class HostThrottleBusy(Exception):
    """A local rate limit that prevented an upstream connection."""

    def __init__(self, retry_after_s: int) -> None:
        self.retry_after_s = retry_after_s


HOST_THROTTLE = HostThrottle()


def admit_host(host: str) -> None:
    """Apply the shared remote-host admission boundary before connect."""
    if host in _UNTHROTTLED_HOSTS:
        return
    HOST_THROTTLE.admit(host)


def connect_upstream(host: str, port: int,
                     approved_ip: str | None) -> socket.socket:
    """Admit then open one vetted upstream connection."""
    admit_host(host)
    return socket.create_connection((approved_ip or host, port), timeout=10)


def deny(client: socket.socket, host: str, reason: str) -> None:
    log(f"DENY {host} ({reason})")
    try:
        code = reason if re.fullmatch(r"[a-z_]+", reason) else "proxy_denied"
        client.sendall(
            b"HTTP/1.1 403 Forbidden\r\n"
            b"Content-Type: text/plain\r\n"
            + f"X-Assist-Egress-Result: {code}\r\n".encode()
            + f"Content-Length: {len(DENY_BODY)}\r\n".encode()
            + b"Connection: close\r\n\r\n" + DENY_BODY
        )
    except OSError:
        pass


def request_correlation(header_block: str) -> str | None:
    """Return a hashed per-execution proxy marker, if one is well formed.

    The sandbox host sets a fresh username in its proxy URL for every command.
    Clients put it in Proxy-Authorization, which is visible before a CONNECT
    tunnel starts. Hash it before logging: the log only needs a correlation
    marker and must not retain the token itself.
    """
    for line in header_block.split("\r\n"):
        name, separator, value = line.partition(":")
        if not separator or name.strip().lower() != "proxy-authorization":
            continue
        scheme, separator, encoded = value.strip().partition(" ")
        if scheme.lower() != "basic" or not separator:
            return None
        try:
            username, _, _ = base64.b64decode(encoded, validate=True).decode(
                "utf-8").partition(":")
        except (UnicodeDecodeError, ValueError):
            return None
        if _EXECUTION_TOKEN_RE.fullmatch(username):
            return hashlib.sha256(username.encode()).hexdigest()[:16]
        return None
    return None


def throttle(client: socket.socket, client_ip: str, host: str,
             retry_after_s: int, correlation: str | None) -> None:
    """Return a local bounded-rate response without contacting the host."""
    # The host backend reads timestamped proxy events matching both this
    # container's egress-network IP and its fresh execution marker. That is
    # trusted provenance for clients which discard a 429 body and headers.
    log(f"THROTTLE client={client_ip} host={host} retry after {retry_after_s}s "
        f"execution={correlation or '-'}")
    try:
        client.sendall(
            b"HTTP/1.1 429 Too Many Requests\r\n"
            b"Content-Type: text/plain\r\n"
            b"X-Assist-Egress-Result: throttled\r\n"
            + f"Retry-After: {retry_after_s}\r\n".encode()
            + f"Content-Length: {len(THROTTLE_BODY)}\r\n".encode()
            + b"Connection: close\r\n\r\n" + THROTTLE_BODY
        )
    except OSError:
        pass


def _read_small_json(name: str, directory: str = APPROVALS_DIR) -> dict:
    """Fail-closed read of one /approvals file: any problem — missing,
    oversized, unparseable, wrong shape — is an empty dict (base list only)."""
    path = os.path.join(directory, name)
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


def client_record(client_ip: str, kind: str) -> dict | None:
    """Read one current-schema attribution record; legacy values grant nothing."""
    value = _read_small_json("client-map.json", CLIENT_MAP_DIR).get(client_ip)
    if not isinstance(value, dict) or value.get("kind") != kind:
        return None
    required = {"thread_id", "generation", "kind"}
    if kind == "browser":
        required.add("browser_mode")
        if value.get("browser_mode") == "internal":
            required.add("internal_host")
        elif value.get("browser_mode") != "public":
            return None
    if set(value) != required:
        return None
    if (not isinstance(value.get("thread_id"), str)
            or not value["thread_id"] or len(value["thread_id"]) > 128
            or not isinstance(value.get("generation"), str)
            or not value["generation"] or len(value["generation"]) > 128):
        return None
    if kind == "browser" and value.get("browser_mode") == "internal":
        internal_host = value["internal_host"]
        if (not isinstance(internal_host, str) or not internal_host
                or internal_host != internal_host.lower()
                or internal_host not in ALLOWLIST):
            return None
    return value


def approved_target(host: str, port: int, tid: str | None) -> bool:
    """True iff a live, thread-matching grant covers (host, port) for the
    thread that owns the connecting sandbox. Direct key lookup (the
    projection is keyed ``tid:host:port``), but the ENTRY's own fields are
    the authority — the key is never trusted on its own. Fail-closed on
    every parse problem."""
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


def vet_resolved(host: str, port: int, *, global_only: bool = True) -> str | None:
    """Resolve once and return an address that the caller will actually dial."""
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
        if ip.is_global == global_only:
            return addr
    return None


def target_policy(host: str, port: int, client_ip: str) -> tuple[str | None, str | None]:
    """Return a vetted dial IP and no error, or a stable denial reason."""
    kind = source_kind(client_ip)
    if kind is None:
        return None, "unknown_proxy_source"
    record = client_record(client_ip, kind)
    if kind == "browser" and record is None:
        return None, "browser_attribution_missing"
    if kind == "browser" and record["browser_mode"] == "internal":
        if host != record["internal_host"] or host not in ALLOWLIST:
            return None, "browser_internal_policy"
        address = vet_resolved(host, port, global_only=False)
        return (address, None) if address else (None, "browser_internal_address")
    if host not in ALLOWLIST:
        if not approved_target(host, port, record["thread_id"] if record else None):
            return None, "host_not_approved"
    if kind == "browser" or host not in ALLOWLIST:
        # Every public browser destination, including an operator base
        # host, dials the checked address. The older shell base list keeps
        # its existing resolution contract.
        address = vet_resolved(host, port)
        if address is None:
            return None, ("browser_internal_policy" if kind == "browser"
                          else "approved_address_not_public")
    else:
        address = None
    return address, None


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
        correlation = request_correlation(header_block)
        parts = request_line.split(" ")
        if len(parts) != 3:
            deny(client, "<malformed>", "bad_request")
            return
        method, target, _ = parts

        if method == "CONNECT":
            host, _, port_str = target.partition(":")
            host = host.lower()  # DNS hostnames are case-insensitive (RFC 4343)
            try:
                port = int(port_str) if port_str else 443
            except ValueError:
                deny(client, host, "bad_port")
                return
            if not 1 <= port <= 65535:
                deny(client, host, "bad_port")
                return
            approved_ip, rejection = target_policy(host, port, addr[0])
            if rejection:
                deny(client, host, rejection)
                return
            try:
                upstream = connect_upstream(host, port, approved_ip)
            except HostThrottleBusy as e:
                throttle(client, addr[0], host, e.retry_after_s, correlation)
                return
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
            deny(client, "<malformed>", "bad_request")
            return
        u = urlparse(target)
        host = (u.hostname or "").lower()
        try:
            port = u.port or 80
        except ValueError:
            deny(client, host, "bad_port")
            return
        if not host or not 1 <= port <= 65535 or u.username or u.password:
            deny(client, host or "<malformed>", "bad_request")
            return
        approved_ip, rejection = target_policy(host, port, addr[0])
        if rejection:
            deny(client, host, rejection)
            return
        path = u.path or "/"
        if u.query:
            path += "?" + u.query
        kept_headers = []
        for line in header_block.split("\r\n"):
            if not line:
                continue
            name = line.split(":", 1)[0].strip().lower()
            if name in ("proxy-connection", "proxy-authorization", "connection"):
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
            upstream = connect_upstream(host, port, approved_ip)
        except HostThrottleBusy as e:
            throttle(client, addr[0], host, e.retry_after_s, correlation)
            return
        except OSError as e:
            deny(client, host, f"upstream connect failed: {e}")
            return
        if approved_ip:
            log(f"ALLOW {method} {host}:{port} via {approved_ip}")
        else:
            log(f"ALLOW {method} {host}:{port}")
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
    if SANDBOX_CIDR is None or (BROWSER_CIDR is not None
                                and SANDBOX_CIDR.overlaps(BROWSER_CIDR)):
        log("ERROR: invalid proxy ingress networks")
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
