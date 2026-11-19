"""Permanent destination boundary shared by agent HTTP transports and the proxy.

The web listener occupies ASSIST_PORT on every local IPv4 address. Private,
loopback and other non-public addresses on that port are never agent egress.
ASSIST_BASE_URL additionally identifies the operator's published web origin.
"""
import ipaddress
import os
import socket
from urllib.parse import urlsplit


class ControlApiDenied(ValueError):
    """An agent request names the Assist control service."""


def control_config() -> tuple[int, str]:
    """Return the trusted service port and optional published origin."""
    port = int(os.environ.get("ASSIST_PORT") or "8000")
    if not 1 <= port <= 65535:
        raise ValueError("ASSIST_PORT is outside 1-65535")
    return port, os.environ.get("ASSIST_BASE_URL", "")


def _address(value: str):
    ip = ipaddress.ip_address(value)
    return ip.ipv4_mapped if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped else ip


def resolve_targets(host: str, port: int, *, public_only: bool = False) -> tuple[str, ...]:
    """Check every resolved address, then return vetted numeric dial candidates.

    A base allowlist or egress grant cannot override this check. Rejecting a
    mixed protected/public answer set prevents address-order dependent policy.
    """
    control_port, origin = control_config()
    published_ips = set()
    published_port = None
    if origin:
        parsed = urlsplit(origin)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("ASSIST_BASE_URL must be an HTTP(S) origin")
        published_port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if port == published_port:
            if host.lower().rstrip(".") == parsed.hostname.lower().rstrip("."):
                raise ControlApiDenied("Assist control APIs are unavailable to agents")
            published_ips = {_address(info[4][0]) for info in socket.getaddrinfo(
                parsed.hostname, published_port, proto=socket.IPPROTO_TCP)}
    addresses = [info[4][0] for info in socket.getaddrinfo(
        host, port, proto=socket.IPPROTO_TCP)]
    for address in addresses:
        ip = _address(address)
        if ((port == control_port and not ip.is_global)
                or port == published_port and ip in published_ips):
            raise ControlApiDenied("Assist control APIs are unavailable to agents")
    permitted = tuple(dict.fromkeys(address for address in addresses
                                    if not public_only or _address(address).is_global))
    if permitted:
        return permitted
    raise OSError("destination has no permitted address")
