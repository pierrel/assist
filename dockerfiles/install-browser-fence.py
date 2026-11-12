"""Separate browser and shell proxy access inside the turn sandbox."""
import ipaddress
import subprocess
import sys
import time


BROWSER_UID = "10001"
BROWSER_PROXY_PORT = "8889"
CHAIN = "ASSIST_BROWSER"


def run(deadline, *args):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("browser network fence setup timed out")
    subprocess.run(args, check=True, timeout=remaining,
                   stdout=subprocess.DEVNULL)


def install(proxy_ip, attached_ips):
    deadline = time.monotonic() + 12
    address = ipaddress.IPv4Address(proxy_ip)
    if (not address.is_private or address.is_loopback
            or address.is_link_local or address.is_unspecified):
        raise ValueError("browser proxy must use the internal bridge")
    proxy_ip = str(address)
    attached = []
    for raw in attached_ips:
        ip = ipaddress.IPv4Address(raw)
        if not ip.is_private or ip.is_loopback or ip.is_link_local:
            raise ValueError("invalid proxy attachment")
        attached.append(str(ip))
    if proxy_ip not in attached or len(attached) != len(set(attached)):
        raise ValueError("incomplete proxy attachment set")
    # Exclude every untrusted UID before Chromium can bind its control port.
    run(deadline, "/usr/bin/iptables", "-w", "3", "-I", "OUTPUT", "1",
        "-d", "127.0.0.1", "-p", "tcp", "--dport", "9222",
        "-m", "owner", "!", "--uid-owner", BROWSER_UID, "-j", "REJECT")
    run(deadline, "/usr/bin/iptables", "-w", "3", "-N", CHAIN)
    run(deadline, "/usr/bin/iptables", "-w", "3", "-A", CHAIN,
        "-d", "127.0.0.1", "-p", "tcp", "--dport", "9222", "-j", "RETURN")
    run(deadline, "/usr/bin/iptables", "-w", "3", "-A", CHAIN,
        "-d", "127.0.0.1", "-p", "tcp", "--sport", "9222",
        "-m", "conntrack", "--ctstate", "ESTABLISHED", "-j", "RETURN")
    run(deadline, "/usr/bin/iptables", "-w", "3", "-A", CHAIN, "-d", proxy_ip,
        "-p", "tcp", "--dport", BROWSER_PROXY_PORT, "-j", "RETURN")
    run(deadline, "/usr/bin/iptables", "-w", "3", "-A", CHAIN, "-j", "REJECT")
    run(deadline, "/usr/bin/iptables", "-w", "3", "-I", "OUTPUT", "1", "-m", "owner",
        "--uid-owner", BROWSER_UID, "-j", CHAIN)
    for attached_ip in attached:
        run(deadline, "/usr/bin/iptables", "-w", "3", "-I", "OUTPUT", "1",
            "-d", attached_ip, "-p", "tcp", "--dport", BROWSER_PROXY_PORT,
            "-m", "owner", "!", "--uid-owner", BROWSER_UID, "-j", "REJECT")
    run(deadline, "/usr/bin/ip6tables", "-w", "3", "-I", "OUTPUT", "1", "-m", "owner",
        "--uid-owner", BROWSER_UID, "-j", "REJECT")
    run(deadline, "/usr/bin/iptables", "-w", "3", "-C", "OUTPUT", "-m", "owner",
        "--uid-owner", BROWSER_UID, "-j", CHAIN)
    for attached_ip in attached:
        run(deadline, "/usr/bin/iptables", "-w", "3", "-C", "OUTPUT",
            "-d", attached_ip, "-p", "tcp", "--dport", BROWSER_PROXY_PORT,
            "-m", "owner", "!", "--uid-owner", BROWSER_UID, "-j", "REJECT")
    run(deadline, "/usr/bin/ip6tables", "-w", "3", "-C", "OUTPUT", "-m", "owner",
        "--uid-owner", BROWSER_UID, "-j", "REJECT")


def seal_control():
    """Keep the trusted established channel; deny every new CDP connection."""
    run(time.monotonic() + 5, "/usr/bin/iptables", "-w", "3", "-I", "OUTPUT", "1",
        "-d", "127.0.0.1", "-p", "tcp", "--dport", "9222",
        "-m", "conntrack", "!", "--ctstate", "ESTABLISHED", "-j", "REJECT")


if __name__ == "__main__":
    if sys.argv[1:] == ["seal-control"]:
        seal_control()
        raise SystemExit(0)
    if len(sys.argv) < 3:
        raise SystemExit(2)
    install(sys.argv[1], sys.argv[2:])
