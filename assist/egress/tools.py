"""The agent's egress management tools — request, inspect, reduce.

Three tools (Pierre's PR #200 review: the agent manages its thread's
grants, not just requests new ones), all THREAD-scoped via the run config's
thread id and all returning corrective strings, never raising into the
agent loop:

- ``request_egress(host, port, task)`` — records a proposal for the user to
  approve in this thread. Records only; nothing opens on the agent's say-so.
- ``list_allowed_hosts()`` — what this thread can reach: the operator-owned
  base allowlist plus this thread's live grants (Pierre: enumeration is
  yes — it enables voluntary reduction).
- ``remove_allowed_host(host, port)`` — drops one of THIS thread's grants.
  Privilege reduction only (it can never widen access), so no HITL; base
  entries are refused (the committed conf is operator-owned).

Host validation is ADVISORY (the authoritative guard is the proxy's
resolved-address vet): URL-tolerant (the model habitually pastes full URLs
— parse the hostname out rather than bounce a corrective), lowercased,
anchored ASCII, no IP literals, and the base-infra names are refused as
proposals (already granted; proposing them only creates a foothold).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from urllib.parse import urlsplit

from langgraph.config import get_config

from assist.egress.store import (EgressRequest, EgressStore, REVOKED_ONLY,
                                 request_key)

_HOST_RE = re.compile(r"[a-z0-9]([a-z0-9.-]{0,251}[a-z0-9])?")
# Base-infra names never proposable: already granted host-wide, and a grant
# record for them would only be a rebinding-adjacent foothold.
_INFRA_HOSTS = {"host.docker.internal", "10.0.0.1"}
PER_THREAD_PENDING_CAP = 3


def _thread_id() -> str | None:
    try:
        tid = ((get_config() or {}).get("configurable") or {}).get("thread_id")
    except RuntimeError:   # outside a langgraph runtime (direct call in a test)
        return None
    return str(tid) if tid is not None else None


def _parse_host_port(host, port):
    """Normalize the model's habitual arg shapes: a full URL in ``host``, a
    string port, a ``host:port``. Returns (host, port, error_string)."""
    raw = str(host or "").strip().lower()
    if "//" in raw:
        u = urlsplit(raw if "://" in raw else f"//{raw}")
        raw = u.hostname or ""
        if port in (None, "", 0) and u.port:
            port = u.port
    elif raw.count(":") == 1:
        raw, _, p = raw.partition(":")
        if port in (None, "", 0):
            port = p
    raw = raw.rstrip(".")
    try:
        port = int(port) if port not in (None, "") else 443
    except (TypeError, ValueError):
        return None, None, f"'{port}' is not a valid port number."
    if not (0 < port < 65536):
        return None, None, f"port {port} is out of range (1-65535)."
    if not raw or "." not in raw or not _HOST_RE.fullmatch(raw):
        return None, None, (
            f"'{raw or host}' is not a valid hostname. Give the bare "
            "lowercase DNS name, e.g. api.example.com (internationalized "
            "names must be given in punycode/A-label form).")
    if all(re.fullmatch(r"0x[0-9a-f]+|0[0-7]*|[0-9]+", lbl)
           for lbl in raw.split(".")):
        # Every label is numeric (decimal/octal/hex) — a dotted IP encoding
        # (127.0.0.1, 0x7f.0x0.0x0.0x1, 0177.0.0.1). Advisory only: the
        # proxy's resolved-address vet is the authoritative guard.
        return None, None, ("IP addresses can't be requested — only DNS "
                            "hostnames.")
    if raw in _INFRA_HOSTS or raw.endswith(".internal"):
        return None, None, (f"{raw} is assist infrastructure and already "
                            "reachable — no request needed.")
    return raw, port, None


def egress_tools(store: EgressStore, base_hosts: frozenset[str]) -> list:
    """Build the three tools over the store + the committed base allowlist."""

    def request_egress(host: str, port: int, task: str) -> str:
        """Ask the user to approve outbound network access to one exact host
        and port for THIS thread. This does NOT open anything — an approval
        card appears in this thread and access starts only when the user
        approves it.

        ``task`` must be a complete instruction for your future self: after
        approval a follow-up turn runs with exactly this text, so include
        what to fetch and what to do with it (if the work already succeeded
        by then, just confirm). If you already know you need several hosts,
        request them ALL before ending your turn — the follow-up runs once,
        after the user resolves every pending request. After calling this,
        tell the user approval is waiting in this thread and do NOT retry
        the blocked command until approved.
        """
        h, p, err = _parse_host_port(host, port)
        if err:
            return err
        if h in base_hosts:
            return (f"{h} is already on the base allowlist (any port) — "
                    "no request needed; just retry your command.")
        tid = _thread_id()
        if not tid:
            return "Couldn't record the request: no active thread."
        existing = {r.key: r for r in store.for_thread(tid)}
        key = request_key(tid, h, p)
        rec = existing.get(key)
        if rec is not None:
            if rec.state == "pending":
                return (f"{h}:{p} is already awaiting the user's approval "
                        "in this thread — stop retrying and finish your "
                        "answer.")
            if rec.state == "declined":
                return (f"The user already DECLINED access to {h}:{p} — do "
                        "not re-ask; proceed without it.")
            return (f"{h}:{p} is already approved for this thread — just "
                    "retry your command.")
        pending = sum(1 for r in existing.values() if r.state == "pending")
        if pending >= PER_THREAD_PENDING_CAP:
            return (f"This thread already has {pending} requests awaiting "
                    "approval — tell the user, and don't request more until "
                    "those are resolved.")
        store.add_pending(EgressRequest(
            host=h, port=p,
            task=" ".join(str(task or "").split())[:500],
            origin_tid=tid,
            created_at=datetime.now(timezone.utc).isoformat()))
        return (f"Recorded — access to {h}:{p} now awaits the user's "
                "approval in this thread. Finish your answer, tell the user "
                "approval is needed, and do NOT retry until approved.")

    def list_allowed_hosts() -> str:
        """List every host this thread can currently reach: the operator's
        base allowlist (any port; managed only via the committed config) and
        this thread's own approved grants with their remaining lifetime."""
        tid = _thread_id()
        lines = [f"- {h} (any port; base allowlist, operator-managed)"
                 for h in sorted(base_hosts)]
        if tid:
            now = datetime.now(timezone.utc)
            for r in sorted(store.for_thread(tid), key=lambda r: r.key):
                if r.state != "approved":
                    continue
                if r.expires_at == REVOKED_ONLY:
                    left = "until removed or this thread is deleted"
                else:
                    try:
                        mins = max(0, int((datetime.fromisoformat(r.expires_at)
                                           - now).total_seconds() // 60))
                        left = f"~{mins} min left"
                    except Exception:
                        left = "expiring"
                lines.append(f"- {r.host}:{r.port} (this thread's grant, {left})")
        return "Hosts reachable from this thread:\n" + "\n".join(lines)

    def remove_allowed_host(host: str, port: int) -> str:
        """Remove one of THIS thread's approved egress grants once it is no
        longer needed (good practice after finishing with a host). Only this
        thread's own grants can be removed — the base allowlist is managed
        by the operator via the committed config, not from here."""
        h, p, err = _parse_host_port(host, port)
        if err:
            return err
        if h in base_hosts:
            return (f"{h} is on the operator-managed base allowlist — it "
                    "can't be removed from a thread.")
        tid = _thread_id()
        if not tid:
            return "No active thread."
        if store.revoke(request_key(tid, h, p)):
            return f"Removed this thread's access to {h}:{p}."
        return f"This thread has no grant for {h}:{p} — nothing to remove."

    return [request_egress, list_allowed_hosts, remove_allowed_host]
