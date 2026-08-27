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
from langgraph.types import interrupt

from assist.egress.store import (EgressRequest, EgressStore, EgressWaiter,
                                 remaining_lifetime, request_key)

_HOST_RE = re.compile(r"[a-z0-9]([a-z0-9.-]{0,251}[a-z0-9])?")
# Per-label shape (RFC 1035-ish): no empty labels (a..b), ≤63 chars, no
# leading/trailing hyphen — an unresolvable host would only make a confusing
# approval card.
_LABEL_RE = re.compile(r"[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?")
# Base-infra names never proposable: already granted host-wide, and a grant
# record for them would only be a rebinding-adjacent foothold.
_INFRA_HOSTS = {"host.docker.internal", "10.0.0.1"}
EGRESS_ORIGIN_THREAD_ID = "egress_origin_thread_id"
EGRESS_WAITER_THREAD_ID = "egress_waiter_thread_id"
EGRESS_WAITER_RUN_ID = "egress_waiter_run_id"


def _thread_id() -> str | None:
    try:
        tid = ((get_config() or {}).get("configurable") or {}).get("thread_id")
    except RuntimeError:   # outside a langgraph runtime (direct call in a test)
        return None
    return str(tid) if tid is not None else None


def _child_waiter() -> EgressWaiter | None:
    """Return a web executor-injected child identity, never model arguments."""
    try:
        configurable = ((get_config() or {}).get("configurable") or {})
    except RuntimeError:
        return None
    task_id = configurable.get(EGRESS_WAITER_THREAD_ID)
    run_id = configurable.get(EGRESS_WAITER_RUN_ID)
    if not isinstance(task_id, str) or not isinstance(run_id, str):
        return None
    return EgressWaiter(task_id, run_id)


def _origin_thread_id() -> str | None:
    try:
        configurable = ((get_config() or {}).get("configurable") or {})
    except RuntimeError:
        return _thread_id()
    origin = configurable.get(EGRESS_ORIGIN_THREAD_ID)
    return origin if isinstance(origin, str) and origin else _thread_id()


def _parse_host_port(host, port):
    """Normalize the model's habitual arg shapes: a full URL in ``host``, a
    string port, a ``host:port``. Returns (host, port, error_string)."""
    raw = str(host or "").strip().lower()
    default_port = 443
    if "//" in raw:
        u = urlsplit(raw if "://" in raw else f"//{raw}")
        raw = u.hostname or ""
        if port in (None, "", 0) and u.port:
            port = u.port
        if u.scheme == "http":
            # an http:// URL's implied port is 80 — a :443 grant would not
            # match the command the agent retries
            default_port = 80
    elif raw.count(":") == 1:
        raw, _, p = raw.partition(":")
        if port in (None, "", 0):
            port = p
    raw = raw.rstrip(".")
    try:
        # 0 means "unset" everywhere in this function, incl. here
        port = int(port) if port not in (None, "", 0) else default_port
    except (TypeError, ValueError):
        return None, None, f"'{port}' is not a valid port number."
    if not (0 < port < 65536):
        return None, None, f"port {port} is out of range (1-65535)."
    if (not raw or "." not in raw or not _HOST_RE.fullmatch(raw)
            or not all(_LABEL_RE.fullmatch(lbl) for lbl in raw.split("."))):
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


def egress_tools(store: EgressStore, base_hosts: frozenset[str],
                 thread_dir=None) -> list:
    """Build the three tools over the store + the committed base allowlist.
    ``thread_dir(tid) -> path`` (optional, the notify_tools factory shape)
    enables the event trail (egress_requested / egress_revoked); absent —
    CLI/evals — events are skipped."""

    def _event(tid, kind, **fields):
        if thread_dir is None:
            return
        from assist.events.thread_log import append_event
        append_event(thread_dir(tid), kind, **fields)

    def request_egress(host: str, port: int, task: str) -> str:
        """Ask the user to approve outbound network access to one exact host
        and port for THIS thread. This does NOT open anything — an approval
        card appears in this thread and access starts only when the user
        approves it.

        ``task`` must be a complete instruction for an ordinary agent's
        follow-up turn after approval. An async child instead checkpoints here
        and the exact child resumes after the decision. If you already know
        you need several hosts, request them ALL before ending your turn — an
        ordinary follow-up runs once after the user resolves every pending
        request. After calling this, tell the user approval is waiting in this
        thread and do NOT retry the blocked command until approved.
        """
        h, p, err = _parse_host_port(host, port)
        if err:
            return err
        if h in base_hosts:
            return (f"{h} is already on the base allowlist (any port) — "
                    "no request needed; just retry your command.")
        tid = _origin_thread_id()
        if not tid:
            return "Couldn't record the request: no active thread."
        existing = {r.key: r for r in store.for_thread(tid)}
        key = request_key(tid, h, p)
        rec = existing.get(key)
        if rec is None and store.discard_expired(key):
            # ``for_thread`` correctly hides expired grants; remove the stale
            # raw record too, otherwise a resumed child would be told to retry
            # a proxy-denied connection forever.
            rec = None
        waiter = _child_waiter()
        if rec is not None:
            if rec.state == "pending":
                if waiter is not None:
                    current = store.wait_for_resolution(key, waiter)
                    if current is not None and current.state == "pending":
                        interrupt({"egress_request": key})
                        rec = store.get(key)
                    else:
                        rec = current
                    if rec is not None and rec.state != "pending":
                        if rec.state == "declined":
                            return (f"The user already DECLINED access to {h}:{p} — do "
                                    "not re-ask; proceed without it.")
                        return (f"{h}:{p} is already approved for this thread — just "
                                "retry your command.")
                return (f"{h}:{p} is already awaiting the user's approval "
                        "in this thread — stop retrying and finish your "
                        "answer.")
            if rec.state == "declined":
                return (f"The user already DECLINED access to {h}:{p} — do "
                        "not re-ask; proceed without it.")
            return (f"{h}:{p} is already approved for this thread — just "
                    "retry your command.")
        refused = store.add_pending(EgressRequest(
            host=h, port=p,
            task=" ".join(str(task or "").split())[:500],
            origin_tid=tid,
            dispatch_main=(waiter is None),
            created_at=datetime.now(timezone.utc).isoformat()))
        if refused == "thread-cap":
            return ("This thread already has several requests awaiting "
                    "approval — tell the user, and don't request more until "
                    "those are resolved.")
        if refused == "global-cap":
            return ("Too many requests are already awaiting approval across "
                    "threads — tell the user and don't request more for now.")
        if refused == "existing":
            # A sibling child won the insert race. Re-enter the existing-record
            # path so this child joins its card instead of publishing a second
            # story about the same approval.
            return request_egress(h, p, task)
        _event(tid, "egress_requested", host=h, port=p)
        if waiter is not None:
            current = store.wait_for_resolution(key, waiter)
            if current is not None and current.state == "pending":
                interrupt({"egress_request": key})
                current = store.get(key)
            if current is not None and current.state != "pending":
                if current.state == "declined":
                    return (f"The user DECLINED access to {h}:{p} — do not re-ask; "
                            "proceed without it.")
                return f"{h}:{p} is approved for this thread — retry your command."
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
            for r in sorted(store.for_thread(tid), key=lambda r: r.key):
                if r.state != "approved":
                    continue
                lines.append(f"- {r.host}:{r.port} (this thread's grant, "
                             f"{remaining_lifetime(r)})")
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
            _event(tid, "egress_revoked", host=h, port=p)
            return f"Removed this thread's access to {h}:{p}."
        return f"This thread has no grant for {h}:{p} — nothing to remove."

    return [request_egress, list_allowed_hosts, remove_allowed_host]
