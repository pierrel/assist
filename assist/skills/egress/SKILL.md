---
name: egress
description: "Ordinary sandbox commands use exact-host egress approvals; the isolated browser also enforces a separate public/internal policy. EXAMPLES — curl/pip/git gets a proxy 403; browser_probe reports host_not_approved; auditing or reducing this thread's grants. MUST load before requesting new network access."
allowed-tools: request_egress list_allowed_hosts remove_allowed_host
---

# Egress — restricted network access and the approval workflow

## The situation

Ordinary sandbox commands reach the network through an exact-host proxy.
A denied host commonly produces a proxy HTTP 403 ("CONNECT tunnel failed",
"Tunnel connection failed", "Proxy tunneling failed"). The isolated browser
uses that proxy too, but has an additional public/internal destination policy:
an HTTP 403 alone does not mean a host is approvable. For browser failures,
request a grant only when `browser_probe` reports `host_not_approved` for the
exact observed host and port. `browser_internal_policy` cannot be fixed by
requesting egress.

## When a command is denied

1. Decide whether the host is genuinely required for the user's task. Many
   denials are incidental (telemetry, analytics, CDN extras) — if the work
   can proceed without the host, proceed without it and don't request it.
2. If it IS required and the denial is `host_not_approved` (or an ordinary
   shell proxy denial): call `request_egress(host, port, task)`.
   - `host` is the exact DNS hostname (from the failed command's URL).
   - `task` must be a complete instruction for an ordinary agent's future
     follow-up. An async child instead parks here and that exact child resumes
     after the user decides.
   - If you already know you need several hosts, request them ALL now: the
     follow-up runs once, after the user resolves every request.
3. An ordinary agent tells the user which host and why, then finishes its
   answer. An async child pauses at the request and has no user-facing reply;
   the visible parent thread owns the approval card. Neither retries the
   blocked command until approval arrives.
4. If the user declines, do not ask again for that host — proceed without
   it and say what that means for the result.

## Managing this thread's access

- `list_allowed_hosts()` — the operator's base allowlist plus this thread's
  approved grants. It does not override the isolated browser's internal-host
  policy or promise that every destination is reachable.
- `remove_allowed_host(host, port)` — drop one of this thread's grants once
  you're done with it. Good practice: when a granted host has served its
  purpose, remove it.
- Grants are scoped to THIS thread and expire (1 hour, unless the user chose
  "always allow for this thread"). Permanent, thread-independent access is
  the operator's call — the committed allowlist — not something you can
  request from here.

## What NOT to do

- Don't retry a denied command in a loop — the denial is deterministic.
- Don't work around the proxy (IP literals, alternate ports, mirrors of the
  same blocked content) — the restriction is the user's policy.
- Don't request hosts speculatively "in case" — request exactly what the
  task in front of you needs.
