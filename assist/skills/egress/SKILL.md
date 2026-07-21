---
name: egress
description: "Sandbox network access is restricted to an approved host allowlist — commands that reach a blocked host fail with a proxy 403. EXAMPLES — a curl/pip/git command failed with '403' and 'proxy' or 'tunnel' in the output; you need to fetch a page or API the sandbox can't reach; auditing or reducing which hosts this thread can access. MUST load when a network command is denied or before requesting new network access."
---

# Egress — restricted network access and the approval workflow

## The situation

The sandbox's only route to the network is a proxy that enforces an exact
allowlist of hosts. A command that touches any other host fails with an HTTP
403 from the proxy ("CONNECT tunnel failed", "Tunnel connection failed",
"Proxy tunneling failed"). This is a POLICY denial, not an outage — the
internet is fine; that host just isn't approved.

## When a command is denied

1. Decide whether the host is genuinely required for the user's task. Many
   denials are incidental (telemetry, analytics, CDN extras) — if the work
   can proceed without the host, proceed without it and don't request it.
2. If it IS required: call `request_egress(host, port, task)`.
   - `host` is the exact DNS hostname (from the failed command's URL).
   - `task` must be a complete instruction for your future self — after the
     user approves, a follow-up turn runs with exactly that text.
   - If you already know you need several hosts, request them ALL now: the
     follow-up runs once, after the user resolves every request.
3. Tell the user in your answer: which host, why, and that an approval card
   is waiting in this thread. Then finish your answer. Do NOT retry the
   blocked command until the approval arrives.
4. If the user declines, do not ask again for that host — proceed without
   it and say what that means for the result.

## Managing this thread's access

- `list_allowed_hosts()` — everything this thread can currently reach: the
  operator's base allowlist plus this thread's own time-limited grants.
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
