"""The denial guidance — the infrastructure↔agent contract text, centralized.

Pierre's review (PR #200): this text is how the egress gate explains itself
to the agent, so it lives HERE — one reviewable place — never embedded in
proxy implementation code. Consumers:

- ``DockerSandboxBackend.execute`` prepends ``EGRESS_DENIED_GUIDANCE`` to a
  command result matching the proxy-denial signature (the HTTPS path: curl
  discards CONNECT response bodies, so only the host side can steer).
- The egress proxy's 403 body is ``EGRESS_DENY_BODY``, delivered via the
  ``EGRESS_DENY_BODY`` env var at proxy-container create (the
  ``EGRESS_ALLOWLIST`` wire-protocol precedent) — plain-HTTP clients DO
  surface bodies, and log readers see it too.

``EGRESS_DENIED_GUIDANCE`` is an exact constant on purpose: a future
guard-of-last-resort can equality-count it in the message tail (the
``_SEARCH_UNAVAILABLE_MESSAGE`` breaker pattern) without any restructuring.
"""

EGRESS_DENIED_GUIDANCE = (
    "[Network access denied by the egress allowlist] Outbound network "
    "access from the sandbox is restricted to an approved host list — this "
    "is a policy denial, not an outage. If a blocked host is genuinely "
    "required, call request_egress(host, port, task) to ask the user to "
    "approve it (an approval card appears in this thread), tell the user "
    "you are waiting on their approval, and do NOT retry the command until "
    "approved. Use list_allowed_hosts() to see what this thread can already "
    "reach. Load the 'egress' skill for the full workflow.\n\n"
)

EGRESS_DENY_BODY = (
    "assist egress: host not on the allowlist. This is a policy denial. "
    "Agent: call request_egress(host, port, task) to ask the user to "
    "approve this host; do not retry until approved.\n"
)
