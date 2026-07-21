#!/bin/bash
# Build-time smoke for the sandbox egress allowlist.  Wired into
# `make sandbox-smoke` and `make deploy-sandbox-build`.  Fails the
# build if:
#
#   - A non-allowlisted hostname returns any 2xx/3xx (allowlist leak)
#   - A direct-IP CONNECT succeeds (DNS bypass)
#   - A raw TCP connect to an off-allowlist endpoint succeeds
#     (would mean the sandbox network isn't internal)
#   - A mixed-case allowlisted hostname fails (case-insensitive match broken)
#   - HTTP-via-proxy is unreachable for an allowed host, or reachable
#     for a denied one (covers the absolute-URL request-line path)
#   - `pip install -r requirements.txt` (the same install command
#     test_dev_agent_runs_eval uses) fails through the proxy
#
# Both images must already be built: `assist-sandbox` and
# `assist-egress-proxy`.  This harness creates a temporary internal
# network, brings the proxy up on it, runs probes inside a sandbox
# container attached to it, then tears everything down.

NETWORK="assist-egress-smoke-$$"
PROXY="assist-egress-proxy-smoke-$$"
HOST_DIR=$(mktemp -d)
APPROVALS_DIR=$(mktemp -d)
SANDBOX2="assist-egress-smoke-sb-$$"

# Mount the real requirements.txt + pyproject.toml + assist source so
# the positive case probes EXACTLY what dev-agent's eval install does
# (`pip install -r requirements.txt -e .`).  Drift to a non-allowlisted
# host (a future `git+https://github.com/...` line, a private index,
# etc.) trips this smoke.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cp "$REPO_DIR/requirements.txt" "$HOST_DIR/"
cp "$REPO_DIR/pyproject.toml" "$HOST_DIR/" 2>/dev/null || true
cp -r "$REPO_DIR/assist" "$HOST_DIR/" 2>/dev/null || true

# The container runs as --user 1000:1000 (the `sandbox` user baked
# into the image).  In production this is the same uid as the deploy
# user and the bind-mount owner — three-way alignment that makes
# /home/sandbox writable and pip --user land in /home/sandbox/.local
# the same way it does in prod.
#
# On a host where the runner's uid isn't 1000 (e.g. ubuntu-latest
# GitHub runners = 1001), the bind-mount belongs to the runner uid
# but the container reads it as uid 1000.  Without chmod, traversal
# fails.  chmod 0755 on the dir + a+rX recursively closes that gap;
# capital X gives x to dirs only, not regular files.
#
# Tried `--user $(id -u):$(id -g)` (commit d3931ed) — broke pip
# --user because uid 1001 has no /etc/passwd entry in the image so
# $HOME resolves to /, and /.local isn't writable.  Going with the
# chmod approach keeps the smoke uid-aligned with production.
chmod 0755 "$HOST_DIR"
chmod -R a+rX "$HOST_DIR"

cleanup() {
    docker rm -f "$PROXY" "$SANDBOX2" >/dev/null 2>&1
    docker network rm "$NETWORK" >/dev/null 2>&1
    rm -rf "$HOST_DIR" "$APPROVALS_DIR"
}
trap cleanup EXIT

echo "→ Creating internal network $NETWORK"
docker network create --internal --driver bridge "$NETWORK" >/dev/null || {
    echo "FAIL: could not create test network"; exit 1
}

echo "→ Starting proxy with PyPI-only allowlist (host.docker.internal omitted)"
# Deliberately narrow allowlist: pypi.org + files.pythonhosted.org +
# pip.pypa.io ONLY.  host.docker.internal is NOT in this list so the
# negative-case host-resolution probes have a non-allowlisted target
# distinct from random external hosts.
docker run -d \
    --name "$PROXY" \
    --network bridge \
    --add-host=host.docker.internal:host-gateway \
    -e EGRESS_ALLOWLIST="pypi.org,files.pythonhosted.org,pip.pypa.io" \
    -v "$APPROVALS_DIR":/approvals:ro \
    assist-egress-proxy >/dev/null || {
    echo "FAIL: proxy container did not start"; exit 1
}
docker network connect "$NETWORK" "$PROXY" || {
    echo "FAIL: could not attach proxy to internal network"; exit 1
}

# Give the proxy a beat to bind its listener.
for _ in 1 2 3 4 5; do
    docker logs "$PROXY" 2>&1 | grep -q "listening on" && break
    sleep 0.5
done
docker logs "$PROXY" 2>&1 | grep -q "listening on" || {
    echo "FAIL: proxy never logged 'listening on'"
    docker logs "$PROXY" 2>&1
    exit 1
}

PROXY_URL="http://${PROXY}:8888"

echo "→ Running negative + positive probes inside sandbox container"
OUTPUT=$(docker run --rm \
    --network "$NETWORK" \
    -v "$HOST_DIR":/workspace \
    --user 1000:1000 \
    -e HTTPS_PROXY="$PROXY_URL" \
    -e HTTP_PROXY="$PROXY_URL" \
    -e https_proxy="$PROXY_URL" \
    -e http_proxy="$PROXY_URL" \
    assist-sandbox bash -c '
set +e

# Helper: did curl reach upstream with a non-error response?
# Treats any 2xx/3xx as "got through" (3xx is a real upstream
# redirect, which means we reached the host).  Drops stderr to
# /dev/null so curl error text doesn'\''t pollute $status — we only
# care about the 3-digit %{http_code} (or "000" / empty on fail).
upstream_reached() {
    local s="$1"
    [[ "$s" =~ ^[23][0-9][0-9]$ ]]
}

# (1) Hostname not in allowlist — proxy should 403 (curl returns
#     non-zero or HTTP 403).  Use --max-time so a hung proxy fails
#     loudly instead of stalling the smoke gate.
status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 8 \
    https://example.com/ 2>/dev/null)
if upstream_reached "$status"; then
    echo "FAIL: example.com returned $status (allowlist not enforced)"; exit 1
fi
echo "ok  (1) https://example.com  blocked (status=$status)"

# (2) Direct IP — bypasses DNS entirely.  This is the "agent
#     hardcodes 1.1.1.1:443" exfiltration path.  Allowlist match is
#     against the CONNECT line hostname; an IP literal is never on
#     the list.
status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 8 \
    --resolve example.com:443:1.1.1.1 https://example.com/ 2>/dev/null)
if upstream_reached "$status"; then
    echo "FAIL: direct-IP CONNECT to 1.1.1.1 returned $status (DNS bypass)"; exit 1
fi
echo "ok  (2) direct-IP CONNECT  blocked (status=$status)"

# (3) Raw TCP — no proxy involvement.  If this succeeds, the sandbox
#     network is not internal=True and traffic is escaping the gate
#     entirely.  /dev/tcp is bash-builtin, no curl env to interfere.
if timeout 5 bash -c "exec 3<>/dev/tcp/1.1.1.1/443" 2>/dev/null; then
    echo "FAIL: raw TCP to 1.1.1.1:443 succeeded (network not internal)"; exit 1
fi
echo "ok  (3) raw TCP to 1.1.1.1:443  blocked"

# (4) Mixed-case allowlist hostname — DNS hostnames are case-insensitive
#     per RFC 4343.  The proxy lowercases before comparing.  Probes
#     that the comparison is consistent between client and allowlist.
status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 8 \
    https://Pypi.Org/ 2>/dev/null)
if ! upstream_reached "$status"; then
    echo "FAIL: mixed-case Pypi.Org did not reach upstream (status=$status, "
    echo "case-insensitive allowlist match likely broken)"
    exit 1
fi
echo "ok  (4) mixed-case https://Pypi.Org  allowed (status=$status)"

# (5) HTTP-via-proxy code path — the absolute-URL request line that
#     ASSIST_MODEL_URL traffic uses.  pypi over plain HTTP returns
#     301-to-https; we just need to see that we reached upstream.
status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 8 \
    --proxy "$HTTP_PROXY" http://pypi.org/ 2>/dev/null)
if ! upstream_reached "$status"; then
    echo "FAIL: HTTP-via-proxy to pypi.org did not reach upstream (status=$status)"
    exit 1
fi
echo "ok  (5) HTTP-via-proxy http://pypi.org/  reached (status=$status)"

# (6) HTTP-via-proxy denied for off-allowlist host — same code path
#     as (5) but the allowlist gate fires.  Gives both branches of
#     the HTTP path coverage.
status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 8 \
    --proxy "$HTTP_PROXY" http://example.com/ 2>/dev/null)
if upstream_reached "$status"; then
    echo "FAIL: HTTP-via-proxy to example.com reached upstream (status=$status, allowlist not enforced)"
    exit 1
fi
echo "ok  (6) HTTP-via-proxy http://example.com/  blocked (status=$status)"

# (7) Real install path — exactly what dev-agent does in
#     test_dev_agent_runs_eval.py.  Catches any drift in
#     requirements.txt that adds a non-allowlisted host.
if ! pip install --user --break-system-packages --no-cache-dir --quiet \
        -r /workspace/requirements.txt 2>/tmp/pip.err; then
    echo "FAIL: pip install -r requirements.txt via proxy failed:"
    cat /tmp/pip.err
    exit 1
fi
echo "ok  (7) pip install -r requirements.txt  succeeded via proxy"

echo "PASS"
' 2>&1)

EXIT=$?
echo "$OUTPUT"
if [ $EXIT -ne 0 ] || ! echo "$OUTPUT" | grep -q "^PASS$"; then
    echo "----- proxy logs -----"
    docker logs "$PROXY" 2>&1
    echo "----------------------"
    exit 1
fi

# ---------------------------------------------------------------------------
# Phase 2: user-approved grants (docs/2026-07-21-egress-approval-hitl.org).
# A NAMED sandbox (we need its IP for the client map) + host-side writes to
# the mounted approvals dir. Writing AFTER the proxy started also proves the
# directory-mount visibility property (a single-file mount would pin the
# pre-write inode and silently freeze approvals).
# ---------------------------------------------------------------------------
echo "→ Phase 2: approval grants (thread-scoped, fail-closed)"
docker run -d --name "$SANDBOX2" --network "$NETWORK" \
    -e HTTPS_PROXY="http://${PROXY}:8888" -e https_proxy="http://${PROXY}:8888" \
    assist-sandbox sleep 300 >/dev/null || { echo "FAIL: sandbox2 start"; exit 1; }
SB_IP=$(docker inspect -f "{{(index .NetworkSettings.Networks \"$NETWORK\").IPAddress}}" "$SANDBOX2")
[ -n "$SB_IP" ] || { echo "FAIL: no sandbox2 IP"; exit 1; }

EXP_OK=$(date -u -d "+1 hour" +%Y-%m-%dT%H:%M:%S+00:00)
EXP_OLD=$(date -u -d "-1 hour" +%Y-%m-%dT%H:%M:%S+00:00)
cat > "$APPROVALS_DIR/client-map.json" <<EOF
{"$SB_IP": "t-smoke"}
EOF
cat > "$APPROVALS_DIR/approved-hosts.json" <<EOF
{"t-smoke:example.com:443": {"host": "example.com", "port": 443, "origin_tid": "t-smoke", "expires_at": "$EXP_OK"},
 "t-smoke:httpbin.org:443": {"host": "httpbin.org", "port": 443, "origin_tid": "t-smoke", "expires_at": "$EXP_OLD"},
 "t-other:example.net:443": {"host": "example.net", "port": 443, "origin_tid": "t-other", "expires_at": "$EXP_OK"},
 "t-smoke:host.docker.internal:443": {"host": "host.docker.internal", "port": 443, "origin_tid": "t-smoke", "expires_at": "$EXP_OK"}}
EOF

probe() { docker exec "$SANDBOX2" curl -s -o /dev/null -w "%{http_code}" --max-time 8 "$1" 2>/dev/null; }

s1=$(probe https://example.com/)
case "$s1" in 2*|3*) echo "ok  (8) approved example.com reachable ($s1)";;
    *) echo "FAIL: approved example.com not reachable ($s1)"; docker logs "$PROXY" | tail -20; exit 1;; esac

s2=$(docker exec "$SANDBOX2" curl -s -o /dev/null -w "%{http_code}" --max-time 8 https://example.com:8443/ 2>/dev/null)
case "$s2" in 2*|3*) echo "FAIL: port 8443 reachable despite 443-only grant"; exit 1;;
    *) echo "ok  (9) approved host, other port blocked ($s2)";; esac

s3=$(probe https://httpbin.org/)
case "$s3" in 2*|3*) echo "FAIL: EXPIRED grant reachable ($s3)"; exit 1;;
    *) echo "ok (10) expired grant blocked ($s3)";; esac

s4=$(probe https://example.net/)
case "$s4" in 2*|3*) echo "FAIL: OTHER thread's grant usable ($s4)"; exit 1;;
    *) echo "ok (11) wrong-thread grant blocked ($s4)";; esac

s5=$(probe https://host.docker.internal/)
case "$s5" in 2*|3*) echo "FAIL: private-resolving approved host reachable ($s5)"; exit 1;;
    *) echo "ok (12) private-resolving approved host blocked ($s5)";; esac

echo "{corrupt" > "$APPROVALS_DIR/approved-hosts.json"
s6=$(probe https://example.com/)
s7=$(probe https://pypi.org/)
case "$s6" in 2*|3*) echo "FAIL: corrupt approvals still grants ($s6)"; exit 1;; esac
case "$s7" in 2*|3*) echo "ok (13) corrupt approvals: base allowed, grants denied";;
    *) echo "FAIL: corrupt approvals broke the BASE allowlist ($s7)"; exit 1;; esac

echo "PASS (approvals phase)"
exit 0
