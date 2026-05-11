#!/bin/bash
# Build-time smoke for the sandbox egress allowlist.  Wired into
# `make sandbox-smoke` and `make deploy-sandbox-build`.  Fails the
# build if:
#
#   - A non-allowlisted hostname returns anything but 403/connect-fail
#   - A direct-IP connection (DNS bypass attempt) succeeds
#   - A raw TCP connect to an off-allowlist endpoint succeeds
#     (would mean the sandbox network isn't internal)
#   - `pip install` of a small allowlisted package fails through the proxy
#
# Both images must already be built: `assist-sandbox` and
# `assist-egress-proxy`.  This harness creates a temporary internal
# network, brings the proxy up on it, runs probes inside a sandbox
# container attached to it, then tears everything down.

NETWORK="assist-egress-smoke-$$"
PROXY="assist-egress-proxy-smoke-$$"
HOST_DIR=$(mktemp -d)

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

# Run the container as the bind-mount owner — the same pattern
# SandboxManager.get_sandbox_backend uses in production (it reads
# stat(work_dir).st_uid and passes it via --user).  This means the
# smoke runs under the same uid alignment as prod instead of the
# hardcoded uid 1000 + world-readable workaround.
HOST_UID=$(id -u)
HOST_GID=$(id -g)

cleanup() {
    docker rm -f "$PROXY" >/dev/null 2>&1
    docker network rm "$NETWORK" >/dev/null 2>&1
    rm -rf "$HOST_DIR"
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
    --user "$HOST_UID:$HOST_GID" \
    -e HTTPS_PROXY="$PROXY_URL" \
    -e HTTP_PROXY="$PROXY_URL" \
    -e https_proxy="$PROXY_URL" \
    -e http_proxy="$PROXY_URL" \
    assist-sandbox bash -c '
set +e

# (1) Hostname not in allowlist — proxy should 403 (curl returns
#     non-zero or HTTP 403).  Use --max-time so a hung proxy fails
#     loudly instead of stalling the smoke gate.
status=$(curl -sS -o /dev/null -w "%{http_code}" --max-time 8 \
    https://example.com/ 2>&1)
if [ "$status" = "200" ]; then
    echo "FAIL: example.com returned 200 (allowlist not enforced)"; exit 1
fi
echo "ok  (1) https://example.com  blocked (status=$status)"

# (2) Direct IP — bypasses DNS entirely.  This is the "agent
#     hardcodes 1.1.1.1:443" exfiltration path.  Allowlist match is
#     against the CONNECT line hostname; an IP literal is never on
#     the list.
status=$(curl -sS -o /dev/null -w "%{http_code}" --max-time 8 \
    --resolve example.com:443:1.1.1.1 https://example.com/ 2>&1)
if [ "$status" = "200" ]; then
    echo "FAIL: direct-IP CONNECT to 1.1.1.1 succeeded (DNS bypass)"; exit 1
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
status=$(curl -sS -o /dev/null -w "%{http_code}" --max-time 8 \
    https://Pypi.Org/ 2>&1)
if echo "$status" | grep -q "403"; then
    echo "FAIL: mixed-case Pypi.Org returned 403 (case-insensitive match broken)"
    exit 1
fi
echo "ok  (4) mixed-case https://Pypi.Org  allowed (status=$status)"

# (5) HTTP-via-proxy code path — the absolute-URL request line that
#     ASSIST_MODEL_URL traffic uses.  pypi over plain HTTP returns
#     301-to-https; either way, NOT 000 (unreachable) and NOT 403.
status=$(curl -sS -o /dev/null -w "%{http_code}" --max-time 8 \
    --proxy "$HTTP_PROXY" http://pypi.org/ 2>&1)
if [ "$status" = "000" ]; then
    echo "FAIL: HTTP-via-proxy to pypi.org reachability failed (status=000)"
    exit 1
fi
if [ "$status" = "403" ]; then
    echo "FAIL: HTTP-via-proxy to pypi.org refused (allowlisted host got 403)"
    exit 1
fi
echo "ok  (5) HTTP-via-proxy http://pypi.org/  reached (status=$status)"

# (6) HTTP-via-proxy denied for off-allowlist host — same code path
#     as (5) but the allowlist gate fires.  Gives both branches of
#     the HTTP path coverage.
status=$(curl -sS -o /dev/null -w "%{http_code}" --max-time 8 \
    --proxy "$HTTP_PROXY" http://example.com/ 2>&1)
if echo "$status" | grep -qE "^(200|301|302|3[0-9]+)$"; then
    echo "FAIL: HTTP-via-proxy to example.com succeeded (status=$status, allowlist not enforced)"
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
exit 0
