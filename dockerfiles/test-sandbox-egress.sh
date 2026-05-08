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

echo "→ Starting proxy with restrictive allowlist (pypi only — host.docker.internal omitted)"
# Deliberately narrow allowlist: pypi.org + files.pythonhosted.org only.
# This makes the negative test specific — host.docker.internal is NOT
# allowed in this run, so any request to it must 403.
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
    --user 1000:1000 \
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

# (4) Positive: pip install of a small allowlisted package.  This is
#     the "dev-agent must keep working" contract.  --no-cache-dir
#     forces a fresh download (cache could mask a broken proxy).
if ! pip install --user --break-system-packages --no-cache-dir --quiet \
        "requests==2.32.3" 2>/tmp/pip.err; then
    echo "FAIL: pip install via proxy failed:"
    cat /tmp/pip.err
    exit 1
fi
if ! python3 -c "import requests; print(requests.__version__)" >/dev/null 2>&1; then
    echo "FAIL: requests imported but version probe failed"; exit 1
fi
echo "ok  (4) pip install requests==2.32.3  succeeded via proxy"

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
