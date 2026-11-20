#!/usr/bin/env bash
#
# Smoke test for the assist web UI.
# Exercises the core user flow: create thread → send message → wait for
# response → send follow-up → wait for response → delete thread.
#
# Usage:
#   make smoke          # existing Deep Agents two-turn smoke
#   ASSIST_SMOKE_HERMES=1 make smoke  # also exercise enabled Hermes with a file write
#   ./scripts/smoke_test.sh [base_url]
#
set -euo pipefail

# $1 wins; then ASSIST_BASE_URL.  Match the local server's TLS configuration
# when neither is provided.
if [[ -n "${1:-}" ]]; then
    BASE_URL="$1"
elif [[ -n "${ASSIST_BASE_URL:-}" ]]; then
    BASE_URL="$ASSIST_BASE_URL"
elif [[ -n "${ASSIST_SSL_CERT:-}" && -n "${ASSIST_SSL_KEY:-}" ]]; then
    BASE_URL="https://localhost:${ASSIST_PORT:-5050}"
else
    BASE_URL="http://localhost:${ASSIST_PORT:-5050}"
fi
CURL=(curl --connect-timeout 5 --max-time 30)
POLL_INTERVAL=3      # seconds between polls
POLL_TIMEOUT=300     # max seconds to wait for a response

log()  { printf "\033[1;34m[smoke]\033[0m %s\n" "$*"; }
pass() { printf "\033[1;32m[PASS]\033[0m  %s\n" "$*"; }
fail() { printf "\033[1;31m[FAIL]\033[0m  %s\n" "$*"; exit 1; }

THREAD_ID=""

cleanup() {
    if [ -n "$THREAD_ID" ]; then
        log "Cleaning up thread $THREAD_ID ..."
        "${CURL[@]}" -sf -X POST "$BASE_URL/thread/$THREAD_ID/delete" \
             -o /dev/null -w '' 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ------------------------------------------------------------------
# 1. Verify the server is up
# ------------------------------------------------------------------
log "Checking server at $BASE_URL ..."
HTTP_CODE=$("${CURL[@]}" -sf -o /dev/null -w "%{http_code}" "$BASE_URL/" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" != "200" ]; then
    fail "Server not reachable at $BASE_URL (HTTP $HTTP_CODE)"
fi
pass "Server is up"

# ------------------------------------------------------------------
# 2. Create a new thread with an initial message
# ------------------------------------------------------------------
log "Creating new thread with initial message ..."
RESPONSE=$("${CURL[@]}" -sf -X POST "$BASE_URL/threads/with-message" \
    -F "text=This is an automated smoke test. Please respond with a short greeting." \
    -o /dev/null -w "%{redirect_url}" 2>/dev/null)

# Extract thread ID from redirect URL (e.g. http://host/thread/20260303...-abcdef)
THREAD_ID=$(echo "$RESPONSE" | grep -oP '/thread/\K[^/]+$' || true)
if [ -z "$THREAD_ID" ]; then
    fail "Could not extract thread ID from redirect: $RESPONSE"
fi
pass "Thread created: $THREAD_ID"

# ------------------------------------------------------------------
# 3. Wait for the assistant to respond
# ------------------------------------------------------------------
wait_for_assistant() {
    local msg_num=$1
    local elapsed=0
    local started=$SECONDS
    local ready_required=${2:-false}
    log "Waiting for assistant response #$msg_num (timeout ${POLL_TIMEOUT}s) ..."
    while [ $elapsed -lt $POLL_TIMEOUT ]; do
        if [ "$ready_required" = true ]; then
            STATUS=$("${CURL[@]}" -sf "$BASE_URL/thread/$THREAD_ID/status") || fail "Could not read Hermes status"
            STAGE=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("stage", ""))' <<< "$STATUS")
            case "$STAGE" in
                error|cancelled|timeout) fail "Hermes ended with status $STAGE" ;;
            esac
        fi
        PAGE=$("${CURL[@]}" -sf "$BASE_URL/thread/$THREAD_ID" 2>/dev/null || echo "")
        # Count assistant message bubbles in the HTML
        COUNT=$(echo "$PAGE" | grep -c 'class="msg assistant"' || true)
        if [ "$COUNT" -ge "$msg_num" ] && { [ "$ready_required" != true ] || [ "$STAGE" = ready ]; }; then
            pass "Assistant response #$msg_num received (after ${elapsed}s)"
            return 0
        fi
        sleep "$POLL_INTERVAL"
        elapsed=$((SECONDS - started))
    done
    fail "Timed out waiting for assistant response #$msg_num after ${POLL_TIMEOUT}s"
}

wait_for_assistant 1

# ------------------------------------------------------------------
# 4. Send a follow-up message
# ------------------------------------------------------------------
log "Sending follow-up message ..."
"${CURL[@]}" -sf -X POST "$BASE_URL/thread/$THREAD_ID/message" \
    -F "text=Thanks! Can you repeat what you just said in one sentence?" \
    -o /dev/null 2>/dev/null
pass "Follow-up message sent"

# ------------------------------------------------------------------
# 5. Wait for second assistant response
# ------------------------------------------------------------------
wait_for_assistant 2

# ------------------------------------------------------------------
# 6. Delete the thread
# ------------------------------------------------------------------
log "Deleting thread $THREAD_ID ..."
DELETED_ID="$THREAD_ID"
DEL_CODE=$("${CURL[@]}" -sf -X POST "$BASE_URL/thread/$THREAD_ID/delete" \
    -o /dev/null -w "%{http_code}" 2>/dev/null || echo "000")
if [ "$DEL_CODE" = "200" ] || [ "$DEL_CODE" = "303" ]; then
    THREAD_ID=""  # only forget the owned thread after successful deletion
    pass "Thread deleted"
else
    fail "Delete returned HTTP $DEL_CODE"
fi

# ------------------------------------------------------------------
# 7. Verify thread is gone from listing
# ------------------------------------------------------------------
log "Verifying thread is no longer listed ..."
INDEX=$("${CURL[@]}" -sf "$BASE_URL/" 2>/dev/null || echo "")
if echo "$INDEX" | grep -q "$DELETED_ID" 2>/dev/null; then
    fail "Deleted thread still appears in index"
fi
pass "Thread no longer listed"

# Optional post-deploy capability check against the configured real provider.
# Keep its only requested file in this new thread's persistent /tmp mount, not
# the domain checkout. A final answer alone does not prove a file was written.
if [ "${ASSIST_SMOKE_HERMES:-0}" = 1 ]; then
    TOKEN="hermes-smoke-$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
    SMOKE_PATH="/tmp/$TOKEN.txt"
    log "Creating a Hermes smoke thread ..."
    RESPONSE=$("${CURL[@]}" -sf -X POST "$BASE_URL/threads/with-message" \
        -F "engine=hermes" \
        --form-string "text=Create a one-line text file at $SMOKE_PATH containing $TOKEN. Then tell me it is saved." \
        -o /dev/null -w "%{redirect_url}") || fail "Hermes thread creation failed (preview must be enabled)"
    if [[ "$RESPONSE" =~ /thread/([A-Za-z0-9_-]+)$ ]]; then
        THREAD_ID="${BASH_REMATCH[1]}"
    else
        fail "Hermes creation did not return a thread redirect"
    fi
    pass "Hermes thread created: $THREAD_ID"
    wait_for_assistant 1 true
    FILE_PAGE=$("${CURL[@]}" -sf --get "$BASE_URL/thread/$THREAD_ID/show" \
        --data-urlencode "path=$SMOKE_PATH") || fail "Hermes did not create the requested file"
    if ! python3 -c 'import html,re,sys; page=sys.stdin.read(); m=re.search(r"<pre class=\"show-text\">(.*?)</pre>",page,re.S); sys.exit(0 if m and html.unescape(m[1]).strip()==sys.argv[1] else 1)' "$TOKEN" <<< "$FILE_PAGE"; then
        fail "Hermes file content did not match the smoke token"
    fi
    pass "Hermes provider, file write and final HTTP reply completed"
    DEL_CODE=$("${CURL[@]}" -s -X POST "$BASE_URL/thread/$THREAD_ID/delete" -o /dev/null -w "%{http_code}") || fail "Hermes smoke cleanup failed"
    if [ "$DEL_CODE" != 200 ] && [ "$DEL_CODE" != 303 ]; then
        fail "Hermes smoke deletion returned HTTP $DEL_CODE"
    fi
    GONE_CODE=$("${CURL[@]}" -s -o /dev/null -w "%{http_code}" "$BASE_URL/thread/$THREAD_ID/status") || fail "Could not verify Hermes cleanup"
    [ "$GONE_CODE" = 404 ] || fail "Hermes smoke thread still exists after deletion"
    THREAD_ID=""
    pass "Hermes smoke thread and its temporary file removed"
fi

echo ""
printf "\033[1;32m✓ All smoke tests passed.\033[0m\n"
