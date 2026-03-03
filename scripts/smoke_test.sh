#!/usr/bin/env bash
#
# Smoke test for the assist web UI.
# Exercises the core user flow: create thread → send message → wait for
# response → send follow-up → wait for response → delete thread.
#
# Usage:
#   make smoke          # uses ASSIST_PORT from .dev.env
#   ./scripts/smoke_test.sh [base_url]
#
set -euo pipefail

BASE_URL="${1:-http://localhost:${ASSIST_PORT:-5050}}"
POLL_INTERVAL=3      # seconds between polls
POLL_TIMEOUT=300     # max seconds to wait for a response

log()  { printf "\033[1;34m[smoke]\033[0m %s\n" "$*"; }
pass() { printf "\033[1;32m[PASS]\033[0m  %s\n" "$*"; }
fail() { printf "\033[1;31m[FAIL]\033[0m  %s\n" "$*"; exit 1; }

THREAD_ID=""

cleanup() {
    if [ -n "$THREAD_ID" ]; then
        log "Cleaning up thread $THREAD_ID ..."
        curl -sf -X POST "$BASE_URL/thread/$THREAD_ID/delete" \
             -o /dev/null -w '' 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ------------------------------------------------------------------
# 1. Verify the server is up
# ------------------------------------------------------------------
log "Checking server at $BASE_URL ..."
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "$BASE_URL/" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" != "200" ]; then
    fail "Server not reachable at $BASE_URL (HTTP $HTTP_CODE)"
fi
pass "Server is up"

# ------------------------------------------------------------------
# 2. Create a new thread with an initial message
# ------------------------------------------------------------------
log "Creating new thread with initial message ..."
RESPONSE=$(curl -sf -X POST "$BASE_URL/threads/with-message" \
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
    log "Waiting for assistant response #$msg_num (timeout ${POLL_TIMEOUT}s) ..."
    while [ $elapsed -lt $POLL_TIMEOUT ]; do
        PAGE=$(curl -sf "$BASE_URL/thread/$THREAD_ID" 2>/dev/null || echo "")
        # Count assistant message bubbles in the HTML
        COUNT=$(echo "$PAGE" | grep -c 'class="msg assistant"' || true)
        if [ "$COUNT" -ge "$msg_num" ]; then
            pass "Assistant response #$msg_num received (after ${elapsed}s)"
            return 0
        fi
        sleep "$POLL_INTERVAL"
        elapsed=$((elapsed + POLL_INTERVAL))
    done
    fail "Timed out waiting for assistant response #$msg_num after ${POLL_TIMEOUT}s"
}

wait_for_assistant 1

# ------------------------------------------------------------------
# 4. Send a follow-up message
# ------------------------------------------------------------------
log "Sending follow-up message ..."
curl -sf -X POST "$BASE_URL/thread/$THREAD_ID/message" \
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
DEL_CODE=$(curl -sf -X POST "$BASE_URL/thread/$THREAD_ID/delete" \
    -o /dev/null -w "%{http_code}" 2>/dev/null || echo "000")
THREAD_ID=""  # already deleted, don't clean up again
if [ "$DEL_CODE" = "200" ] || [ "$DEL_CODE" = "303" ]; then
    pass "Thread deleted"
else
    fail "Delete returned HTTP $DEL_CODE"
fi

# ------------------------------------------------------------------
# 7. Verify thread is gone from listing
# ------------------------------------------------------------------
log "Verifying thread is no longer listed ..."
INDEX=$(curl -sf "$BASE_URL/" 2>/dev/null || echo "")
if echo "$INDEX" | grep -q "$DELETED_ID" 2>/dev/null; then
    fail "Deleted thread still appears in index"
fi
pass "Thread no longer listed"

echo ""
printf "\033[1;32m✓ All smoke tests passed.\033[0m\n"
