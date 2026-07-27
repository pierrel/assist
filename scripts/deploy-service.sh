#!/bin/bash
# Relay deployment configuration without constructing a remote shell command.
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 HOST" >&2
    exit 2
fi

host=$1
installer=${INSTALL_SERVICE_SCRIPT:-scripts/install-service.sh}
variables=(
    DEPLOY_PATH SERVICE_NAME ASSIST_THREADS_DIR ASSIST_PORT ASSIST_MODEL_URL
    ASSIST_DOMAINS ASSIST_SEARCH_URL TAVILY_API_KEY ASSIST_ROUTING_URL
    ASSIST_GEOCODER_URL TRAVEL_INFRA_DIR ASSIST_EGRESS_APPROVALS_DIR
    ASSIST_SSL_CERT ASSIST_SSL_KEY ASSIST_SMS_SECRET ASSIST_SMS_OUTBOUND_URL
    EMAIL_RESEND_API_KEY_FILE EMAIL_FROM_ADDRESS EMAIL_FROM_NAME EMAIL_ALWAYS_CC
    ASSIST_THREAD_QUANTUM_S
)

{
    printf 'set -euo pipefail\n'
    for name in "${variables[@]}"; do
        encoded=$(printf %s "${!name-}" | base64 --wrap=0)
        printf '%s=$(printf %%s %s | base64 -d)\nexport %s\n' "$name" "$encoded" "$name"
    done
    cat "$installer"
} | ssh "$host" 'bash -s'
