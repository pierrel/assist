#!/bin/bash
# Install systemd service for Assist Web
# This script is executed on the remote server via SSH
# Environment variables are passed from the Makefile

set -e

# Default values if not provided
DEPLOY_PATH="${DEPLOY_PATH:-/opt/assist}"
SERVICE_NAME="${SERVICE_NAME:-assist-web}"
ASSIST_THREADS_DIR="${ASSIST_THREADS_DIR:-/var/lib/assist/threads}"

echo "Installing service: $SERVICE_NAME"
echo "Deploy path: $DEPLOY_PATH"
echo "Data directory: $ASSIST_THREADS_DIR"

# Thread storage is provisioned with the service user's ownership. Do not create it
# through sudo here: a root-owned fresh directory makes the running service unable to
# create its database or thread directories.
if [ ! -d "$ASSIST_THREADS_DIR" ] || [ ! -w "$ASSIST_THREADS_DIR" ]; then
    echo "Thread data directory must already exist and be writable: $ASSIST_THREADS_DIR" >&2
    exit 1
fi

# Build environment variables section
ENV_VARS=""
[ -n "$ASSIST_PORT" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_PORT=$ASSIST_PORT\"\n"
[ -n "$ASSIST_MODEL_URL" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_MODEL_URL=$ASSIST_MODEL_URL\"\n"
[ -n "$ASSIST_DOMAINS" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_DOMAINS=$ASSIST_DOMAINS\"\n"
[ -n "$ASSIST_SEARCH_URL" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_SEARCH_URL=$ASSIST_SEARCH_URL\"\n"
[ -n "$TAVILY_API_KEY" ] && ENV_VARS="${ENV_VARS}Environment=\"TAVILY_API_KEY=$TAVILY_API_KEY\"\n"
[ -n "$ASSIST_ROUTING_URL" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_ROUTING_URL=$ASSIST_ROUTING_URL\"\n"
[ -n "$ASSIST_GEOCODER_URL" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_GEOCODER_URL=$ASSIST_GEOCODER_URL\"\n"
# TRAVEL_INFRA_DIR enables the geo region-download feature (registry/catalog/proposals +
# the provisioning scripts). Unset → the geo tools + /geo page are simply absent.
[ -n "$TRAVEL_INFRA_DIR" ] && ENV_VARS="${ENV_VARS}Environment=\"TRAVEL_INFRA_DIR=$TRAVEL_INFRA_DIR\"\n"
# ASSIST_EGRESS_APPROVALS_DIR enables the egress approval HITL (requests store +
# the proxy-mounted approvals subdir); unset = feature dormant. Must NOT be
# under ASSIST_THREADS_DIR (both the web wiring and the proxy mount refuse it).
[ -n "$ASSIST_EGRESS_APPROVALS_DIR" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_EGRESS_APPROVALS_DIR=$ASSIST_EGRESS_APPROVALS_DIR\"\n"
[ -n "$ASSIST_SSL_CERT" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_SSL_CERT=$ASSIST_SSL_CERT\"\n"
[ -n "$ASSIST_SSL_KEY" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_SSL_KEY=$ASSIST_SSL_KEY\"\n"
[ -n "$ASSIST_SMS_SECRET" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_SMS_SECRET=$ASSIST_SMS_SECRET\"\n"
[ -n "$ASSIST_THREAD_QUANTUM_S" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_THREAD_QUANTUM_S=$ASSIST_THREAD_QUANTUM_S\"\n"
[ -n "$ASSIST_SMS_OUTBOUND_URL" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_SMS_OUTBOUND_URL=$ASSIST_SMS_OUTBOUND_URL\"\n"
[ -n "$URGENT_SMS_RECIPIENT" ] && ENV_VARS="${ENV_VARS}Environment=\"URGENT_SMS_RECIPIENT=$URGENT_SMS_RECIPIENT\"\n"
[ -n "$URGENT_SMS_THREAD_URL_BASE" ] && ENV_VARS="${ENV_VARS}Environment=\"URGENT_SMS_THREAD_URL_BASE=$URGENT_SMS_THREAD_URL_BASE\"\n"
[ -n "$ASSIST_VOICE_SECRET" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_VOICE_SECRET=$ASSIST_VOICE_SECRET\"\n"
[ -n "$ASSIST_VOICE_PIN" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_VOICE_PIN=$ASSIST_VOICE_PIN\"\n"
[ -n "$ASSIST_VOICE_CALLERS" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_VOICE_CALLERS=$ASSIST_VOICE_CALLERS\"\n"
[ -n "$ASSIST_VOICE_CALL_LOG_DIR" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_VOICE_CALL_LOG_DIR=$ASSIST_VOICE_CALL_LOG_DIR\"\n"
[ -n "$ASSIST_VOICE_PIPER_MODEL" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_VOICE_PIPER_MODEL=$ASSIST_VOICE_PIPER_MODEL\"\n"
[ -n "$ASSIST_VOICE_WHISPER_MODEL" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_VOICE_WHISPER_MODEL=$ASSIST_VOICE_WHISPER_MODEL\"\n"
[ -n "$EMAIL_RESEND_API_KEY_FILE" ] && ENV_VARS="${ENV_VARS}Environment=\"EMAIL_RESEND_API_KEY_FILE=$EMAIL_RESEND_API_KEY_FILE\"\n"
[ -n "$EMAIL_FROM_ADDRESS" ] && ENV_VARS="${ENV_VARS}Environment=\"EMAIL_FROM_ADDRESS=$EMAIL_FROM_ADDRESS\"\n"
[ -n "$EMAIL_FROM_NAME" ] && ENV_VARS="${ENV_VARS}Environment=\"EMAIL_FROM_NAME=$EMAIL_FROM_NAME\"\n"
[ -n "$EMAIL_ALWAYS_CC" ] && ENV_VARS="${ENV_VARS}Environment=\"EMAIL_ALWAYS_CC=$EMAIL_ALWAYS_CC\"\n"

# Generate service file from template and install it
cat "$DEPLOY_PATH/scripts/assist-web.service.template" | \
    sed "s|{{USER}}|$USER|g" | \
    sed "s|{{DEPLOY_PATH}}|$DEPLOY_PATH|g" | \
    sed "s|{{ASSIST_THREADS_DIR}}|$ASSIST_THREADS_DIR|g" | \
    sed "s|{{ENVIRONMENT_VARS}}|$ENV_VARS|g" | \
    sudo tee "/etc/systemd/system/$SERVICE_NAME.service" > /dev/null

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"

echo "✓ Service installed successfully"
echo "Start with: sudo systemctl start $SERVICE_NAME"
