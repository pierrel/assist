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

# Create data directory + migrate ownership.
#
# The non-root-sandbox layer
# (docs/2026-05-08-restrict-git-real-via-non-root-sandbox.org)
# requires every thread workspace to be owned by the invoking user,
# not root.  Pre-existing thread workspaces from before that layer
# still hold root-owned files inside (sandbox used to run as root);
# without a recursive chown they'd silently fail
# SandboxManager.get_sandbox_backend's "uid != 0" check on first
# turn after deploy.  The recursive chown is idempotent — re-running
# on already-migrated workspaces is a no-op — so it's safe to run
# on every install, not just first install.
sudo mkdir -p "$ASSIST_THREADS_DIR"
sudo chown -R $USER:$USER "$ASSIST_THREADS_DIR"

# Build environment variables section
ENV_VARS=""
[ -n "$ASSIST_PORT" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_PORT=$ASSIST_PORT\"\n"
[ -n "$ASSIST_MODEL_URL" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_MODEL_URL=$ASSIST_MODEL_URL\"\n"
[ -n "$ASSIST_DOMAINS" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_DOMAINS=$ASSIST_DOMAINS\"\n"
[ -n "$ASSIST_SEARCH_URL" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_SEARCH_URL=$ASSIST_SEARCH_URL\"\n"
[ -n "$ASSIST_ROUTING_URL" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_ROUTING_URL=$ASSIST_ROUTING_URL\"\n"
[ -n "$ASSIST_GEOCODER_URL" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_GEOCODER_URL=$ASSIST_GEOCODER_URL\"\n"
[ -n "$ASSIST_SSL_CERT" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_SSL_CERT=$ASSIST_SSL_CERT\"\n"
[ -n "$ASSIST_SSL_KEY" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_SSL_KEY=$ASSIST_SSL_KEY\"\n"
[ -n "$ASSIST_SMS_SECRET" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_SMS_SECRET=$ASSIST_SMS_SECRET\"\n"
[ -n "$ASSIST_SMS_OUTBOUND_URL" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_SMS_OUTBOUND_URL=$ASSIST_SMS_OUTBOUND_URL\"\n"

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
