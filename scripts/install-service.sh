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

# Create data directory
sudo mkdir -p "$ASSIST_THREADS_DIR"
sudo chown $USER:$USER "$ASSIST_THREADS_DIR"

# Build environment variables section
ENV_VARS=""
[ -n "$ASSIST_MODEL_URL" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_MODEL_URL=$ASSIST_MODEL_URL\"\n"
[ -n "$ASSIST_MODEL_NAME" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_MODEL_NAME=$ASSIST_MODEL_NAME\"\n"
[ -n "$ASSIST_API_KEY" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_API_KEY=$ASSIST_API_KEY\"\n"
[ -n "$ASSIST_CONTEXT_LEN" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_CONTEXT_LEN=$ASSIST_CONTEXT_LEN\"\n"
[ -n "$ASSIST_DOMAIN" ] && ENV_VARS="${ENV_VARS}Environment=\"ASSIST_DOMAIN=$ASSIST_DOMAIN\"\n"

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
