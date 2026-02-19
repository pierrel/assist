#!/bin/bash
# Setup passwordless sudo for deployment commands
# Run this ONCE on the remote server: ./scripts/setup-passwordless-sudo.sh

set -e

SERVICE_NAME="${SERVICE_NAME:-assist-web}"
DEPLOY_PATH="${DEPLOY_PATH:-/opt/assist}"

echo "=== Setup Passwordless Sudo for Deployment ==="
echo "This will allow deployment commands to run without password prompts"
echo ""

# Create sudoers file
SUDOERS_FILE="/etc/sudoers.d/assist-deploy"
TEMP_FILE=$(mktemp)

cat > "$TEMP_FILE" <<EOF
# Allow $USER to manage assist deployment without password
# Created by setup-passwordless-sudo.sh

# Systemd service management
$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl daemon-reload
$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl enable ${SERVICE_NAME}
$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl disable ${SERVICE_NAME}
$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl start ${SERVICE_NAME}
$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl stop ${SERVICE_NAME}
$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart ${SERVICE_NAME}
$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl status ${SERVICE_NAME}
$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl reload ${SERVICE_NAME}

# Journal logs
$USER ALL=(ALL) NOPASSWD: /usr/bin/journalctl -u ${SERVICE_NAME} *

# File operations for deployment
$USER ALL=(ALL) NOPASSWD: /usr/bin/mkdir -p /var/lib/assist
$USER ALL=(ALL) NOPASSWD: /usr/bin/mkdir -p /var/lib/assist/*
$USER ALL=(ALL) NOPASSWD: /usr/bin/chown $USER\\:$USER /var/lib/assist
$USER ALL=(ALL) NOPASSWD: /usr/bin/chown $USER\\:$USER /var/lib/assist/*
$USER ALL=(ALL) NOPASSWD: /usr/bin/tee /etc/systemd/system/${SERVICE_NAME}.service
EOF

echo "Generated sudoers configuration:"
echo "================================"
cat "$TEMP_FILE"
echo "================================"
echo ""

# Validate sudoers syntax
echo "→ Validating sudoers syntax..."
if sudo visudo -c -f "$TEMP_FILE"; then
    echo "✓ Syntax is valid"
else
    echo "✗ Syntax error in sudoers file!"
    rm "$TEMP_FILE"
    exit 1
fi

# Install sudoers file
echo "→ Installing sudoers file..."
sudo cp "$TEMP_FILE" "$SUDOERS_FILE"
sudo chmod 0440 "$SUDOERS_FILE"
rm "$TEMP_FILE"

echo "✓ Passwordless sudo configured successfully!"
echo ""
echo "You can now run deployment commands without entering a password:"
echo "  - make deploy"
echo "  - make restart"
echo "  - make status"
echo "  - make logs"
echo ""
echo "To remove this configuration:"
echo "  sudo rm $SUDOERS_FILE"
