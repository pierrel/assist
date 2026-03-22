#!/bin/bash
# Setup passwordless sudo for vLLM service management
# Run this ONCE on the GPU server: bash -s < scripts/setup-vllm-sudo.sh

set -e

VLLM_SERVICE_NAME="${VLLM_SERVICE_NAME:-vllm-serve}"

echo "=== Setup Passwordless Sudo for vLLM Service Management ==="
echo ""

SUDOERS_FILE="/etc/sudoers.d/vllm-service"
TEMP_FILE=$(mktemp)

cat > "$TEMP_FILE" <<EOF
# Allow $USER to manage the vLLM inference service without password
# Created by scripts/setup-vllm-sudo.sh

# Systemd service management
$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl daemon-reload
$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl enable ${VLLM_SERVICE_NAME}
$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl start ${VLLM_SERVICE_NAME}
$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl stop ${VLLM_SERVICE_NAME}
$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart ${VLLM_SERVICE_NAME}
$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl status ${VLLM_SERVICE_NAME}

# Journal logs
$USER ALL=(ALL) NOPASSWD: /usr/bin/journalctl -u ${VLLM_SERVICE_NAME} *

# Service file installation
$USER ALL=(ALL) NOPASSWD: /usr/bin/tee /etc/systemd/system/${VLLM_SERVICE_NAME}.service
EOF

echo "Generated sudoers configuration:"
echo "================================"
cat "$TEMP_FILE"
echo "================================"
echo ""

echo "→ Validating sudoers syntax..."
if sudo visudo -c -f "$TEMP_FILE"; then
    echo "✓ Syntax is valid"
else
    echo "✗ Syntax error in sudoers file!"
    rm "$TEMP_FILE"
    exit 1
fi

echo "→ Installing sudoers file to $SUDOERS_FILE..."
sudo cp "$TEMP_FILE" "$SUDOERS_FILE"
sudo chmod 0440 "$SUDOERS_FILE"
rm "$TEMP_FILE"

echo "✓ Passwordless sudo configured for vLLM service management!"
echo ""
echo "You can now run vLLM service commands without password prompts:"
echo "  - make vllm-service"
echo "  - make vllm-start"
echo "  - make vllm-stop"
echo "  - make vllm-restart"
echo "  - make vllm-status"
echo "  - make vllm-logs"
echo ""
echo "To remove this configuration:"
echo "  sudo rm $SUDOERS_FILE"
