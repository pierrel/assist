#!/bin/bash
# Manual setup for passwordless sudo
# Run this in your terminal (not from Emacs): bash scripts/setup-sudo-manual.sh

set -e

# Load config
if [ -f .deploy.env ]; then
    source .deploy.env
fi

DEPLOY_HOST="${DEPLOY_HOST:-assist-prod}"
SERVICE_NAME="${SERVICE_NAME:-assist-web}"
ASSIST_THREADS_DIR="${ASSIST_THREADS_DIR:-/var/lib/assist}"

echo "=== Passwordless Sudo Setup ==="
echo ""
echo "This will configure passwordless sudo on $DEPLOY_HOST"
echo "for deployment commands. You'll need to enter your"
echo "password ONCE."
echo ""
read -p "Press Enter to continue..."

# Generate sudoers content locally
SUDOERS_CONTENT="# Allow deployment commands without password
Defaults:$USER !requiretty
$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl daemon-reload
$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl enable ${SERVICE_NAME}
$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl disable ${SERVICE_NAME}
$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl start ${SERVICE_NAME}
$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl stop ${SERVICE_NAME}
$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart ${SERVICE_NAME}
$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl status ${SERVICE_NAME}
$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl reload ${SERVICE_NAME}
$USER ALL=(ALL) NOPASSWD: /usr/bin/journalctl -u ${SERVICE_NAME} *
$USER ALL=(ALL) NOPASSWD: /usr/bin/mkdir -p ${ASSIST_THREADS_DIR}
$USER ALL=(ALL) NOPASSWD: /usr/bin/mkdir -p ${ASSIST_THREADS_DIR}/*
$USER ALL=(ALL) NOPASSWD: /usr/bin/chown $USER\:$USER ${ASSIST_THREADS_DIR}
$USER ALL=(ALL) NOPASSWD: /usr/bin/chown $USER\:$USER ${ASSIST_THREADS_DIR}/*
$USER ALL=(ALL) NOPASSWD: /usr/bin/tee /etc/systemd/system/${SERVICE_NAME}.service"

# Create temp file on remote server
TEMP_FILE="assist-deploy.sudoers"
echo "$SUDOERS_CONTENT" | ssh $DEPLOY_HOST "cat > $TEMP_FILE"

# Install it with sudo (this will prompt for password)
echo ""
echo "→ Installing sudoers configuration..."
ssh -tt $DEPLOY_HOST "sudo cp $TEMP_FILE /etc/sudoers.d/assist-deploy && \
    sudo chmod 0440 /etc/sudoers.d/assist-deploy && \
    rm $TEMP_FILE && \
    echo '' && \
    echo '✓ Passwordless sudo configured!' && \
    echo '' && \
    echo 'You can now run deployment from Emacs without passwords:' && \
    echo '  - make deploy' && \
    echo '  - make restart' && \
    echo '  - make status'"

echo ""
echo "✓ Setup complete!"
