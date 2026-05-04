#!/usr/bin/env bash
# Install the assist-vacuum systemd service + timer on the deploy host.
# Mirrors scripts/install-service.sh's pattern.

set -euo pipefail

DEPLOY_PATH="${DEPLOY_PATH:?DEPLOY_PATH not set}"
SERVICE_NAME="${SERVICE_NAME:-assist-web}"
ASSIST_THREADS_DIR="${ASSIST_THREADS_DIR:?ASSIST_THREADS_DIR not set}"
TIMER_NAME="assist-vacuum"

echo "Installing $TIMER_NAME timer (deploy_path=$DEPLOY_PATH user=$USER)"

# Render and install the service unit.
sed \
    -e "s|{{USER}}|$USER|g" \
    -e "s|{{DEPLOY_PATH}}|$DEPLOY_PATH|g" \
    -e "s|{{ASSIST_THREADS_DIR}}|$ASSIST_THREADS_DIR|g" \
    -e "s|{{SERVICE_NAME}}|$SERVICE_NAME|g" \
    "$DEPLOY_PATH/scripts/assist-vacuum.service.template" \
    | sudo tee "/etc/systemd/system/$TIMER_NAME.service" > /dev/null

# Timer unit has no template params today, but keep the same pattern
# so future tweaks (alternate schedules, env-driven knobs) drop in.
sudo cp "$DEPLOY_PATH/scripts/assist-vacuum.timer.template" \
    "/etc/systemd/system/$TIMER_NAME.timer"

# Make sure the script is executable on the host.
chmod +x "$DEPLOY_PATH/scripts/vacuum-prod-db.sh"

sudo systemctl daemon-reload
sudo systemctl enable --now "$TIMER_NAME.timer"

echo "✓ Timer installed and enabled.  Next run:"
systemctl list-timers "$TIMER_NAME.timer" --no-pager | head -3
