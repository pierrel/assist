# Include deployment configuration (if exists)
-include .deploy.env

# Default values
DEPLOY_HOST ?= assist-prod
DEPLOY_PATH ?= /opt/assist
SERVICE_NAME ?= assist-web
PYTHON ?= python3.14

# Function to run commands with development environment
define with-dev-env
	@if [ -f .dev.env ]; then \
		echo "→ Loading development environment from .dev.env"; \
		export $$(grep -v '^#' .dev.env | grep -v '^$$' | xargs) && $(1); \
	else \
		echo "⚠  Warning: .dev.env not found"; \
		$(1); \
	fi
endef

# Function to run commands with production environment
define with-prod-env
	@if [ -f .deploy.env ]; then \
		echo "→ Loading production environment from .deploy.env"; \
		export $$(grep -v '^#' .deploy.env | grep -v '^$$' | xargs) && $(1); \
	else \
		echo "⚠  Warning: .deploy.env not found"; \
		$(1); \
	fi
endef

.PHONY: eval test web smoke deploy deploy-code deploy-sandbox-build deploy-service deploy-install restart status logs setup-sudo help sandbox-build egress-proxy-build sandbox-smoke sandbox-shell pull-eval-history vacuum-now

eval:
	$(call with-dev-env,./scripts/run-evals.sh)

test:
	$(call with-dev-env,.venv/bin/pytest --junit-xml=tests/history/results-$$(date +%Y%m%d-%H%M).xml tests)

web: sandbox-build
	$(call with-dev-env,.venv/bin/python -m manage.web)

smoke:
	$(call with-dev-env,./scripts/smoke_test.sh)

pull-eval-history:
	@echo "→ Pulling eval history from $(DEPLOY_HOST):$(DEPLOY_PATH)/edd/history/ ..."
	@mkdir -p edd/history
	@rsync -avz $(DEPLOY_HOST):$(DEPLOY_PATH)/edd/history/ edd/history/
	@echo "✓ Eval history synced"

sandbox-build: egress-proxy-build
	docker build -t assist-sandbox -f dockerfiles/Dockerfile.sandbox .

# Egress allowlist proxy.  Tiny image (python:3-alpine + ~150 LOC of
# stdlib).  Used by SandboxManager._ensure_egress_proxy_running to
# gate every byte leaving the sandbox.  See
# dockerfiles/egress-proxy.py and docs/2026-05-08-sandbox-network-allowlist.org.
egress-proxy-build:
	docker build -t assist-egress-proxy -f dockerfiles/Dockerfile.egress-proxy .

# Build-time smoke.  Three layers, fail-on-first-regression:
#   - test-sandbox-shim.sh: 18 push-bypass variants + privilege-drop checks
#   - test-sandbox-egress.sh: positive (pip install via proxy) +
#     negative (off-allowlist host, direct-IP, raw TCP) probes
#   - test_sandbox_egress_integration.py: e2e through SandboxManager.
#     get_sandbox_backend → DockerSandboxBackend.execute("curl ..."),
#     the same call path the agent's tool hits.  Requires Docker;
#     fails loudly if missing (no skip — too important).
#
# `deploy-sandbox-build` only runs the two shell harnesses on the
# remote (no venv there); use `make sandbox-smoke` locally / in CI
# for the full three-layer coverage.
sandbox-smoke: sandbox-build
	bash dockerfiles/test-sandbox-shim.sh
	bash dockerfiles/test-sandbox-egress.sh
	.venv/bin/pytest tests/test_sandbox_egress_integration.py -v

sandbox-shell:
	docker run --rm -it assist-sandbox bash

# === Deployment Targets ===

deploy: deploy-code deploy-sandbox-build deploy-install deploy-service restart
	@echo "✓ Deployment complete!"
	@echo "Check status with: make status"
	@echo "View logs with: make logs"

deploy-code:
	@echo "→ Deploying code to $(DEPLOY_HOST):$(DEPLOY_PATH)..."
	@rsync -avz --delete \
		--filter=':- .gitignore' \
		--exclude '.git' \
		./ $(DEPLOY_HOST):$(DEPLOY_PATH)/
	@echo "✓ Code deployed"

deploy-sandbox-build:
	@echo "→ Building egress-proxy image on $(DEPLOY_HOST)..."
	@ssh $(DEPLOY_HOST) 'cd $(DEPLOY_PATH) && docker build -t assist-egress-proxy -f dockerfiles/Dockerfile.egress-proxy .'
	@echo "→ Building sandbox image on $(DEPLOY_HOST)..."
	@ssh $(DEPLOY_HOST) 'cd $(DEPLOY_PATH) && docker build -t assist-sandbox -f dockerfiles/Dockerfile.sandbox .'
	@echo "→ Running sandbox-smoke on $(DEPLOY_HOST) (push-refusal regression gate)..."
	@ssh $(DEPLOY_HOST) 'cd $(DEPLOY_PATH) && bash dockerfiles/test-sandbox-shim.sh'
	@echo "→ Running egress-smoke on $(DEPLOY_HOST) (allowlist regression gate)..."
	@ssh $(DEPLOY_HOST) 'cd $(DEPLOY_PATH) && bash dockerfiles/test-sandbox-egress.sh'
	@echo "✓ Sandbox + egress-proxy images built and smoked"

# Migrate pre-non-root-sandbox thread workspaces to the deploy
# user's ownership.  Idempotent.  Required after the first deploy
# of the non-root sandbox layer (docs/2026-05-08-...) — without it,
# legacy threads with root-owned files fail
# SandboxManager.get_sandbox_backend's "uid != 0" check on first
# turn.  Also wired into install-service.sh so it runs on every
# install; this target is for ad-hoc re-application.
deploy-migrate-workspaces:
	@echo "→ Migrating thread workspaces on $(DEPLOY_HOST) to deploy-user ownership..."
	@ssh $(DEPLOY_HOST) 'sudo chown -R $$USER:$$USER $(ASSIST_THREADS_DIR)'
	@echo "✓ Workspace ownership migrated"

deploy-service:
	@echo "→ Installing systemd service..."
	@ssh $(DEPLOY_HOST) \
		DEPLOY_PATH=$(DEPLOY_PATH) \
		SERVICE_NAME=$(SERVICE_NAME) \
		ASSIST_THREADS_DIR=$(ASSIST_THREADS_DIR) \
		ASSIST_PORT='$(ASSIST_PORT)' \
		ASSIST_MODEL_URL='$(ASSIST_MODEL_URL)' \
		ASSIST_DOMAINS='$(ASSIST_DOMAINS)' \
		'bash -s' < scripts/install-service.sh
	@echo "✓ Service installed"

deploy-install:
	@echo "→ Installing dependencies on remote server..."
	@ssh $(DEPLOY_HOST) 'cd $(DEPLOY_PATH) && \
		$(PYTHON) -m venv .venv && \
		.venv/bin/pip install --upgrade pip && \
		.venv/bin/pip install -e . && \
		.venv/bin/pip install -r requirements.txt'
	@echo "✓ Dependencies installed"

restart:
	@echo "→ Restarting $(SERVICE_NAME) service..."
	@ssh $(DEPLOY_HOST) 'sudo systemctl restart $(SERVICE_NAME)'
	@sleep 2
	@echo "→ Following service logs (Press Ctrl+C to exit)..."
	@ssh $(DEPLOY_HOST) 'sudo journalctl -u $(SERVICE_NAME) -f'

status:
	@ssh $(DEPLOY_HOST) 'sudo systemctl status $(SERVICE_NAME) --no-pager'

logs:
	@echo "→ Tailing logs from $(SERVICE_NAME)..."
	@echo "  (Press Ctrl+C to exit)"
	@ssh $(DEPLOY_HOST) 'sudo journalctl -u $(SERVICE_NAME) -f'

setup-sudo:
	@echo "→ Setting up passwordless sudo on $(DEPLOY_HOST)..."
	@echo "  This requires entering your password ONCE"
	@ssh -t $(DEPLOY_HOST) \
		SERVICE_NAME=$(SERVICE_NAME) \
		DEPLOY_PATH=$(DEPLOY_PATH) \
		ASSIST_THREADS_DIR=$(ASSIST_THREADS_DIR) \
		'bash -s' < scripts/setup-passwordless-sudo.sh
	@echo ""
	@echo "✓ Setup complete! You can now deploy from Emacs without password prompts."

# Run the threads.db VACUUM script on the deploy host — the same
# script the weekly user-cron entry invokes.  Stops assist-web for
# the duration; expect minutes on a small DB, hours on a >100 GB
# one.  Uses the existing passwordless sudo entries for systemctl
# stop/start assist-web.
#
# Paths come from .deploy.env (ASSIST_THREADS_DIR, SERVICE_NAME) so
# the committed script holds no host-specific defaults.
vacuum-now:
	@echo "→ Triggering vacuum-prod-db.sh on $(DEPLOY_HOST) (synchronous)..."
	@ssh $(DEPLOY_HOST) \
		ASSIST_THREADS_DIR=$(ASSIST_THREADS_DIR) \
		SERVICE_NAME=$(SERVICE_NAME) \
		DEPLOY_PATH=$(DEPLOY_PATH) \
		MIN_THREADS=$(or $(MIN_THREADS),100) \
		'$(DEPLOY_PATH)/scripts/vacuum-prod-db.sh'

help:
	@echo "Assist Commands:"
	@echo ""
	@echo "Development:"
	@echo "  make web            - Run web server locally (uses .dev.env)"
	@echo "  make test           - Run tests"
	@echo "  make eval           - Run evals"
	@echo "  make smoke          - Run smoke test against running server"
	@echo "  make sandbox-build  - Build Docker sandbox image"
	@echo "  make sandbox-shell  - Run interactive sandbox shell"
	@echo "  make pull-eval-history - Pull eval results from deploy server"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy         - Full deployment (code + service + restart)"
	@echo "  make deploy-code    - Deploy code only (no restart)"
	@echo "  make deploy-service - Install/update systemd service"
	@echo "  make install-prod   - Install dependencies on remote"
	@echo "  make restart        - Restart the service"
	@echo "  make status         - Check service status"
	@echo "  make logs           - View service logs (live tail)"
	@echo "  make setup-sudo     - Setup passwordless sudo (optional)"
	@echo "  make vacuum-now     - Run threads.db VACUUM now (stops assist-web ~10–30 min)"
	@echo ""
	@echo "Configuration:"
	@echo "  Development: Copy .dev.env.example to .dev.env"
	@echo "  Deployment:  Copy .deploy.env.example to .deploy.env"
	@echo "  SSH:         Configure host alias in ~/.ssh/config"
	@echo ""
	@echo "Note: All deployment commands work from Emacs."
	@echo "      Run 'make setup-sudo' once for passwordless operation."
