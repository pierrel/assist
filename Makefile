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

.PHONY: eval test web deploy deploy-code deploy-service install-prod restart status logs setup-sudo help

eval:
	$(call with-dev-env,.venv/bin/pytest --junit-xml=edd/history/results-$$(date +%Y%m%d-%H%M).xml edd/eval)

test:
	$(call with-dev-env,.venv/bin/pytest --junit-xml=tests/history/results-$$(date +%Y%m%d-%H%M).xml tests)

web:
	$(call with-dev-env,.venv/bin/python -m manage.web)

# === Deployment Targets ===

deploy: deploy-code deploy-service restart
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

deploy-service:
	@echo "→ Installing systemd service..."
	@ssh $(DEPLOY_HOST) \
		DEPLOY_PATH=$(DEPLOY_PATH) \
		SERVICE_NAME=$(SERVICE_NAME) \
		ASSIST_THREADS_DIR=$(ASSIST_THREADS_DIR) \
		ASSIST_PORT='$(ASSIST_PORT)' \
		ASSIST_MODEL_URL='$(ASSIST_MODEL_URL)' \
		ASSIST_MODEL_NAME='$(ASSIST_MODEL_NAME)' \
		ASSIST_API_KEY='$(ASSIST_API_KEY)' \
		ASSIST_CONTEXT_LEN='$(ASSIST_CONTEXT_LEN)' \
		ASSIST_TEST_URL_PATH='$(ASSIST_TEST_URL_PATH)' \
		ASSIST_DOMAIN='$(ASSIST_DOMAIN)' \
		'bash -s' < scripts/install-service.sh
	@echo "✓ Service installed"

install-prod:
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
		'bash -s' < scripts/setup-passwordless-sudo.sh
	@echo ""
	@echo "✓ Setup complete! You can now deploy from Emacs without password prompts."

help:
	@echo "Assist Commands:"
	@echo ""
	@echo "Development:"
	@echo "  make web            - Run web server locally (uses .dev.env)"
	@echo "  make test           - Run tests"
	@echo "  make eval           - Run evals"
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
	@echo ""
	@echo "Configuration:"
	@echo "  Development: Copy .dev.env.example to .dev.env"
	@echo "  Deployment:  Copy .deploy.env.example to .deploy.env"
	@echo "  SSH:         Configure host alias in ~/.ssh/config"
	@echo ""
	@echo "Note: All deployment commands work from Emacs."
	@echo "      Run 'make setup-sudo' once for passwordless operation."
