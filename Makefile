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

.PHONY: eval test web smoke deploy deploy-code deploy-sandbox-build deploy-service deploy-install restart status logs setup-sudo help sandbox-build sandbox-shell pull-eval-history vllm-install vllm-download vllm-service vllm-setup-sudo vllm-start vllm-stop vllm-restart vllm-status vllm-health vllm-logs vllm-setup

eval:
	$(call with-dev-env,.venv/bin/pytest --junit-xml=edd/history/results-$$(date +%Y%m%d-%H%M).xml edd/eval)

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

sandbox-build:
	docker build -t assist-sandbox -f dockerfiles/Dockerfile.sandbox .

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
	@echo "→ Building sandbox image on $(DEPLOY_HOST)..."
	@ssh $(DEPLOY_HOST) 'cd $(DEPLOY_PATH) && docker build -t assist-sandbox -f dockerfiles/Dockerfile.sandbox .'
	@echo "✓ Sandbox image built"

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

# === vLLM Remote Service Targets ===
# Manage the vLLM inference server on the GPU server.
# Configure via VLLM_* vars in .dev.env (see .dev.env.example).

vllm-install:
	@echo "→ Installing vLLM on GPU server..."
	$(call with-dev-env, \
		ssh $$VLLM_USER@$$VLLM_HOST \
			"mkdir -p $$VLLM_PATH && \
			 cd $$VLLM_PATH && \
			 python3 -m venv .venv && \
			 .venv/bin/pip install --upgrade pip && \
			 .venv/bin/pip install vllm huggingface_hub")
	@echo "✓ vLLM installed"

vllm-download:
	@echo "→ Downloading model on GPU server..."
	$(call with-dev-env, \
		ssh $$VLLM_USER@$$VLLM_HOST \
			"$$VLLM_PATH/.venv/bin/huggingface-cli download $$VLLM_MODEL")
	@echo "✓ Model download complete"

vllm-service:
	@echo "→ Installing vLLM systemd service on GPU server..."
	$(call with-dev-env, \
		ssh $$VLLM_USER@$$VLLM_HOST \
			VLLM_PATH=$$VLLM_PATH \
			VLLM_MODEL=$$VLLM_MODEL \
			VLLM_PORT=$$VLLM_PORT \
			VLLM_MAX_MODEL_LEN=$$VLLM_MAX_MODEL_LEN \
			VLLM_GPU_MEM_UTIL=$$VLLM_GPU_MEM_UTIL \
			VLLM_SERVICE_NAME=$$VLLM_SERVICE_NAME \
			'bash -s' < scripts/install-vllm-service.sh)
	@echo "✓ vLLM service installed"

vllm-start:
	@echo "→ Starting vLLM service..."
	$(call with-dev-env, \
		ssh $$VLLM_USER@$$VLLM_HOST "sudo systemctl start $$VLLM_SERVICE_NAME")
	@echo "✓ vLLM service started (model load may take 60-120s; use 'make vllm-health' to check)"

vllm-stop:
	@echo "→ Stopping vLLM service..."
	$(call with-dev-env, \
		ssh $$VLLM_USER@$$VLLM_HOST "sudo systemctl stop $$VLLM_SERVICE_NAME")
	@echo "✓ vLLM service stopped"

vllm-restart:
	@echo "→ Restarting vLLM service..."
	$(call with-dev-env, \
		ssh $$VLLM_USER@$$VLLM_HOST "sudo systemctl restart $$VLLM_SERVICE_NAME")
	@echo "✓ vLLM service restarted"

vllm-status:
	$(call with-dev-env, \
		ssh $$VLLM_USER@$$VLLM_HOST \
			"sudo systemctl status $$VLLM_SERVICE_NAME --no-pager")

vllm-health:
	@echo "→ Checking vLLM health..."
	$(call with-dev-env, \
		curl -sf http://$$VLLM_HOST:$$VLLM_PORT/health \
			&& echo "✓ vLLM is healthy" \
			|| echo "✗ vLLM not responding (may still be loading)")

vllm-logs:
	@echo "→ Tailing vLLM logs (Ctrl+C to exit)..."
	$(call with-dev-env, \
		ssh $$VLLM_USER@$$VLLM_HOST \
			"sudo journalctl -u $$VLLM_SERVICE_NAME -f")

vllm-setup-sudo:
	@echo "→ Setting up passwordless sudo on $(VLLM_USER)@$(VLLM_HOST)..."
	$(call with-dev-env, \
		ssh $$VLLM_USER@$$VLLM_HOST \
			VLLM_SERVICE_NAME=$$VLLM_SERVICE_NAME \
			'bash -s' < scripts/setup-vllm-sudo.sh)
	@echo "✓ Passwordless sudo configured for vLLM service management"

vllm-setup: vllm-install vllm-download vllm-service
	@echo "✓ vLLM setup complete. Run 'make vllm-start' to begin serving."

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
	@echo "  make deploy-install - Install dependencies on remote"
	@echo "  make restart        - Restart the service"
	@echo "  make status         - Check service status"
	@echo "  make logs           - View service logs (live tail)"
	@echo "  make setup-sudo     - Setup passwordless sudo (optional)"
	@echo ""
	@echo "vLLM Remote Service (GPU server):"
	@echo "  make vllm-setup     - Full setup: install vLLM, download model, install service"
	@echo "  make vllm-setup-sudo - Configure passwordless sudo on GPU server (run once)"
	@echo "  make vllm-install   - Install vLLM in venv on GPU server"
	@echo "  make vllm-download  - Download model weights on GPU server"
	@echo "  make vllm-service   - Install/update systemd service on GPU server"
	@echo "  make vllm-start     - Start vLLM service"
	@echo "  make vllm-stop      - Stop vLLM service"
	@echo "  make vllm-restart   - Restart vLLM service"
	@echo "  make vllm-status    - Show systemd service status"
	@echo "  make vllm-health    - HTTP health check against vLLM endpoint"
	@echo "  make vllm-logs      - Tail vLLM logs (live)"
	@echo ""
	@echo "Configuration:"
	@echo "  Development: Copy .dev.env.example to .dev.env"
	@echo "  Deployment:  Copy .deploy.env.example to .deploy.env"
	@echo "  SSH:         Configure host alias in ~/.ssh/config"
	@echo ""
	@echo "Note: All deployment commands work from Emacs."
	@echo "      Run 'make setup-sudo' once for passwordless operation."
