#!/bin/bash
# Install systemd service for vLLM inference server
# This script is executed on the remote GPU server via SSH
# Environment variables are passed from the Makefile

set -e

VLLM_PATH="${VLLM_PATH:-/home/pierre/src/serve/vllm}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-14B-Instruct-AWQ}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.92}"
VLLM_SERVICE_NAME="${VLLM_SERVICE_NAME:-vllm-serve}"

echo "Installing service: $VLLM_SERVICE_NAME"
echo "vLLM path: $VLLM_PATH"
echo "Model: $VLLM_MODEL"

sudo tee "/etc/systemd/system/$VLLM_SERVICE_NAME.service" > /dev/null <<EOF
[Unit]
Description=vLLM Inference Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$VLLM_PATH
Environment="PATH=$VLLM_PATH/.venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$VLLM_PATH/.venv/bin/vllm serve $VLLM_MODEL \\
    --host 0.0.0.0 \\
    --port $VLLM_PORT \\
    --served-model-name $VLLM_MODEL \\
    --enable-auto-tool-choice \\
    --tool-call-parser hermes \\
    --max-model-len $VLLM_MAX_MODEL_LEN \\
    --dtype auto \\
    --quantization awq \\
    --gpu-memory-utilization $VLLM_GPU_MEM_UTIL
Restart=on-failure
RestartSec=15
TimeoutStartSec=300
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$VLLM_SERVICE_NAME

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "$VLLM_SERVICE_NAME"

echo "✓ Service installed successfully"
echo "Start with: sudo systemctl start $VLLM_SERVICE_NAME"
