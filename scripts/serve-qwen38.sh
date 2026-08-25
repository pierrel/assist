#!/usr/bin/env bash
# Staged Qwen3.8 llama.cpp launch profile.  It is inert until an operator
# installs it as the llamacpp service override described in the runbook.
set -euo pipefail

LLAMA_ROOT="${LLAMA_ROOT:-$HOME/llama-cpp}"
LLAMA_BIN="${LLAMA_BIN:-$LLAMA_ROOT/llama.cpp/build/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:-$LLAMA_ROOT/models/Qwen3.8-27B-UD-Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf}"
MMPROJ_PATH="${MMPROJ_PATH:-$LLAMA_ROOT/models/Qwen3.8-27B-UD-Q4_K_XL/mmproj-F16.gguf}"
ENABLE_VISION="${ENABLE_VISION:-1}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
THREADS="${THREADS:-8}"
FIT_TARGET_MIB="${FIT_TARGET_MIB:-1536}"

if [[ "$ENABLE_VISION" == "1" ]]; then
    CTX_SIZE="${CTX_SIZE:-32768}"
else
    CTX_SIZE="${CTX_SIZE:-100000}"
fi
KV_TYPE_K="${KV_TYPE_K:-q8_0}"
KV_TYPE_V="${KV_TYPE_V:-q8_0}"

if [[ ! -x "$LLAMA_BIN" ]]; then
    echo "ERROR: llama-server is missing; complete the upgrade before cutover." >&2
    exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: Qwen3.8 model artifact is missing: $MODEL_PATH" >&2
    exit 1
fi
if [[ "$ENABLE_VISION" == "1" && ! -f "$MMPROJ_PATH" ]]; then
    echo "ERROR: Qwen3.8 multimodal projector is missing: $MMPROJ_PATH" >&2
    exit 1
fi
if [[ "$ENABLE_VISION" != "0" && "$ENABLE_VISION" != "1" ]]; then
    echo "ERROR: ENABLE_VISION must be 0 or 1." >&2
    exit 1
fi

# Vision is enabled for the initial Qwen3.8 service.  It starts at 32k because
# its projector needs its own VRAM and quality measurements.  Set
# ENABLE_VISION=0 for the text-only 100k/Q8 and 128k/Q4 qualification tiers.
if [[ "$ENABLE_VISION" == "1" && "$CTX_SIZE" -gt 32768 ]]; then
    echo "ERROR: vision is qualified only at 32768 tokens in this staged profile." >&2
    exit 1
fi

args=(
    --model "$MODEL_PATH"
    --host "$HOST" --port "$PORT"
    --ctx-size "$CTX_SIZE"
    --cache-type-k "$KV_TYPE_K"
    --cache-type-v "$KV_TYPE_V"
    --parallel 1
    --flash-attn on
    --batch-size 2048
    --ubatch-size 256
    --fit on
    --fit-target "$FIT_TARGET_MIB"
    --threads "$THREADS"
    --jinja
)

# Leave GPU layers unset so --fit can select the greatest full-offload fit with
# the requested margin.  The cutover gate rejects a profile that cannot still
# offload every layer; an explicit N_GPU_LAYERS override is deliberately absent.

if [[ "$ENABLE_VISION" == "1" ]]; then
    args+=(--mmproj "$MMPROJ_PATH")
fi

echo "Starting staged Qwen3.8 profile: context=$CTX_SIZE KV=$KV_TYPE_K/$KV_TYPE_V vision=$ENABLE_VISION"
exec "$LLAMA_BIN" "${args[@]}"
