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
HOST="${HOST:-127.0.0.1}"
CORS_ORIGINS="${CORS_ORIGINS:-localhost}"
IMAGE_MIN_TOKENS="${IMAGE_MIN_TOKENS:-1024}"
API_KEY_FILE="${LLAMA_API_KEY_FILE:-}"
THREADS="${THREADS:-8}"
FIT_TARGET_MIB="${FIT_TARGET_MIB:-1536}"
REASONING="${REASONING:-auto}"
REASONING_BUDGET="${REASONING_BUDGET:-}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
REASONING_PRESERVE="${REASONING_PRESERVE:-1}"

if [[ "$ENABLE_VISION" == "1" ]]; then
    CTX_SIZE="${CTX_SIZE:-32768}"
    KV_TYPE_K="${KV_TYPE_K:-q8_0}"
    KV_TYPE_V="${KV_TYPE_V:-q8_0}"
else
    CTX_SIZE="${CTX_SIZE:-131072}"
    KV_TYPE_K="${KV_TYPE_K:-q4_0}"
    KV_TYPE_V="${KV_TYPE_V:-q4_0}"
fi

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
if [[ "$REASONING" != "auto" && "$REASONING" != "on" && "$REASONING" != "off" ]]; then
    echo "ERROR: REASONING must be auto, on, or off." >&2
    exit 1
fi
if [[ -n "$REASONING_BUDGET" && ! "$REASONING_BUDGET" =~ ^-?[0-9]+$ ]]; then
    echo "ERROR: REASONING_BUDGET must be an integer." >&2
    exit 1
fi
if [[ -n "$REASONING_EFFORT" ]]; then
    case "$REASONING_EFFORT" in
        default|minimal|low|medium|high|xhigh|max) ;;
        *)
            echo "ERROR: REASONING_EFFORT must be default, minimal, low, medium, high, xhigh, or max." >&2
            exit 1
            ;;
    esac
fi
if [[ "$REASONING_PRESERVE" != "0" && "$REASONING_PRESERVE" != "1" ]]; then
    echo "ERROR: REASONING_PRESERVE must be 0 or 1." >&2
    exit 1
fi
if [[ -n "$API_KEY_FILE" && ! -r "$API_KEY_FILE" ]]; then
    echo "ERROR: LLAMA_API_KEY_FILE is not readable: $API_KEY_FILE" >&2
    exit 1
fi
if [[ "$HOST" != "127.0.0.1" && "$HOST" != "::1" && -z "$API_KEY_FILE" && -z "${LLAMA_API_KEY:-}" ]]; then
    echo "ERROR: a non-loopback listener requires LLAMA_API_KEY or LLAMA_API_KEY_FILE." >&2
    exit 1
fi

# Vision requires its own 32k tier because the projector consumes VRAM.  The
# deployed text profile sets ENABLE_VISION=0 to retain its qualified 131k
# context; an operator can enable the separate vision tier when needed.
if [[ "$ENABLE_VISION" == "1" && "$CTX_SIZE" -gt 32768 ]]; then
    echo "ERROR: vision is qualified only at 32768 tokens in this staged profile." >&2
    exit 1
fi

args=(
    --model "$MODEL_PATH"
    --host "$HOST" --port "$PORT"
    --cors-origins "$CORS_ORIGINS"
    --ctx-size "$CTX_SIZE"
    --cache-type-k "$KV_TYPE_K"
    --cache-type-v "$KV_TYPE_V"
    --n-gpu-layers all
    --parallel 1
    --flash-attn on
    --batch-size 2048
    --ubatch-size 256
    --fit on
    --fit-target "$FIT_TARGET_MIB"
    --threads "$THREADS"
    --jinja
    --reasoning "$REASONING"
)

if [[ "$REASONING_PRESERVE" == "1" ]]; then
    args+=(--reasoning-preserve)
else
    args+=(--no-reasoning-preserve)
fi

if [[ -n "$REASONING_BUDGET" ]]; then
    args+=(--reasoning-budget "$REASONING_BUDGET")
fi
if [[ -n "$REASONING_EFFORT" ]]; then
    args+=(--reasoning-effort "$REASONING_EFFORT")
fi

if [[ "$ENABLE_VISION" == "1" ]]; then
    args+=(--mmproj "$MMPROJ_PATH" --image-min-tokens "$IMAGE_MIN_TOKENS")
fi
if [[ -n "$API_KEY_FILE" ]]; then
    args+=(--api-key-file "$API_KEY_FILE")
fi

echo "Starting staged Qwen3.8 profile: host=$HOST:$PORT context=$CTX_SIZE KV=$KV_TYPE_K/$KV_TYPE_V vision=$ENABLE_VISION reasoning=$REASONING budget=${REASONING_BUDGET:-unlimited} effort=${REASONING_EFFORT:-default} preserve=$REASONING_PRESERVE"
exec "$LLAMA_BIN" "${args[@]}"
