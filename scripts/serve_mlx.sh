#!/usr/bin/env bash
# Serve a fused MLX model on loopback. Use an SSH tunnel from Kali.
set -euo pipefail

MODEL="${MODEL:-fused/qwen3-14b-pentest-4bit}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMP="${TEMP:-0.2}"

command -v mlx_lm.server >/dev/null 2>&1 || { echo "mlx_lm.server not found; activate the venv"; exit 1; }
[ -e "$MODEL" ] || { echo "model not found: $MODEL"; exit 1; }
[ "$HOST" = "0.0.0.0" ] && echo "WARNING: binding to all interfaces; this server has no auth." >&2

echo "Serving $MODEL on http://$HOST:$PORT/v1"
echo "From Kali: ssh -N -f -L $PORT:127.0.0.1:$PORT <user>@<mac-ip>"
echo "Tip: match the thinking mode used in training; do not blindly pass enable_thinking=false."

exec mlx_lm.server \
  --model "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --max-tokens "$MAX_TOKENS" \
  --temp "$TEMP" \
  --prefill-step-size 512 \
  --log-level INFO
