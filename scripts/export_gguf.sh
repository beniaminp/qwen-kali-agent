#!/usr/bin/env bash
# Fuse an MLX adapter, dequantize, convert to GGUF and quantize with llama.cpp.
# mlx_lm.fuse --export-gguf only supports Llama/Mistral/Mixtral, so we use
# llama.cpp's converter for Qwen3.
set -euo pipefail

MODEL="${MODEL:-mlx-community/Qwen3-14B-4bit}"
ADAPTER="${ADAPTER:-adapters/qwen3-14b-pentest}"
FUSED="${FUSED:-fused/qwen3-14b-pentest-bf16}"
LLAMA_CPP="${LLAMA_CPP:-$HOME/work/llama.cpp}"
OUT_DIR="${OUT_DIR:-models}"
QUANT="${QUANT:-Q4_K_M}"
NAME="${NAME:-qwen3-14b-pentest}"

command -v mlx_lm.fuse >/dev/null 2>&1 || { echo "mlx_lm.fuse not found; activate the venv"; exit 1; }
[ -d "$ADAPTER" ] || { echo "adapter not found: $ADAPTER"; exit 1; }
[ -f "$LLAMA_CPP/convert_hf_to_gguf.py" ] || { echo "llama.cpp converter not found at $LLAMA_CPP"; exit 1; }
[ -x "$LLAMA_CPP/build/bin/llama-quantize" ] || { echo "llama-quantize not built at $LLAMA_CPP/build/bin"; exit 1; }

mkdir -p "$OUT_DIR" "$(dirname "$FUSED")"

echo "==> Fusing $MODEL with $ADAPTER (dequantized bf16)"
mlx_lm.fuse \
  --model "$MODEL" \
  --adapter-path "$ADAPTER" \
  --save-path "$FUSED" \
  --dequantize

echo "==> Converting to GGUF f16"
python3 "$LLAMA_CPP/convert_hf_to_gguf.py" \
  "$FUSED" \
  --outtype f16 \
  --outfile "$OUT_DIR/${NAME}-f16.gguf"

echo "==> Quantizing to $QUANT"
"$LLAMA_CPP/build/bin/llama-quantize" \
  "$OUT_DIR/${NAME}-f16.gguf" \
  "$OUT_DIR/${NAME}-${QUANT}.gguf" "$QUANT"

echo "==> Checksums"
if command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "$OUT_DIR/${NAME}-${QUANT}.gguf"
else
  sha256sum "$OUT_DIR/${NAME}-${QUANT}.gguf"
fi

echo "==> Sizes"
du -h "$FUSED" "$OUT_DIR/${NAME}-${QUANT}.gguf" | tail -5
echo "Done: $OUT_DIR/${NAME}-${QUANT}.gguf"
