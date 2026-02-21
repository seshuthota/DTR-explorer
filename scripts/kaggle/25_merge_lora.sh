#!/usr/bin/env bash
set -euo pipefail

# Merge LoRA adapter into base model for standalone inference.
#
# Example:
#   ADAPTER_ID=CuriousDragon/dtr-tuned-1.2b-DTR-lora \
#   OUT_DIR=models/dtr-tuned-1.2b-DTR-merged \
#   bash scripts/kaggle/25_merge_lora.sh

ADAPTER_ID="${ADAPTER_ID:-CuriousDragon/dtr-tuned-1.2b-DTR-lora}"
BASE_MODEL="${BASE_MODEL:-}"
OUT_DIR="${OUT_DIR:-models/dtr-tuned-1.2b-DTR-merged}"

BASE_ARGS=()
if [[ -n "$BASE_MODEL" ]]; then
  BASE_ARGS=(--base-model "$BASE_MODEL")
fi

python -u training/merge_lora.py \
  --adapter-id "$ADAPTER_ID" \
  --output-dir "$OUT_DIR" \
  "${BASE_ARGS[@]}"
