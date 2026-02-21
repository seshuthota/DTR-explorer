#!/usr/bin/env bash
set -euo pipefail

# Compare base model vs merged model with identical Think@n settings.
#
# Example:
#   MERGED_MODEL_ID=models/dtr-tuned-1.2b-DTR-merged \
#   bash scripts/kaggle/31_compare_base_vs_merged.sh

mkdir -p outputs

BASE_MODEL_ID="${BASE_MODEL_ID:-DavidAU/LFM2.5-1.2B-Thinking-Claude-4.6-Opus-Heretic-Uncensored-DISTILL}"
MERGED_MODEL_ID="${MERGED_MODEL_ID:-models/dtr-tuned-1.2b-DTR-merged}"

QUESTIONS="${QUESTIONS:-12}"
N="${N:-16}"
PREFIX_TOKENS="${PREFIX_TOKENS:-50}"
KEEP_RATIO="${KEEP_RATIO:-0.5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
TEMPERATURE="${TEMPERATURE:-0.5}"
TOP_K="${TOP_K:-20}"
SEED="${SEED:-42}"

echo "Running BASE evaluation..."
DTR_MODEL_ID="$BASE_MODEL_ID" python -u experiments/think_n.py \
  --questions "$QUESTIONS" \
  --n "$N" \
  --prefix-tokens "$PREFIX_TOKENS" \
  --keep-ratio "$KEEP_RATIO" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-k "$TOP_K" \
  --seed "$SEED" \
  --csv-out outputs/think_n_base.csv | tee outputs/think_n_base.log

echo "Running MERGED evaluation..."
DTR_MODEL_ID="$MERGED_MODEL_ID" python -u experiments/think_n.py \
  --questions "$QUESTIONS" \
  --n "$N" \
  --prefix-tokens "$PREFIX_TOKENS" \
  --keep-ratio "$KEEP_RATIO" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-k "$TOP_K" \
  --seed "$SEED" \
  --csv-out outputs/think_n_merged.csv | tee outputs/think_n_merged.log

echo "Comparison logs:"
echo "  outputs/think_n_base.log"
echo "  outputs/think_n_merged.log"
