#!/usr/bin/env bash
set -euo pipefail

# Colab-friendly DTR-filtered SFT data generation wrapper.
# Conservative defaults for 15-16GB VRAM.
#
# Example resume run:
#   QUESTIONS=200 SAMPLES_PER_Q=16 RESUME=1 bash scripts/colab/10_generate_dataset.sh

QUESTIONS="${QUESTIONS:-200}"
SAMPLES_PER_Q="${SAMPLES_PER_Q:-16}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-224}"
MIN_DTR="${MIN_DTR:-0.32}"
KEEP_PER_Q="${KEEP_PER_Q:-4}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_K="${TOP_K:-50}"
OUTPUT_JSONL="${OUTPUT_JSONL:-data/dtr_filtered_sft.jsonl}"
OUTPUT_CSV="${OUTPUT_CSV:-data/dtr_candidates_debug.csv}"
STATE_PATH="${STATE_PATH:-${OUTPUT_JSONL}.state.json}"
LOG_EVERY="${LOG_EVERY:-2}"
RESUME="${RESUME:-1}"
OVERWRITE="${OVERWRITE:-0}"
REQUIRE_CORRECT="${REQUIRE_CORRECT:-1}"
FALLBACK_BEST_CORRECT="${FALLBACK_BEST_CORRECT:-1}"

EXTRA_FLAGS=()
if [[ "$RESUME" == "1" ]]; then
  EXTRA_FLAGS+=(--resume)
fi
if [[ "$OVERWRITE" == "1" ]]; then
  EXTRA_FLAGS+=(--overwrite)
fi
if [[ "$REQUIRE_CORRECT" == "1" ]]; then
  EXTRA_FLAGS+=(--require-correct)
fi
if [[ "$FALLBACK_BEST_CORRECT" == "1" ]]; then
  EXTRA_FLAGS+=(--fallback-best-correct)
fi

python -u experiments/generate_dtr_dataset.py \
  --questions "$QUESTIONS" \
  --samples-per-q "$SAMPLES_PER_Q" \
  --sample-batch-size "$SAMPLE_BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-k "$TOP_K" \
  --min-dtr "$MIN_DTR" \
  --keep-per-q "$KEEP_PER_Q" \
  --log-every "$LOG_EVERY" \
  --state-path "$STATE_PATH" \
  --output "$OUTPUT_JSONL" \
  --candidates-out "$OUTPUT_CSV" \
  "${EXTRA_FLAGS[@]}"
