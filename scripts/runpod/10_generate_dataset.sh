#!/usr/bin/env bash
set -euo pipefail

# DTR-filtered SFT data generation wrapper.
# Override vars inline, e.g. QUESTIONS=300 SAMPLES_PER_Q=20 bash scripts/runpod/10_generate_dataset.sh

# shellcheck disable=SC1090
source "${VENV_DIR:-.venv-runpod}/bin/activate"

QUESTIONS="${QUESTIONS:-200}"
SAMPLES_PER_Q="${SAMPLES_PER_Q:-16}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-320}"
MIN_DTR="${MIN_DTR:-0.32}"
KEEP_PER_Q="${KEEP_PER_Q:-4}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_K="${TOP_K:-50}"
OUTPUT_JSONL="${OUTPUT_JSONL:-data/dtr_filtered_sft.jsonl}"
OUTPUT_CSV="${OUTPUT_CSV:-data/dtr_candidates_debug.csv}"
STATE_PATH="${STATE_PATH:-${OUTPUT_JSONL}.state.json}"
LOG_EVERY="${LOG_EVERY:-1}"
DTR_MAX_TOKENS="${DTR_MAX_TOKENS:-0}"
RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-0}"
REQUIRE_CORRECT="${REQUIRE_CORRECT:-1}"
FALLBACK_BEST_CORRECT="${FALLBACK_BEST_CORRECT:-1}"
REQUIRE_BOXED="${REQUIRE_BOXED:-0}"
EXCLUDE_TRUNCATED="${EXCLUDE_TRUNCATED:-0}"

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
if [[ "$REQUIRE_BOXED" == "1" ]]; then
  EXTRA_FLAGS+=(--require-boxed)
fi
if [[ "$EXCLUDE_TRUNCATED" == "1" ]]; then
  EXTRA_FLAGS+=(--exclude-truncated)
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
  --dtr-max-tokens "$DTR_MAX_TOKENS" \
  --log-every "$LOG_EVERY" \
  --state-path "$STATE_PATH" \
  --output "$OUTPUT_JSONL" \
  --candidates-out "$OUTPUT_CSV" \
  "${EXTRA_FLAGS[@]}"
