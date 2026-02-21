#!/usr/bin/env bash
set -euo pipefail

# DTR-filtered SFT data generation wrapper.
# Override vars inline, e.g. QUESTIONS=300 SAMPLES_PER_Q=20 bash scripts/runpod/10_generate_dataset.sh

# shellcheck disable=SC1090
source "${VENV_DIR:-.venv-runpod}/bin/activate"

QUESTIONS="${QUESTIONS:-200}"
SAMPLES_PER_Q="${SAMPLES_PER_Q:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-320}"
MIN_DTR="${MIN_DTR:-0.32}"
KEEP_PER_Q="${KEEP_PER_Q:-4}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_K="${TOP_K:-50}"
OUTPUT_JSONL="${OUTPUT_JSONL:-data/dtr_filtered_sft.jsonl}"
OUTPUT_CSV="${OUTPUT_CSV:-data/dtr_candidates_debug.csv}"

python -u experiments/generate_dtr_dataset.py \
  --questions "$QUESTIONS" \
  --samples-per-q "$SAMPLES_PER_Q" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-k "$TOP_K" \
  --min-dtr "$MIN_DTR" \
  --keep-per-q "$KEEP_PER_Q" \
  --require-correct \
  --fallback-best-correct \
  --output "$OUTPUT_JSONL" \
  --candidates-out "$OUTPUT_CSV"
