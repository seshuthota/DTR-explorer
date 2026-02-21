#!/usr/bin/env bash
set -euo pipefail

# Quick post-train checks for Kaggle.

mkdir -p outputs

if [[ -n "${MODEL_ID:-}" ]]; then
  export DTR_MODEL_ID="$MODEL_ID"
elif [[ -n "${DTR_MODEL_ID:-}" ]]; then
  export DTR_MODEL_ID
fi

python -u experiments/think_n.py \
  --questions "${EVAL_QUESTIONS:-6}" \
  --n "${EVAL_N:-16}" \
  --prefix-tokens "${EVAL_PREFIX:-50}" \
  --keep-ratio "${EVAL_KEEP_RATIO:-0.5}" \
  --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-384}" \
  --csv-out "${EVAL_THINK_CSV:-outputs/think_n_posttrain_kaggle.csv}"

python -u experiments/length_vs_dtr_correlation.py \
  --questions "${CORR_QUESTIONS:-2}" \
  --samples-per-question "${CORR_SAMPLES_PER_Q:-30}" \
  --max-new-tokens "${CORR_MAX_NEW_TOKENS:-384}" \
  --csv-out "${EVAL_CORR_CSV:-outputs/len_vs_dtr_posttrain_kaggle.csv}" \
  --plot-out "${EVAL_CORR_PLOT:-outputs/len_vs_dtr_posttrain_kaggle.png}"
