#!/usr/bin/env bash
set -euo pipefail

# SFT LoRA training wrapper.
# Override vars inline, e.g. BASE_MODEL=... OUT_DIR=... bash scripts/runpod/20_train_sft.sh

# shellcheck disable=SC1090
source "${VENV_DIR:-.venv-runpod}/bin/activate"

BASE_MODEL="${BASE_MODEL:-DavidAU/LFM2.5-1.2B-Thinking-Claude-4.6-Opus-Heretic-Uncensored-DISTILL}"
DATASET_PATH="${DATASET_PATH:-data/dtr_filtered_sft.jsonl}"
OUT_DIR="${OUT_DIR:-models/dtr-tuned-1.2b-v1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-2e-5}"
MICRO_BATCH="${MICRO_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
SAVE_STEPS="${SAVE_STEPS:-100}"
LOG_STEPS="${LOG_STEPS:-10}"

python -u training/train_sft_lora.py \
  --base-model "$BASE_MODEL" \
  --dataset-path "$DATASET_PATH" \
  --output-dir "$OUT_DIR" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --epochs "$EPOCHS" \
  --learning-rate "$LR" \
  --micro-batch-size "$MICRO_BATCH" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  --save-steps "$SAVE_STEPS" \
  --logging-steps "$LOG_STEPS"
