#!/usr/bin/env bash
set -euo pipefail

# Restore dataset artifacts from a Hugging Face Dataset repo into ./data.
#
# Example:
#   HF_TOKEN=hf_xxx HF_REPO_ID=username/dtr-generated-data-v1 \
#   bash scripts/colab/06_restore_data_from_hf.sh

HF_REPO_ID="${HF_REPO_ID:-}"
HF_REVISION="${HF_REVISION:-main}"
DEST_DIR="${DEST_DIR:-data}"
HF_ALLOW_PATTERNS="${HF_ALLOW_PATTERNS:-dtr_filtered_sft.jsonl,dtr_candidates_debug.csv,dtr_filtered_sft.jsonl.state.json}"
HF_IGNORE_PATTERNS="${HF_IGNORE_PATTERNS:-}"

if [[ -z "$HF_REPO_ID" ]]; then
  echo "HF_REPO_ID is required (example: username/dtr-generated-data-v1)" >&2
  exit 1
fi

python -u training/download_from_hf.py \
  --repo-id "$HF_REPO_ID" \
  --repo-type "dataset" \
  --revision "$HF_REVISION" \
  --local-dir "$DEST_DIR" \
  --allow-patterns "$HF_ALLOW_PATTERNS" \
  --ignore-patterns "$HF_IGNORE_PATTERNS"

echo "Restored files:"
ls -lah "$DEST_DIR"
