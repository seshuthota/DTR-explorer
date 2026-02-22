#!/usr/bin/env bash
set -euo pipefail

# Upload generation dataset artifacts to a Hugging Face Dataset repo.
#
# Example:
#   HF_TOKEN=hf_xxx HF_REPO_ID=username/dtr-generated-data-v1 \
#   bash scripts/colab/41_upload_dataset_hf.sh

LOCAL_DIR="${LOCAL_DIR:-data}"
HF_REPO_ID="${HF_REPO_ID:-}"
HF_PRIVATE="${HF_PRIVATE:-1}"
HF_REVISION="${HF_REVISION:-main}"
HF_COMMIT_MESSAGE="${HF_COMMIT_MESSAGE:-Upload DTR generation dataset checkpoint}"
HF_ALLOW_PATTERNS="${HF_ALLOW_PATTERNS:-dtr_filtered_sft.jsonl,dtr_candidates_debug.csv,dtr_filtered_sft.jsonl.state.json}"
HF_IGNORE_PATTERNS="${HF_IGNORE_PATTERNS:-}"

if [[ -z "$HF_REPO_ID" ]]; then
  echo "HF_REPO_ID is required (example: username/dtr-generated-data-v1)" >&2
  exit 1
fi

PRIVATE_FLAG=()
if [[ "$HF_PRIVATE" == "1" ]]; then
  PRIVATE_FLAG=(--private)
fi

python -u training/upload_to_hf.py \
  --local-dir "$LOCAL_DIR" \
  --repo-id "$HF_REPO_ID" \
  --repo-type "dataset" \
  --revision "$HF_REVISION" \
  --commit-message "$HF_COMMIT_MESSAGE" \
  --allow-patterns "$HF_ALLOW_PATTERNS" \
  --ignore-patterns "$HF_IGNORE_PATTERNS" \
  "${PRIVATE_FLAG[@]}"
