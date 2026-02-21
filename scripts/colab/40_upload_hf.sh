#!/usr/bin/env bash
set -euo pipefail

# Upload trained artifacts to Hugging Face Hub from Colab.
# Example:
#   HF_TOKEN=hf_xxx HF_REPO_ID=username/dtr-tuned-1.2b-colab-lora \
#   bash scripts/colab/40_upload_hf.sh

LOCAL_DIR="${LOCAL_DIR:-models/dtr-tuned-1.2b-colab}"
HF_REPO_ID="${HF_REPO_ID:-}"
HF_REPO_TYPE="${HF_REPO_TYPE:-model}"
HF_PRIVATE="${HF_PRIVATE:-0}"
HF_REVISION="${HF_REVISION:-main}"
HF_COMMIT_MESSAGE="${HF_COMMIT_MESSAGE:-Upload trained artifact from Colab}"
HF_ALLOW_PATTERNS="${HF_ALLOW_PATTERNS:-}"
HF_IGNORE_PATTERNS="${HF_IGNORE_PATTERNS:-}"

if [[ -z "$HF_REPO_ID" ]]; then
  echo "HF_REPO_ID is required (example: username/dtr-tuned-1.2b-colab-lora)" >&2
  exit 1
fi

PRIVATE_FLAG=()
if [[ "$HF_PRIVATE" == "1" ]]; then
  PRIVATE_FLAG=(--private)
fi

python -u training/upload_to_hf.py \
  --local-dir "$LOCAL_DIR" \
  --repo-id "$HF_REPO_ID" \
  --repo-type "$HF_REPO_TYPE" \
  --revision "$HF_REVISION" \
  --commit-message "$HF_COMMIT_MESSAGE" \
  --allow-patterns "$HF_ALLOW_PATTERNS" \
  --ignore-patterns "$HF_IGNORE_PATTERNS" \
  "${PRIVATE_FLAG[@]}"
