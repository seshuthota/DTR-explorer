#!/usr/bin/env bash
set -euo pipefail

# Restore partial generation artifacts from a Kaggle input dataset or archive.
#
# Expected files (either directly in SOURCE_DIR or inside an extracted archive):
# - dtr_filtered_sft.jsonl
# - dtr_candidates_debug.csv
# - dtr_filtered_sft.jsonl.state.json
#
# Examples:
#   SOURCE_DIR=/kaggle/input/dtr-partial-checkpoint bash scripts/kaggle/05_restore_data.sh
#   SOURCE_ARCHIVE=/kaggle/input/dtr-partial-checkpoint/dtr_gen_checkpoint_partial.tar.gz \
#     bash scripts/kaggle/05_restore_data.sh

SOURCE_DIR="${SOURCE_DIR:-}"
SOURCE_ARCHIVE="${SOURCE_ARCHIVE:-}"
DEST_DIR="${DEST_DIR:-data}"

mkdir -p "$DEST_DIR"

if [[ -n "$SOURCE_ARCHIVE" ]]; then
  if [[ ! -f "$SOURCE_ARCHIVE" ]]; then
    echo "SOURCE_ARCHIVE not found: $SOURCE_ARCHIVE" >&2
    exit 1
  fi
  echo "Extracting archive: $SOURCE_ARCHIVE"
  tar -xzf "$SOURCE_ARCHIVE" -C "$DEST_DIR"
fi

if [[ -n "$SOURCE_DIR" ]]; then
  if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "SOURCE_DIR not found: $SOURCE_DIR" >&2
    exit 1
  fi
  for f in dtr_filtered_sft.jsonl dtr_candidates_debug.csv dtr_filtered_sft.jsonl.state.json; do
    if [[ -f "$SOURCE_DIR/$f" ]]; then
      cp -f "$SOURCE_DIR/$f" "$DEST_DIR/$f"
      echo "Copied: $SOURCE_DIR/$f -> $DEST_DIR/$f"
    fi
  done
fi

echo "Current data files:"
ls -lah "$DEST_DIR" || true
