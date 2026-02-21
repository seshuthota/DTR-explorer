#!/usr/bin/env bash
set -euo pipefail

# Kaggle bootstrap.
# Usage:
#   bash scripts/kaggle/00_setup.sh

PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_QUIET="${PIP_QUIET:-1}"

if [[ "$PIP_QUIET" == "1" ]]; then
  PIP_FLAGS=(-q)
else
  PIP_FLAGS=()
fi

"$PYTHON_BIN" -m pip install "${PIP_FLAGS[@]}" --upgrade pip wheel setuptools
pip install "${PIP_FLAGS[@]}" -r requirements-runpod.txt

mkdir -p data outputs models

echo "Kaggle environment ready."
