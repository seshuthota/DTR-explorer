#!/usr/bin/env bash
set -euo pipefail

# RunPod bootstrap for this repo.
# Usage:
#   bash scripts/runpod/00_setup.sh

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv-runpod}"

$PYTHON_BIN -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools
pip install -r requirements-runpod.txt

mkdir -p data outputs models

echo "RunPod environment ready."
echo "Activate with: source $VENV_DIR/bin/activate"
