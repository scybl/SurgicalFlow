#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${PYTHON_BIN:-}" && -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
PYCACHE_DIR="${PYTHONPYCACHEPREFIX:-${TMPDIR:-/tmp}/surgical_workflow_prediction_pycache}"

PYTHONPYCACHEPREFIX="$PYCACHE_DIR" "$PYTHON_BIN" -m py_compile \
  model_backbone.py \
  model_out_head.py \
  taskA_data_loader.py \
  taskB_data_loader.py \
  train_backbone.py \
  train_taskA_out_head.py \
  train_taskB_out_head.py \
  test_backbone.py \
  test_taskA_out_head.py \
  test_taskB_out_head.py \
  general_compare_diagram.py \
  checkdata.py

echo "Code structure check passed."

if [[ ! -d data/cholec80 ]]; then
  echo "Dataset not found at data/cholec80. Follow README.md to link or pass --data_root."
fi
