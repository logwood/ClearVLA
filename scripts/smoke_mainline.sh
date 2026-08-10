#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MAINLINE_BATCH_SIZE="${MAINLINE_BATCH_SIZE:-1}"
export MAINLINE_NUM_WORKERS="${MAINLINE_NUM_WORKERS:-0}"
export OUT_DIR="${OUT_DIR:-runs/clearvla_mainline_smoke}"

exec bash "${SCRIPT_DIR}/train_mainline.sh" \
  --smoke \
  --max-train-batches "${SMOKE_TRAIN_BATCHES:-2}" \
  --max-val-batches "${SMOKE_VAL_BATCHES:-1}" \
  "$@"
