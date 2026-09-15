#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUT_DIR="${OUT_DIR:-runs/v122_object_intent_dynamics_323_identity_innovation_smoke}"
export OBJECT_323_BATCH_SIZE="${OBJECT_323_BATCH_SIZE:-1}"

exec bash "${SCRIPT_DIR}/current_object_intent_dynamics_323.sh" \
  --epochs 1 \
  --max-train-batches "${SMOKE_TRAIN_BATCHES:-2}" \
  --max-val-batches "${SMOKE_VAL_BATCHES:-1}" \
  --log-every 1 \
  "$@"
