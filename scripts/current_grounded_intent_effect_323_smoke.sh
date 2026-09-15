#!/usr/bin/env bash
# Fresh BF16 smoke for the grounded_intent_effect_323 capability.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SMOKE_TRAIN_BATCHES="${GROUNDED_323_SMOKE_TRAIN_BATCHES:-8}"
SMOKE_VAL_BATCHES="${GROUNDED_323_SMOKE_VAL_BATCHES:-1}"
SMOKE_BATCH_SIZE="${GROUNDED_323_SMOKE_BATCH_SIZE:-1}"
SMOKE_MEMORY_EVERY="${GROUNDED_323_SMOKE_MEMORY_EVERY:-1}"

for value_name in \
  SMOKE_TRAIN_BATCHES SMOKE_VAL_BATCHES SMOKE_BATCH_SIZE SMOKE_MEMORY_EVERY; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[grounded-intent-effect-323-smoke] ${value_name} must be positive, got ${value}" >&2
    exit 2
  fi
done

for argument in "$@"; do
  case "${argument}" in
    --resume|--resume=*)
      echo "[grounded-intent-effect-323-smoke] resume is disabled" >&2
      exit 2
      ;;
    --out-dir|--out-dir=*)
      echo "[grounded-intent-effect-323-smoke] set OUT_DIR in the environment" >&2
      exit 2
      ;;
  esac
done

export OUT_DIR="${OUT_DIR:-runs/v119_grounded_323_smoke_$(date +%Y%m%d_%H%M%S)}"
export GROUNDED_323_BATCH_SIZE="${SMOKE_BATCH_SIZE}"

printf '[grounded-intent-effect-323-smoke] out_dir=%s cuda_visible_devices=%s train_batches=%s val_batches=%s batch=%s memory_every=%s resume=off\n' \
  "${OUT_DIR}" "${CUDA_VISIBLE_DEVICES:-<unset>}" "${SMOKE_TRAIN_BATCHES}" \
  "${SMOKE_VAL_BATCHES}" "${SMOKE_BATCH_SIZE}" "${SMOKE_MEMORY_EVERY}"

exec bash "${SCRIPT_DIR}/current_grounded_intent_effect_323.sh" \
  "$@" \
  --epochs 1 \
  --max-train-batches "${SMOKE_TRAIN_BATCHES}" \
  --max-val-batches "${SMOKE_VAL_BATCHES}" \
  --log-every 1 \
  --eval-sampling-diagnostic-batches 1 \
  --eval-proposal-ablation-batches 1 \
  --eval-execution-ablation-batches 1 \
  --eval-representation-batches 1 \
  --memory-report-every "${SMOKE_MEMORY_EVERY}" \
  --memory-report-detail 1 \
  --memory-report-sync 1
