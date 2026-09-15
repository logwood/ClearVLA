#!/usr/bin/env bash
# Fresh production-graph V115 BF16 smoke.  A real precomputed T5 condition is
# required by the inherited formal launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SMOKE_TRAIN_BATCHES="${V115_SMOKE_TRAIN_BATCHES:-8}"
SMOKE_VAL_BATCHES="${V115_SMOKE_VAL_BATCHES:-1}"
SMOKE_BATCH_SIZE="${V115_SMOKE_BATCH_SIZE:-1}"
SMOKE_MEMORY_EVERY="${V115_SMOKE_MEMORY_EVERY:-1}"

for value_name in \
  SMOKE_TRAIN_BATCHES SMOKE_VAL_BATCHES SMOKE_BATCH_SIZE SMOKE_MEMORY_EVERY; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[v115-smoke] ${value_name} must be a positive integer, got ${value}" >&2
    exit 2
  fi
done

for argument in "$@"; do
  case "${argument}" in
    --resume|--resume=*)
      echo "[v115-smoke] resume is disabled; use fresh model state" >&2
      exit 2
      ;;
    --out-dir|--out-dir=*)
      echo "[v115-smoke] set OUT_DIR in the environment" >&2
      exit 2
      ;;
  esac
done

export OUT_DIR="${OUT_DIR:-runs/v115_smoke_$(date +%Y%m%d_%H%M%S)}"
export V115_BATCH_SIZE="${SMOKE_BATCH_SIZE}"

printf '[v115-smoke] out_dir=%s cuda_visible_devices=%s train_batches=%s val_batches=%s batch=%s memory_every=%s resume=off\n' \
  "${OUT_DIR}" \
  "${CUDA_VISIBLE_DEVICES:-<unset>}" \
  "${SMOKE_TRAIN_BATCHES}" \
  "${SMOKE_VAL_BATCHES}" \
  "${SMOKE_BATCH_SIZE}" \
  "${SMOKE_MEMORY_EVERY}"

exec bash "${SCRIPT_DIR}/current_v115_g_aligned_goal_phase_323.sh" \
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
