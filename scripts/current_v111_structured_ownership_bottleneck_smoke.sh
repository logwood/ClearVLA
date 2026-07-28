#!/usr/bin/env bash
# Production-shaped V111 smoke with the real BF16/backward/deploy path.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SMOKE_TRAIN_BATCHES="${V111_SMOKE_TRAIN_BATCHES:-20}"
SMOKE_VAL_BATCHES="${V111_SMOKE_VAL_BATCHES:-1}"
SMOKE_BATCH_SIZE="${V111_SMOKE_BATCH_SIZE:-4}"
SMOKE_MEMORY_EVERY="${V111_SMOKE_MEMORY_EVERY:-1}"

for value_name in \
  SMOKE_TRAIN_BATCHES SMOKE_VAL_BATCHES SMOKE_BATCH_SIZE SMOKE_MEMORY_EVERY; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[v111-smoke] ${value_name} must be a positive integer, got ${value}" >&2
    exit 2
  fi
done

for argument in "$@"; do
  case "${argument}" in
    --resume|--resume=*)
      echo "[v111-smoke] resume is disabled; use fresh model state" >&2
      exit 2
      ;;
    --out-dir|--out-dir=*)
      echo "[v111-smoke] set OUT_DIR in the environment" >&2
      exit 2
      ;;
  esac
done

export OUT_DIR="${OUT_DIR:-runs/v111_smoke_$(date +%Y%m%d_%H%M%S)}"
export V111_BATCH_SIZE="${SMOKE_BATCH_SIZE}"

printf '[v111-smoke] out_dir=%s cuda_visible_devices=%s train_batches=%s val_batches=%s batch=%s memory_every=%s resume=off\n' \
  "${OUT_DIR}" \
  "${CUDA_VISIBLE_DEVICES:-<unset>}" \
  "${SMOKE_TRAIN_BATCHES}" \
  "${SMOKE_VAL_BATCHES}" \
  "${SMOKE_BATCH_SIZE}" \
  "${SMOKE_MEMORY_EVERY}"

exec bash "${SCRIPT_DIR}/current_v111_structured_ownership_bottleneck.sh" \
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
