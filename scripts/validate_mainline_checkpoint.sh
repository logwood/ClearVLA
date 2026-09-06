#!/usr/bin/env bash
# Read-only existing-checkpoint validation for the capability-named mainline.
# Keep this launcher LF-only: it is executed directly on the Linux benchmark host.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${CHECKPOINT:-}" ]]; then
  printf 'CHECKPOINT must name one clearvla-mainline-checkpoint-v4 file\n' >&2
  exit 2
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ARGS=(
  --config "${MAINLINE_CONFIG:-configs/mainline/object_intent_dynamics_323.json}"
  --device "${MAINLINE_DEVICE:-auto}"
  --dtype "${MAINLINE_DTYPE:-bf16}"
  --batch-size "${MAINLINE_BATCH_SIZE:-8}"
  --num-workers "${MAINLINE_NUM_WORKERS:-4}"
  --output-dir "${OUT_DIR:-runs/clearvla_mainline_validation}"
  --validate-checkpoint "${CHECKPOINT}"
)
# Match training: the selected config owns data/cache/language paths unless
# the caller explicitly relocates one. Never turn another outlet's validation
# into a Pen read through an implicit shell default.
if [[ -n "${DATA_ROOT:-}" ]]; then ARGS+=(--data-root "${DATA_ROOT}"); fi
if [[ -n "${CACHE_DIR:-}" ]]; then ARGS+=(--decoded-cache "${CACHE_DIR}"); fi
if [[ -n "${DINO_CACHE_DIR:-}" ]]; then ARGS+=(--dino-cache "${DINO_CACHE_DIR}"); fi
if [[ -n "${T5_CONDITION_PATH:-}" ]]; then ARGS+=(--t5-condition "${T5_CONDITION_PATH}"); fi

printf '[mainline-validation-only] checkpoint=%s batch=%s out=%s\n' \
  "${CHECKPOINT}" \
  "${MAINLINE_BATCH_SIZE:-8}" \
  "${OUT_DIR:-runs/clearvla_mainline_validation}"

exec python -u -m clearvla.mainline.train "${ARGS[@]}" "$@"
