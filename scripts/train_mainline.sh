#!/usr/bin/env bash
# The only launcher owned by the capability-named candidate mainline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG_PATH="${MAINLINE_CONFIG:-configs/mainline/object_intent_dynamics_323.json}"
OUTPUT_PATH="${OUT_DIR:-runs/clearvla_mainline}"
ARGS=(
  --config "${CONFIG_PATH}"
  --device "${MAINLINE_DEVICE:-auto}"
  --dtype "${MAINLINE_DTYPE:-bf16}"
  --batch-size "${MAINLINE_BATCH_SIZE:-8}"
  --num-workers "${MAINLINE_NUM_WORKERS:-4}"
  --output-dir "${OUTPUT_PATH}"
)
# A config is the source of truth for benchmark-specific data paths.  Add
# legacy CLI overrides only when the caller explicitly supplies them; passing
# Pen defaults unconditionally used to silently replace CALVIN/LIBERO paths.
if [[ -n "${DATA_ROOT:-}" ]]; then ARGS+=(--data-root "${DATA_ROOT}"); fi
if [[ -n "${CACHE_DIR:-}" ]]; then ARGS+=(--decoded-cache "${CACHE_DIR}"); fi
if [[ -n "${DINO_CACHE_DIR:-}" ]]; then ARGS+=(--dino-cache "${DINO_CACHE_DIR}"); fi
if [[ -n "${T5_CONDITION_PATH:-}" ]]; then ARGS+=(--t5-condition "${T5_CONDITION_PATH}"); fi

printf '[mainline] capability=object_intent_dynamics_323 batch=%s data=%s dino=%s t5=%s out=%s\n' \
  "${MAINLINE_BATCH_SIZE:-8}" \
  "${DATA_ROOT:-<config>}" \
  "${DINO_CACHE_DIR:-<config>}" \
  "${T5_CONDITION_PATH:-<config>}" \
  "${OUTPUT_PATH}"

exec python -u -m clearvla.mainline.train "${ARGS[@]}" "$@"
