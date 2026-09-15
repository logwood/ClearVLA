#!/usr/bin/env bash
# One real-batch cache0/cache1 VJP attribution; no optimizer or checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ARGS=(
  --config "${MAINLINE_CONFIG:-configs/mainline/object_intent_dynamics_323.json}"
  --device "${MAINLINE_DEVICE:-auto}"
  --dtype "${MAINLINE_DTYPE:-bf16}"
  --batch-size "${MAINLINE_BATCH_SIZE:-8}"
  --num-workers "${MAINLINE_NUM_WORKERS:-4}"
  --data-root "${DATA_ROOT:-/data/liang.zhang/dataset/grab_pen_single/grab_pen_single}"
  --decoded-cache "${CACHE_DIR:-/data/senwang/data/cache_336}"
  --dino-cache "${DINO_CACHE_DIR:-/data/senwang/data/dinov2_cache_336}"
  --t5-condition "${T5_CONDITION_PATH:-/data/senwang/checkpoint/grasp_pen_embed.pt}"
)

if [[ -n "${EXPECTED_COMMIT:-}" ]]; then
  ARGS+=(--expected-source-commit "${EXPECTED_COMMIT}")
fi
if [[ -n "${PROBE_OUTPUT:-}" ]]; then
  ARGS+=(--output "${PROBE_OUTPUT}")
fi

printf '[schema29-real-batch-probe] batch=%s device=%s output=%s\n' \
  "${MAINLINE_BATCH_SIZE:-8}" \
  "${MAINLINE_DEVICE:-auto}" \
  "${PROBE_OUTPUT:-stdout-only}"

exec python -u -m clearvla.tools.probe_schema29_real_batch "${ARGS[@]}" "$@"
