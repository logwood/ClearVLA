#!/usr/bin/env bash
# The only launcher owned by the capability-named candidate mainline.
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
  --output-dir "${OUT_DIR:-runs/clearvla_mainline}"
  --data-root "${DATA_ROOT:-/data/liang.zhang/dataset/grab_pen_single/grab_pen_single}"
  --decoded-cache "${CACHE_DIR:-/data/senwang/data/cache_336}"
  --dino-cache "${DINO_CACHE_DIR:-/data/senwang/data/dinov2_cache_336}"
  --t5-condition "${T5_CONDITION_PATH:-/data/senwang/checkpoint/grasp_pen_embed.pt}"
)

printf '[mainline] capability=object_intent_dynamics_323 batch=%s data=%s dino=%s t5=%s out=%s\n' \
  "${MAINLINE_BATCH_SIZE:-8}" \
  "${DATA_ROOT:-/data/liang.zhang/dataset/grab_pen_single/grab_pen_single}" \
  "${DINO_CACHE_DIR:-/data/senwang/data/dinov2_cache_336}" \
  "${T5_CONDITION_PATH:-/data/senwang/checkpoint/grasp_pen_embed.pt}" \
  "${OUT_DIR:-runs/clearvla_mainline}"

exec python -u -m clearvla.mainline.train "${ARGS[@]}" "$@"
