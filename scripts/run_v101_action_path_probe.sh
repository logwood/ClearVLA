#!/usr/bin/env bash
# Frozen-checkpoint paired causal probe for the V101 world -> policy -> action path.
# It stratifies validation-wide and gripper-event batches by episode, separates
# world anchor/xy ownership, and tests raw coordinates versus the exact
# high-frequency detail residual after reader output projection.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CHECKPOINT="${CHECKPOINT:-runs/v101_information_balanced_long_horizon/checkpoints/latest.pt}"
DATA_ROOT="${DATA_ROOT:-/data/liang.zhang/dataset/grab_pen_single/grab_pen_single}"
CACHE_DIR="${CACHE_DIR:-/home/sen.wang/workspace/robotics/clear/data/cache_336}"
DINO_CACHE_DIR="${DINO_CACHE_DIR:-/home/sen.wang/workspace/robotics/clear/data/dinov2_cache_336}"
DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-runs/diagnostics/v101_action_path}"
PROBE_BATCHES="${PROBE_BATCHES:-10}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BOOTSTRAP_REPS="${BOOTSTRAP_REPS:-2000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-101}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_JSON="${RESULT_JSON:-${DIAGNOSTICS_DIR}/probe_v3_${PROBE_BATCHES}b.json}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  printf 'checkpoint not found: %s\n' "${CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "$(dirname "${RESULT_JSON}")"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

printf '[v101-action-path-probe] checkpoint=%s batches=%s max_val_batches=%s batch_size=%s result=%s\n' \
  "${CHECKPOINT}" "${PROBE_BATCHES}" "${MAX_VAL_BATCHES}" "${BATCH_SIZE}" "${RESULT_JSON}"

"${PYTHON_BIN}" -m clearvla.cli.eval_v39_policy \
  --checkpoint "${CHECKPOINT}" \
  --data-root "${DATA_ROOT}" \
  --decoded-image-cache-dir "${CACHE_DIR}" \
  --condition-mode dinov2-cache \
  --dinov2-token-cache-dir "${DINO_CACHE_DIR}" \
  --cache-resize 336 336 \
  --cameras top wrist \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --device cuda \
  --dtype bf16 \
  --eval-inference-steps 5 \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  --action-path-intervention-batches "${PROBE_BATCHES}" \
  --action-path-bootstrap-reps "${BOOTSTRAP_REPS}" \
  --action-path-bootstrap-seed "${BOOTSTRAP_SEED}" \
  --out-json "${RESULT_JSON}"

"${PYTHON_BIN}" -m clearvla.tools.summarize_v101_action_path_probe "${RESULT_JSON}"
