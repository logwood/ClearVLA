#!/usr/bin/env bash
# Frozen V104 bridge/call-contract/off-path/solver probe.
#
# The checkpoint is only read.  Every comparison in a batch shares the same
# validation sample, source noise and non-action conditions.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CHECKPOINT="${CHECKPOINT:-runs/v104_sequential_bounded_flow_jepa/checkpoints/latest.pt}"
DATA_ROOT="${DATA_ROOT:-/data/liang.zhang/dataset/grab_pen_single/grab_pen_single}"
CACHE_DIR="${CACHE_DIR:-/home/sen.wang/workspace/robotics/clear/data/cache_336}"
DINO_CACHE_DIR="${DINO_CACHE_DIR:-/home/sen.wang/workspace/robotics/clear/data/dinov2_cache_336}"
DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-runs/diagnostics/v104_sampling_path}"
PROBE_BATCHES="${PROBE_BATCHES:-2}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SOLVER_STEPS="${SOLVER_STEPS:-5 10 20}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_JSON="${RESULT_JSON:-${DIAGNOSTICS_DIR}/sampling_path_${PROBE_BATCHES}b.json}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  printf 'checkpoint not found: %s\n' "${CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "$(dirname "${RESULT_JSON}")"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

printf '[v104-sampling-path-probe] checkpoint=%s batches=%s solver_steps=%s result=%s\n' \
  "${CHECKPOINT}" "${PROBE_BATCHES}" "${SOLVER_STEPS}" "${RESULT_JSON}"

# Intentional word splitting turns "5 10 20" into argparse's integer list.
# shellcheck disable=SC2086
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
  --sampling-path-probe-batches "${PROBE_BATCHES}" \
  --sampling-path-probe-steps ${SOLVER_STEPS} \
  --sampling-path-require-v104-contract \
  --out-json "${RESULT_JSON}"
