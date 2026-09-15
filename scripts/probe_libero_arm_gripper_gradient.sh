#!/usr/bin/env bash
# Read-only one-batch arm/gripper shared-gradient attribution for LIBERO.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${CLEARVLA_PYTHON:-/data/senwang/envs/clearvla-schema30-boundary/bin/python}"
CONFIG="${MAINLINE_CONFIG:-configs/mainline/object_intent_dynamics_323.json}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT must name a LIBERO mainline checkpoint}"
OUT="${PROBE_OUTPUT:-}"

ARGS=(
  --config "${CONFIG}"
  --checkpoint "${CHECKPOINT}"
  --device "${MAINLINE_DEVICE:-auto}"
  --batch-size "${MAINLINE_BATCH_SIZE:-2}"
  --num-workers "${MAINLINE_NUM_WORKERS:-0}"
  --probe-seed "${PROBE_SEED:-20260909}"
)
if [[ -n "${OUT}" ]]; then
  ARGS+=(--output "${OUT}")
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
printf '[libero-arm-gripper-gradient] checkpoint=%s batch=%s device=%s output=%s\n' \
  "${CHECKPOINT}" "${MAINLINE_BATCH_SIZE:-2}" "${MAINLINE_DEVICE:-auto}" \
  "${OUT:-stdout-only}"
exec "${PYTHON_BIN}" -u -m clearvla.tools.probe_arm_gripper_gradient "${ARGS[@]}" "$@"
