#!/usr/bin/env bash
# Run action-boundary diagnostics and display policy RGB through SSH X11.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ENV="${CLEARVLA_SIM_ENV:-/data/senwang/envs/clearvla-sim}"
SIM_DATA_ROOT="${CLEARVLA_SIM_DATA_ROOT:-/data/senwang/data/clearvla_sim}"
ASSET_ROOT="${CLEARVLA_SIM_ASSET_ROOT:-${SIM_DATA_ROOT}/assets}"
RUN_ID="${CLEARVLA_SIM_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${CLEARVLA_EXTREME_OUTPUT:-${SIM_DATA_ROOT}/diagnostics/maniskill_stackcube_extremes/${RUN_ID}}"

if [[ -z "${DISPLAY:-}" ]]; then
  printf 'DISPLAY is empty; connect with ssh -Y while the MobaXterm X server is running\n' >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export MS_ASSET_DIR="${ASSET_ROOT}/maniskill"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XDG_CACHE_HOME="${ASSET_ROOT}/xdg-cache"
export TORCH_HOME="${ASSET_ROOT}/torch"
export HF_HOME="${ASSET_ROOT}/huggingface"
mkdir -p "${OUTPUT_DIR}" "${MS_ASSET_DIR}" "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${HF_HOME}"

cd "${REPO_ROOT}"
exec "${SIM_ENV}/bin/python" -m clearvla.simulation.extreme_smoke \
  --environment maniskill-stackcube \
  --output-dir "${OUTPUT_DIR}" \
  --steps-per-case "${CLEARVLA_EXTREME_STEPS:-3}" \
  --display \
  --display-ms "${CLEARVLA_EXTREME_DISPLAY_MS:-700}" \
  "$@"


