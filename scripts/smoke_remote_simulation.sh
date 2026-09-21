#!/usr/bin/env bash
# Exercise observation -> history -> action -> control -> HDF5 on both backends.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ENV="${CLEARVLA_SIM_ENV:-/data/senwang/envs/clearvla-sim}"
SIM_DATA_ROOT="${CLEARVLA_SIM_DATA_ROOT:-/data/senwang/data/clearvla_sim}"
ASSET_ROOT="${CLEARVLA_SIM_ASSET_ROOT:-${SIM_DATA_ROOT}/assets}"
PYTHON_BIN="${SIM_ENV}/bin/python"
RUN_ID="${CLEARVLA_SIM_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'missing simulator Python: %s; run setup_remote_simulation.sh first\n' \
    "${PYTHON_BIN}" >&2
  exit 2
fi

export MS_ASSET_DIR="${ASSET_ROOT}/maniskill"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XDG_CACHE_HOME="${ASSET_ROOT}/xdg-cache"
export TORCH_HOME="${ASSET_ROOT}/torch"
export HF_HOME="${ASSET_ROOT}/huggingface"
mkdir -p \
  "${MS_ASSET_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${TORCH_HOME}" \
  "${HF_HOME}" \
  "${SIM_DATA_ROOT}/alicia_proxy_smoke" \
  "${SIM_DATA_ROOT}/maniskill_stackcube_v1"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m clearvla.simulation.rollout \
  --environment alicia-proxy \
  --policy hold \
  --seed 0 \
  --steps 8 \
  --max-episode-steps 200 \
  --record-dir "${SIM_DATA_ROOT}/alicia_proxy_smoke" \
  --episode-id "episode_${RUN_ID}" \
  --json-out "${SIM_DATA_ROOT}/alicia_proxy_smoke/smoke_${RUN_ID}.json"

"${PYTHON_BIN}" -m clearvla.simulation.rollout \
  --environment maniskill-stackcube \
  --policy hold \
  --seed 0 \
  --steps 8 \
  --max-episode-steps 200 \
  --maniskill-sim-backend physx_cpu \
  --maniskill-render-backend gpu \
  --record-dir "${SIM_DATA_ROOT}/maniskill_stackcube_v1" \
  --episode-id "smoke_${RUN_ID}" \
  --json-out "${SIM_DATA_ROOT}/maniskill_stackcube_v1/smoke_${RUN_ID}.json"

"${PYTHON_BIN}" -m clearvla.simulation.dataset \
  "${SIM_DATA_ROOT}/alicia_proxy_smoke"
"${PYTHON_BIN}" -m clearvla.simulation.dataset \
  "${SIM_DATA_ROOT}/maniskill_stackcube_v1"

printf '[clearvla-sim-smoke] run_id=%s data=%s assets=%s\n' \
  "${RUN_ID}" "${SIM_DATA_ROOT}" "${ASSET_ROOT}"


