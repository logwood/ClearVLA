#!/usr/bin/env bash
# Convert official motion-planning demonstrations and render ClearVLA wristcam data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ENV="${CLEARVLA_SIM_ENV:-/data/senwang/envs/clearvla-sim}"
SIM_DATA_ROOT="${CLEARVLA_SIM_DATA_ROOT:-/data/senwang/data/clearvla_sim}"
ASSET_ROOT="${CLEARVLA_SIM_ASSET_ROOT:-${SIM_DATA_ROOT}/assets}"
SOURCE_DIR="${CLEARVLA_DEMO_SOURCE:-${SIM_DATA_ROOT}/official_demos/StackCube-v1/motionplanning}"
CONVERT_DIR="${CLEARVLA_DEMO_CONVERTED:-${SIM_DATA_ROOT}/converted_demos/stackcube_pd_ee_delta_pose_73}"
OUTPUT_DIR="${CLEARVLA_EXPERT_DATASET:-${SIM_DATA_ROOT}/datasets/maniskill_stackcube_v1_expert}"
DEMO_COUNT="${CLEARVLA_DEMO_COUNT:-73}"
CONVERTED_H5="${CONVERT_DIR}/trajectory.none.pd_ee_delta_pose.physx_cpu.h5"
CONVERTED_JSON="${CONVERT_DIR}/trajectory.none.pd_ee_delta_pose.physx_cpu.json"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export MS_ASSET_DIR="${ASSET_ROOT}/maniskill"
export XDG_CACHE_HOME="${ASSET_ROOT}/xdg-cache"
export TORCH_HOME="${ASSET_ROOT}/torch"
export HF_HOME="${ASSET_ROOT}/huggingface"
mkdir -p "${CONVERT_DIR}" "${OUTPUT_DIR}" "${MS_ASSET_DIR}" "${XDG_CACHE_HOME}"

if [[ ! -f "${CONVERTED_H5}" || ! -f "${CONVERTED_JSON}" ]]; then
  if [[ -e "${CONVERT_DIR}/trajectory.h5" || -e "${CONVERT_DIR}/trajectory.json" ]]; then
    printf 'partial conversion inputs already exist under %s; refusing ambiguity\n' \
      "${CONVERT_DIR}" >&2
    exit 2
  fi
  cp "${SOURCE_DIR}/trajectory.h5" "${CONVERT_DIR}/trajectory.h5"
  cp "${SOURCE_DIR}/trajectory.json" "${CONVERT_DIR}/trajectory.json"
  "${SIM_ENV}/bin/python" -m mani_skill.trajectory.replay_trajectory \
    --traj-path "${CONVERT_DIR}/trajectory.h5" \
    --sim-backend physx_cpu \
    --obs-mode none \
    --target-control-mode pd_ee_delta_pose \
    --save-traj \
    --count "${DEMO_COUNT}" \
    --max-retry 1 \
    --reward-mode sparse \
    --record-rewards \
    --no-verbose
fi

cd "${REPO_ROOT}"
exec "${SIM_ENV}/bin/python" -m clearvla.simulation.maniskill_import \
  --trajectory "${CONVERTED_H5}" \
  --metadata "${CONVERTED_JSON}" \
  --output-dir "${OUTPUT_DIR}" \
  --count "${DEMO_COUNT}" \
  --min-steps 58 \
  --require-all-success


