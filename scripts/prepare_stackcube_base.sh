#!/usr/bin/env bash
# Explicit, new-namespace preparation. Never starts training or selects a GPU.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
: "${STACKCUBE_ROOT:?Set a new /data-owned StackCube pilot output root}"
: "${STACKCUBE_TRAJECTORY:?Set the converted pd_ee_delta_pose HDF5 path}"
: "${STACKCUBE_METADATA:?Set the converted trajectory JSON path}"
: "${STACKCUBE_COUNT:?Set the number of verified source episodes to replay}"
: "${CUDA_VISIBLE_DEVICES:?Select an available GPU explicitly}"
PYTHON_BIN="${CLEARVLA_SIM_PYTHON:-/data/senwang/envs/clearvla-sim/bin/python}"
EPISODE_STEPS="${STACKCUBE_EPISODE_STEPS:-400}"
if [[ -e "${STACKCUBE_ROOT}" ]]; then
  echo "Use a new StackCube output namespace: ${STACKCUBE_ROOT}" >&2
  exit 2
fi
# Existing source conversion is immutable. The importer replays against the
# actual wristcam robot, then records 48 REAL zero-EE/hold-gripper steps.
"${PYTHON_BIN}" -m clearvla.simulation.maniskill_import \
  --trajectory "${STACKCUBE_TRAJECTORY}" --metadata "${STACKCUBE_METADATA}" \
  --output-dir "${STACKCUBE_ROOT}/experts" --count "${STACKCUBE_COUNT}" \
  --min-steps 58 --settle-steps 48 --state-chart fixed_down_causal_rotvec_v2 \
  --max-episode-steps "${EPISODE_STEPS}" \
  --require-all-success
"${PYTHON_BIN}" -m clearvla.rl.prepare_base \
  --dataset "${STACKCUBE_ROOT}/experts" --output "${STACKCUBE_ROOT}/prepared" \
  --cache-root "${STACKCUBE_ROOT}/caches" \
  --language-bank "${STACKCUBE_ROOT}/language/t5_xxl.pt" \
  --run-root "${STACKCUBE_ROOT}/bc" \
  --gripper-output-mode maniskill_binary_command
"${PYTHON_BIN}" -m clearvla.benchmarks.language_bank \
  --inventory "${STACKCUBE_ROOT}/prepared/instructions.json" \
  --output "${STACKCUBE_ROOT}/language/t5_xxl.pt" \
  --model google/t5-v1_1-xxl --max-tokens 32 --device cuda --local-files-only
"${PYTHON_BIN}" -m clearvla.cli.build_decoded_image_cache \
  --data-root "${STACKCUBE_ROOT}/experts" --cache-dir "${STACKCUBE_ROOT}/caches/decoded_336" \
  --cameras top wrist --top-key observations/images/cam_high \
  --wrist-key observations/images/cam_right_wrist --resize 336 336
"${PYTHON_BIN}" -m clearvla.cli.build_dinov2_token_cache \
  --data-root "${STACKCUBE_ROOT}/experts" \
  --decoded-image-cache-dir "${STACKCUBE_ROOT}/caches/decoded_336" \
  --out-dir "${STACKCUBE_ROOT}/caches/dinov2_base_336" \
  --cameras top wrist --state-key state --top-key observations/images/cam_high \
  --wrist-key observations/images/cam_right_wrist --cache-resize 336 336 \
  --dinov2-model facebook/dinov2-base --dinov2-local-files-only \
  --device cuda --dtype bf16 --batch-size 32
echo "Prepared base inputs only. Run typed CUDA smoke and checkpoint validation before formal BC."
