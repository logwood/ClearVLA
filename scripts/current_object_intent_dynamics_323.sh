#!/usr/bin/env bash
# Capability-named object/intent/dynamics 3-2-3 mainline.
#
# V121 is only the default run label.  Serialized identity is the compact
# object_intent_dynamics_323 ArchitectureManifest.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

for argument in "$@"; do
  case "${argument}" in
    --resume|--resume=*)
      echo "[object-intent-dynamics-323] the new top requires a fresh checkpoint" >&2
      exit 2
      ;;
  esac
done

export OUT_DIR="${OUT_DIR:-runs/v121_object_intent_dynamics_323_typed_dock}"
export OBJECT_323_BATCH_SIZE="${OBJECT_323_BATCH_SIZE:-8}"
export V115_BATCH_SIZE="${OBJECT_323_BATCH_SIZE}"
export FLOW_JEPA_PARENT_VERSION=object_intent_dynamics_323
export CLEARVLA_REQUIRED_MODEL_CONTRACT=object_intent_dynamics_323

# Keep the established raw-HDF5 root.  Cached tensors and model weights use
# the storage roots requested for the current server.
export CLEARVLA_DATA_CACHE_ROOT="${CLEARVLA_DATA_CACHE_ROOT:-/data/senwang/data}"
export CLEARVLA_CHECKPOINT_ROOT="${CLEARVLA_CHECKPOINT_ROOT:-/data/senwang/checkpoint}"
export DATA_ROOT="${DATA_ROOT:-/data/liang.zhang/dataset/grab_pen_single/grab_pen_single}"
export CACHE_DIR="${CACHE_DIR:-${CLEARVLA_DATA_CACHE_ROOT}/cache_336}"
export DINO_CACHE_DIR="${DINO_CACHE_DIR:-${CLEARVLA_DATA_CACHE_ROOT}/dinov2_cache_336}"
export T5_CONDITION_PATH="${T5_CONDITION_PATH:-${CLEARVLA_CHECKPOINT_ROOT}/grasp_pen_embed.pt}"

printf '[object-intent-dynamics-323] run_label=v121 topology=3-2-3 G=shared_object_chart+typed_verifiers S=factorized_stateless_intent W=typed_four_interval_object_dynamics P1=single_precision_object_dock P2=typed_zero_preserving_effect P3=protected_consequence+precision+temporal+state_change bottom_ingress=single completion_terminal=off fresh=1 stage1_init=off batch=%s\n' \
  "${OBJECT_323_BATCH_SIZE}"
printf '[object-intent-dynamics-323-paths] data=%s decoded_cache=%s dino_cache=%s t5=%s out=%s\n' \
  "${DATA_ROOT}" "${CACHE_DIR}" "${DINO_CACHE_DIR}" \
  "${T5_CONDITION_PATH}" "${OUT_DIR}"

exec bash "${SCRIPT_DIR}/current_v115_g_aligned_goal_phase_323.sh" \
  "$@" \
  --flow-jepa-object-intent-dynamics-mainline 1 \
  --flow-jepa-grounded-intent-effect-mainline 0 \
  --flow-jepa-differential-intent-effect-mainline 0 \
  --flow-jepa-supervised-effect-mainline 0 \
  --flow-jepa-stateless-intent-controller 0 \
  --flow-jepa-window-effect-bank 0 \
  --flow-jepa-effect-read-in-p2 0 \
  --flow-jepa-future-slots 4 \
  --flow-matching-time-distribution beta_1_5_1
