#!/usr/bin/env bash
# V87: complete orthonormal DCT coefficient state with a continuous,
# group-aware bandwidth aperture and direct typed frequency readers.
# The deployment interface remains the original physical 24x7 field. The
# train/inference bridge, velocity target, and integrator state are DCT
# coefficients; arm and gripper use their respective linear contracts there.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HIERARCHICAL_MMDIT_SPECTRAL_STATE="${HIERARCHICAL_MMDIT_SPECTRAL_STATE:-1}"
export ARM_FLOW_MODE="${ARM_FLOW_MODE:-manifold_native}"
export GRIPPER_FIELD_MODE="${GRIPPER_FIELD_MODE:-parseval_temporal}"
export HIERARCHICAL_MMDIT_SPECTRAL_ARM_START_FRACTION="${HIERARCHICAL_MMDIT_SPECTRAL_ARM_START_FRACTION:-0.16}"
export HIERARCHICAL_MMDIT_SPECTRAL_GRIPPER_START_FRACTION="${HIERARCHICAL_MMDIT_SPECTRAL_GRIPPER_START_FRACTION:-0.33}"
export HIERARCHICAL_MMDIT_SPECTRAL_TEMPERATURE="${HIERARCHICAL_MMDIT_SPECTRAL_TEMPERATURE:-1.5}"
export HIERARCHICAL_MMDIT_SPECTRAL_SCHEDULE_POWER="${HIERARCHICAL_MMDIT_SPECTRAL_SCHEDULE_POWER:-1.0}"
export HIERARCHICAL_MMDIT_SPECTRAL_CONTROLLER_SHIFT_LIMIT="${HIERARCHICAL_MMDIT_SPECTRAL_CONTROLLER_SHIFT_LIMIT:-2.0}"
export HIERARCHICAL_MMDIT_SPECTRAL_COMPETITION_LOSS_WEIGHT="${HIERARCHICAL_MMDIT_SPECTRAL_COMPETITION_LOSS_WEIGHT:-0.0}"
export HIERARCHICAL_MMDIT_SPECTRAL_COMPETITION_WARMUP_STEPS="${HIERARCHICAL_MMDIT_SPECTRAL_COMPETITION_WARMUP_STEPS:-200}"
export EVAL_SAMPLING_DIAGNOSTIC_BATCHES="${EVAL_SAMPLING_DIAGNOSTIC_BATCHES:-16}"
export EVAL_PROPOSAL_ABLATION_BATCHES="${EVAL_PROPOSAL_ABLATION_BATCHES:-16}"
export OUT_DIR="${OUT_DIR:-runs/v87_spectral_controller}"

printf '[v87] full_dct_state=%s arm_mode=%s gripper_mode=%s arm_start=%s gripper_start=%s temperature=%s competition_weight=%s\n' \
  "${HIERARCHICAL_MMDIT_SPECTRAL_STATE}" \
  "${ARM_FLOW_MODE}" \
  "${GRIPPER_FIELD_MODE}" \
  "${HIERARCHICAL_MMDIT_SPECTRAL_ARM_START_FRACTION}" \
  "${HIERARCHICAL_MMDIT_SPECTRAL_GRIPPER_START_FRACTION}" \
  "${HIERARCHICAL_MMDIT_SPECTRAL_TEMPERATURE}" \
  "${HIERARCHICAL_MMDIT_SPECTRAL_COMPETITION_LOSS_WEIGHT}"

exec bash "${SCRIPT_DIR}/current_v85_unified_controller.sh" \
  --hierarchical-mmdit-spectral-state "${HIERARCHICAL_MMDIT_SPECTRAL_STATE}" \
  --arm-flow-mode "${ARM_FLOW_MODE}" \
  --gripper-field-mode "${GRIPPER_FIELD_MODE}" \
  --hierarchical-mmdit-spectral-arm-start-fraction \
    "${HIERARCHICAL_MMDIT_SPECTRAL_ARM_START_FRACTION}" \
  --hierarchical-mmdit-spectral-gripper-start-fraction \
    "${HIERARCHICAL_MMDIT_SPECTRAL_GRIPPER_START_FRACTION}" \
  --hierarchical-mmdit-spectral-temperature \
    "${HIERARCHICAL_MMDIT_SPECTRAL_TEMPERATURE}" \
  --hierarchical-mmdit-spectral-schedule-power \
    "${HIERARCHICAL_MMDIT_SPECTRAL_SCHEDULE_POWER}" \
  --hierarchical-mmdit-spectral-controller-shift-limit \
    "${HIERARCHICAL_MMDIT_SPECTRAL_CONTROLLER_SHIFT_LIMIT}" \
  --hierarchical-mmdit-spectral-competition-loss-weight \
    "${HIERARCHICAL_MMDIT_SPECTRAL_COMPETITION_LOSS_WEIGHT}" \
  --hierarchical-mmdit-spectral-competition-warmup-steps \
    "${HIERARCHICAL_MMDIT_SPECTRAL_COMPETITION_WARMUP_STEPS}" \
  --eval-sampling-diagnostic-batches "${EVAL_SAMPLING_DIAGNOSTIC_BATCHES}" \
  --eval-proposal-ablation-batches "${EVAL_PROPOSAL_ABLATION_BATCHES}" \
  --hierarchical-mmdit-unified-controller 1 \
  "$@"
