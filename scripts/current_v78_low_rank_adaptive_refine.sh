#!/usr/bin/env bash
# =============================================================================
# V78: shared MMDiT core + mandatory conditional low-rank stage operators.
#
# Training uses monotonic randomized dwell so stage identity is not an alias
# for absolute loop index.  Exhaustion is off by default: the first run records
# detached action-response/stage-pressure probes used to calibrate t-binned
# thresholds.  Set HIERARCHICAL_MMDIT_EXHAUSTION_MODE=shadow only after that
# calibration; adaptive changes real evaluation execution and is deliberately
# never enabled by an uncalibrated default.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HIERARCHICAL_MMDIT_DEPTH="${HIERARCHICAL_MMDIT_DEPTH:-3}"
export HIERARCHICAL_MMDIT_REFINE_STEPS="${HIERARCHICAL_MMDIT_REFINE_STEPS:-6}"
export HIERARCHICAL_MMDIT_OPERATOR_RANK="${HIERARCHICAL_MMDIT_OPERATOR_RANK:-32}"
export HIERARCHICAL_MMDIT_OPERATOR_GROUPS="${HIERARCHICAL_MMDIT_OPERATOR_GROUPS:-8}"
export HIERARCHICAL_MMDIT_OPERATOR_STRENGTH_INIT="${HIERARCHICAL_MMDIT_OPERATOR_STRENGTH_INIT:-0.05}"
export HIERARCHICAL_MMDIT_OPERATOR_STRENGTH_MAX="${HIERARCHICAL_MMDIT_OPERATOR_STRENGTH_MAX:-0.50}"
export HIERARCHICAL_MMDIT_SCHEDULE_MODE="${HIERARCHICAL_MMDIT_SCHEDULE_MODE:-random_dwell}"
export HIERARCHICAL_MMDIT_RANDOM_PREFIX_PROBABILITY="${HIERARCHICAL_MMDIT_RANDOM_PREFIX_PROBABILITY:-0.0}"
export HIERARCHICAL_MMDIT_EXHAUSTION_MODE="${HIERARCHICAL_MMDIT_EXHAUSTION_MODE:-off}"
export HIERARCHICAL_MMDIT_ACTION_RESPONSE_THRESHOLDS="${HIERARCHICAL_MMDIT_ACTION_RESPONSE_THRESHOLDS:-0 0 0}"
export HIERARCHICAL_MMDIT_STAGE_PRESSURE_THRESHOLDS="${HIERARCHICAL_MMDIT_STAGE_PRESSURE_THRESHOLDS:-0 0 0}"
export HIERARCHICAL_MMDIT_ACTION_RESPONSE_FLOOR="${HIERARCHICAL_MMDIT_ACTION_RESPONSE_FLOOR:-0.05}"
export HIERARCHICAL_MMDIT_EXHAUSTION_CONFIRM_STEPS="${HIERARCHICAL_MMDIT_EXHAUSTION_CONFIRM_STEPS:-2}"
export OUT_DIR="${OUT_DIR:-runs/v78_low_rank_adaptive_d${HIERARCHICAL_MMDIT_DEPTH}_s${HIERARCHICAL_MMDIT_REFINE_STEPS}_b8}"

read -r -a action_thresholds <<< "${HIERARCHICAL_MMDIT_ACTION_RESPONSE_THRESHOLDS}"
read -r -a stage_thresholds <<< "${HIERARCHICAL_MMDIT_STAGE_PRESSURE_THRESHOLDS}"
if (( ${#action_thresholds[@]} != 3 || ${#stage_thresholds[@]} != 3 )); then
  printf '%s\n' 'V78 exhaustion threshold variables must each contain exactly three numbers.' >&2
  exit 2
fi

exec bash "${SCRIPT_DIR}/current_v76a_owned_intent_mmdit.sh" \
  --hierarchical-mmdit-depth "${HIERARCHICAL_MMDIT_DEPTH}" \
  --hierarchical-mmdit-refine-steps "${HIERARCHICAL_MMDIT_REFINE_STEPS}" \
  --hierarchical-mmdit-operator-rank "${HIERARCHICAL_MMDIT_OPERATOR_RANK}" \
  --hierarchical-mmdit-operator-groups "${HIERARCHICAL_MMDIT_OPERATOR_GROUPS}" \
  --hierarchical-mmdit-operator-strength-init "${HIERARCHICAL_MMDIT_OPERATOR_STRENGTH_INIT}" \
  --hierarchical-mmdit-operator-strength-max "${HIERARCHICAL_MMDIT_OPERATOR_STRENGTH_MAX}" \
  --hierarchical-mmdit-schedule-mode "${HIERARCHICAL_MMDIT_SCHEDULE_MODE}" \
  --hierarchical-mmdit-random-prefix-probability "${HIERARCHICAL_MMDIT_RANDOM_PREFIX_PROBABILITY}" \
  --hierarchical-mmdit-exhaustion-mode "${HIERARCHICAL_MMDIT_EXHAUSTION_MODE}" \
  --hierarchical-mmdit-action-response-thresholds "${action_thresholds[@]}" \
  --hierarchical-mmdit-stage-pressure-thresholds "${stage_thresholds[@]}" \
  --hierarchical-mmdit-action-response-floor "${HIERARCHICAL_MMDIT_ACTION_RESPONSE_FLOOR}" \
  --hierarchical-mmdit-exhaustion-confirm-steps "${HIERARCHICAL_MMDIT_EXHAUSTION_CONFIRM_STEPS}" \
  "$@"
