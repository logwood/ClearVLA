#!/usr/bin/env bash
# =============================================================================
# V83: V82 nested contraction plus restored adaptive-depth training/evaluation.
#
# One entry point owns the three attribution phases:
#   calibrate: random-dwell training, fixed six-step evaluation, probe logging.
#   shadow:    same training/main evaluation plus detached adaptive shadow route.
#   adaptive:  same training, adaptive execution becomes the evaluation output.
#
# Shadow and adaptive phases require three calibrated thresholds for the same
# flow-time bins reported as hmuT50/hmpT50. No default threshold is invented.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HIERARCHICAL_MMDIT_DEPTH="${HIERARCHICAL_MMDIT_DEPTH:-3}"
export HIERARCHICAL_MMDIT_REFINE_STEPS="${HIERARCHICAL_MMDIT_REFINE_STEPS:-6}"
export HIERARCHICAL_MMDIT_OPERATOR_STAGES="${HIERARCHICAL_MMDIT_OPERATOR_STAGES:-$((2 * HIERARCHICAL_MMDIT_DEPTH))}"
export HIERARCHICAL_MMDIT_RANDOM_PREFIX_PROBABILITY="${HIERARCHICAL_MMDIT_RANDOM_PREFIX_PROBABILITY:-0.0}"
V83_PHASE="${V83_PHASE:-calibrate}"

case "${V83_PHASE}" in
  calibrate)
    exhaustion_mode=off
    action_thresholds=(0 0 0)
    stage_thresholds=(0 0 0)
    ;;
  shadow|adaptive)
    exhaustion_mode="${V83_PHASE}"
    action_threshold_text="${HIERARCHICAL_MMDIT_ACTION_RESPONSE_THRESHOLDS:-}"
    stage_threshold_text="${HIERARCHICAL_MMDIT_STAGE_PRESSURE_THRESHOLDS:-}"
    if [[ -z "${action_threshold_text}" || -z "${stage_threshold_text}" ]]; then
      printf '%s\n' \
        'V83 shadow/adaptive requires calibrated three-bin action and stage thresholds.' \
        'Set HIERARCHICAL_MMDIT_ACTION_RESPONSE_THRESHOLDS="t0 t1 t2" and' \
        'HIERARCHICAL_MMDIT_STAGE_PRESSURE_THRESHOLDS="t0 t1 t2".' >&2
      exit 2
    fi
    read -r -a action_thresholds <<< "${action_threshold_text}"
    read -r -a stage_thresholds <<< "${stage_threshold_text}"
    if (( ${#action_thresholds[@]} != 3 || ${#stage_thresholds[@]} != 3 )); then
      printf '%s\n' 'V83 threshold variables must each contain exactly three numbers.' >&2
      exit 2
    fi
    ;;
  *)
    printf 'V83_PHASE must be calibrate, shadow, or adaptive (got %s).\n' \
      "${V83_PHASE}" >&2
    exit 2
    ;;
esac

export OUT_DIR="${OUT_DIR:-runs/v83_${V83_PHASE}_nested_adaptive_d${HIERARCHICAL_MMDIT_DEPTH}_r${HIERARCHICAL_MMDIT_REFINE_STEPS}_s${HIERARCHICAL_MMDIT_OPERATOR_STAGES}_b8}"

printf '[v83] phase=%s train_schedule=random_dwell eval_exhaustion=%s depth=%s steps=%s stages=%s\n' \
  "${V83_PHASE}" "${exhaustion_mode}" "${HIERARCHICAL_MMDIT_DEPTH}" \
  "${HIERARCHICAL_MMDIT_REFINE_STEPS}" "${HIERARCHICAL_MMDIT_OPERATOR_STAGES}"

exec bash "${SCRIPT_DIR}/current_v82_sidecar_nested_contraction.sh" \
  --hierarchical-mmdit-schedule-mode random_dwell \
  --hierarchical-mmdit-random-prefix-probability "${HIERARCHICAL_MMDIT_RANDOM_PREFIX_PROBABILITY}" \
  --hierarchical-mmdit-exhaustion-mode "${exhaustion_mode}" \
  --hierarchical-mmdit-action-response-thresholds "${action_thresholds[@]}" \
  --hierarchical-mmdit-stage-pressure-thresholds "${stage_thresholds[@]}" \
  "$@"
