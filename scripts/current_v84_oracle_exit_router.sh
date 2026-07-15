#!/usr/bin/env bash
# =============================================================================
# V84: same operator, two depth-routing experiments.
#
#   dynamic: V83 dynamic stage selector with random-dwell prefix training.
#   oracle:  full fixed prefixes supervise a post-block learned exit head;
#            evaluation keeps the learned route in detached shadow mode.
#
# The oracle target is computed from detached physical flow errors. It updates
# only the exit head and never enters the action, workspace, or stage values.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HIERARCHICAL_MMDIT_DEPTH="${HIERARCHICAL_MMDIT_DEPTH:-3}"
export HIERARCHICAL_MMDIT_REFINE_STEPS="${HIERARCHICAL_MMDIT_REFINE_STEPS:-$((2 * HIERARCHICAL_MMDIT_DEPTH))}"
export HIERARCHICAL_MMDIT_OPERATOR_STAGES="${HIERARCHICAL_MMDIT_OPERATOR_STAGES:-$((2 * HIERARCHICAL_MMDIT_DEPTH))}"
export HIERARCHICAL_MMDIT_RANDOM_PREFIX_PROBABILITY="${HIERARCHICAL_MMDIT_RANDOM_PREFIX_PROBABILITY:-0.0}"
export HIERARCHICAL_MMDIT_ORACLE_ROUTE_LOSS_WEIGHT="${HIERARCHICAL_MMDIT_ORACLE_ROUTE_LOSS_WEIGHT:-0.05}"
export HIERARCHICAL_MMDIT_ORACLE_ROUTE_RELATIVE_TOLERANCE="${HIERARCHICAL_MMDIT_ORACLE_ROUTE_RELATIVE_TOLERANCE:-0.0}"
export HIERARCHICAL_MMDIT_ORACLE_ROUTE_WARMUP_STEPS="${HIERARCHICAL_MMDIT_ORACLE_ROUTE_WARMUP_STEPS:-200}"
V84_VARIANT="${V84_VARIANT:-dynamic}"

case "${V84_VARIANT}" in
  dynamic)
    schedule_mode=random_dwell
    exhaustion_mode=off
    route_weight=0
    ;;
  oracle)
    schedule_mode=fixed
    exhaustion_mode=learned_shadow
    route_weight="${HIERARCHICAL_MMDIT_ORACLE_ROUTE_LOSS_WEIGHT}"
    ;;
  *)
    printf 'V84_VARIANT must be dynamic or oracle (got %s).\n' "${V84_VARIANT}" >&2
    exit 2
    ;;
esac

export OUT_DIR="${OUT_DIR:-runs/v84_${V84_VARIANT}_d${HIERARCHICAL_MMDIT_DEPTH}_r${HIERARCHICAL_MMDIT_REFINE_STEPS}_s${HIERARCHICAL_MMDIT_OPERATOR_STAGES}_b8}"

printf '[v84] variant=%s train_schedule=%s eval_exhaustion=%s route_weight=%s depth=%s steps=%s stages=%s\n' \
  "${V84_VARIANT}" "${schedule_mode}" "${exhaustion_mode}" "${route_weight}" \
  "${HIERARCHICAL_MMDIT_DEPTH}" "${HIERARCHICAL_MMDIT_REFINE_STEPS}" \
  "${HIERARCHICAL_MMDIT_OPERATOR_STAGES}"

exec bash "${SCRIPT_DIR}/current_v82_sidecar_nested_contraction.sh" \
  --hierarchical-mmdit-schedule-mode "${schedule_mode}" \
  --hierarchical-mmdit-random-prefix-probability "${HIERARCHICAL_MMDIT_RANDOM_PREFIX_PROBABILITY}" \
  --hierarchical-mmdit-exhaustion-mode "${exhaustion_mode}" \
  --hierarchical-mmdit-action-response-thresholds 0 0 0 \
  --hierarchical-mmdit-stage-pressure-thresholds 0 0 0 \
  --hierarchical-mmdit-oracle-route-loss-weight "${route_weight}" \
  --hierarchical-mmdit-oracle-route-relative-tolerance \
    "${HIERARCHICAL_MMDIT_ORACLE_ROUTE_RELATIVE_TOLERANCE}" \
  --hierarchical-mmdit-oracle-route-warmup-steps \
    "${HIERARCHICAL_MMDIT_ORACLE_ROUTE_WARMUP_STEPS}" \
  "$@"
