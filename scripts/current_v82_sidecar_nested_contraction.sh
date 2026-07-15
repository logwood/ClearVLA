#!/usr/bin/env bash
# =============================================================================
# V82: distinct full-rank blocks with external nested-contraction sidecars.
#
# The complete V77 branch update (including its host gate) is built first.
# Six semantic stages then own only a non-expansive contraction sidecar. During
# identity warm-up the sidecar is absent from both the value and Jacobian; no
# second residual-amplitude controller is introduced.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HIERARCHICAL_MMDIT_DEPTH="${HIERARCHICAL_MMDIT_DEPTH:-3}"
export HIERARCHICAL_MMDIT_REFINE_STEPS="${HIERARCHICAL_MMDIT_REFINE_STEPS:-${HIERARCHICAL_MMDIT_DEPTH}}"
export HIERARCHICAL_MMDIT_OPERATOR_STAGES="${HIERARCHICAL_MMDIT_OPERATOR_STAGES:-$((2 * HIERARCHICAL_MMDIT_DEPTH))}"
export HIERARCHICAL_MMDIT_OPERATOR_RANK="${HIERARCHICAL_MMDIT_OPERATOR_RANK:-32}"
export HIERARCHICAL_MMDIT_OPERATOR_GROUPS="${HIERARCHICAL_MMDIT_OPERATOR_GROUPS:-32}"
export HIERARCHICAL_MMDIT_DEPTH_LOGIT_INIT="${HIERARCHICAL_MMDIT_DEPTH_LOGIT_INIT:-2.0}"
export HIERARCHICAL_MMDIT_CONTRACTION_WARMUP_STEPS="${HIERARCHICAL_MMDIT_CONTRACTION_WARMUP_STEPS:-200}"
export HIERARCHICAL_MMDIT_CONTRACTION_TRANSITION_STEPS="${HIERARCHICAL_MMDIT_CONTRACTION_TRANSITION_STEPS:-1500}"
export HIERARCHICAL_MMDIT_DEPTH_USAGE_LOSS_WEIGHT="${HIERARCHICAL_MMDIT_DEPTH_USAGE_LOSS_WEIGHT:-0.0002}"
export HIERARCHICAL_MMDIT_CONTRACTION_LR_SCALE="${HIERARCHICAL_MMDIT_CONTRACTION_LR_SCALE:-2.0}"
export HIERARCHICAL_MMDIT_RESIDUAL_SCALE_INIT="${HIERARCHICAL_MMDIT_RESIDUAL_SCALE_INIT:-0.05}"
export HIERARCHICAL_MMDIT_RESIDUAL_SCALE_MAX="${HIERARCHICAL_MMDIT_RESIDUAL_SCALE_MAX:-0.20}"
export OUT_DIR="${OUT_DIR:-runs/v82_post_gate_sidecar_d${HIERARCHICAL_MMDIT_DEPTH}_r${HIERARCHICAL_MMDIT_REFINE_STEPS}_s${HIERARCHICAL_MMDIT_OPERATOR_STAGES}_b8}"

if (( HIERARCHICAL_MMDIT_REFINE_STEPS < HIERARCHICAL_MMDIT_DEPTH )); then
  printf '%s\n' 'V82 requires enough refinement steps to execute every full-rank block.' >&2
  exit 2
fi
if (( HIERARCHICAL_MMDIT_OPERATOR_STAGES < HIERARCHICAL_MMDIT_DEPTH )); then
  printf '%s\n' 'V82 requires at least one operator stage per refinement block.' >&2
  exit 2
fi
if (( HIERARCHICAL_MMDIT_OPERATOR_GROUPS < 1 || HIERARCHICAL_MMDIT_OPERATOR_RANK % HIERARCHICAL_MMDIT_OPERATOR_GROUPS != 0 )); then
  printf '%s\n' 'V82 operator rank must be divisible by operator groups.' >&2
  exit 2
fi

exec bash "${SCRIPT_DIR}/current_v76a_owned_intent_mmdit.sh" \
  --hierarchical-mmdit-depth "${HIERARCHICAL_MMDIT_DEPTH}" \
  --hierarchical-mmdit-refine-steps "${HIERARCHICAL_MMDIT_REFINE_STEPS}" \
  --hierarchical-mmdit-stage-slots "${HIERARCHICAL_MMDIT_OPERATOR_STAGES}" \
  --hierarchical-mmdit-operator-stages "${HIERARCHICAL_MMDIT_OPERATOR_STAGES}" \
  --hierarchical-mmdit-operator-rank "${HIERARCHICAL_MMDIT_OPERATOR_RANK}" \
  --hierarchical-mmdit-operator-groups "${HIERARCHICAL_MMDIT_OPERATOR_GROUPS}" \
  --hierarchical-mmdit-operator-depth-logit-init "${HIERARCHICAL_MMDIT_DEPTH_LOGIT_INIT}" \
  --hierarchical-mmdit-operator-contraction-warmup-steps "${HIERARCHICAL_MMDIT_CONTRACTION_WARMUP_STEPS}" \
  --hierarchical-mmdit-operator-contraction-transition-steps "${HIERARCHICAL_MMDIT_CONTRACTION_TRANSITION_STEPS}" \
  --hierarchical-mmdit-residual-scale-init "${HIERARCHICAL_MMDIT_RESIDUAL_SCALE_INIT}" \
  --hierarchical-mmdit-residual-scale-max "${HIERARCHICAL_MMDIT_RESIDUAL_SCALE_MAX}" \
  --hierarchical-mmdit-depth-usage-loss-weight "${HIERARCHICAL_MMDIT_DEPTH_USAGE_LOSS_WEIGHT}" \
  --hierarchical-mmdit-contraction-lr-scale "${HIERARCHICAL_MMDIT_CONTRACTION_LR_SCALE}" \
  --hierarchical-mmdit-schedule-mode fixed \
  --hierarchical-mmdit-random-prefix-probability 0 \
  --hierarchical-mmdit-exhaustion-mode off \
  "$@"
