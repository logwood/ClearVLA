#!/usr/bin/env bash
# =============================================================================
# V80: zero-disturbance stage LoRA with binary function-channel availability.
#
# Three refinement ownership blocks share the full-rank MMDiT function.  Each
# block selects locally between two of six stage-memory-owned LoRA banks.  LoRA
# content starts at exactly zero while all 32 channels are available.  After a
# dense warm-up, a weak L0-style budget may close channels, but an active
# channel is never amplitude-scaled by its mask probability.  The budget never
# sees LoRA coefficients, condition values, or the independent residual scale.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HIERARCHICAL_MMDIT_DEPTH="${HIERARCHICAL_MMDIT_DEPTH:-3}"
export HIERARCHICAL_MMDIT_REFINE_STEPS="${HIERARCHICAL_MMDIT_REFINE_STEPS:-3}"
export HIERARCHICAL_MMDIT_OPERATOR_STAGES="${HIERARCHICAL_MMDIT_OPERATOR_STAGES:-6}"
export HIERARCHICAL_MMDIT_OPERATOR_RANK="${HIERARCHICAL_MMDIT_OPERATOR_RANK:-32}"
export HIERARCHICAL_MMDIT_OPERATOR_GROUPS="${HIERARCHICAL_MMDIT_OPERATOR_GROUPS:-32}"
export HIERARCHICAL_MMDIT_MASK_LOGIT_INIT="${HIERARCHICAL_MMDIT_MASK_LOGIT_INIT:-2.0}"
export HIERARCHICAL_MMDIT_MASK_THRESHOLD="${HIERARCHICAL_MMDIT_MASK_THRESHOLD:-0.5}"
export HIERARCHICAL_MMDIT_MASK_WARMUP_STEPS="${HIERARCHICAL_MMDIT_MASK_WARMUP_STEPS:-500}"
export HIERARCHICAL_MMDIT_MASK_TRANSITION_STEPS="${HIERARCHICAL_MMDIT_MASK_TRANSITION_STEPS:-1500}"
export HIERARCHICAL_MMDIT_MASK_USAGE_LOSS_WEIGHT="${HIERARCHICAL_MMDIT_MASK_USAGE_LOSS_WEIGHT:-0.0002}"
export HIERARCHICAL_MMDIT_ADAPTER_LR_SCALE="${HIERARCHICAL_MMDIT_ADAPTER_LR_SCALE:-2.0}"
export HIERARCHICAL_MMDIT_SHARED_BASE_LR_SCALE="${HIERARCHICAL_MMDIT_SHARED_BASE_LR_SCALE:-0.50}"
export HIERARCHICAL_MMDIT_RESIDUAL_LR_SCALE="${HIERARCHICAL_MMDIT_RESIDUAL_LR_SCALE:-0.25}"
export HIERARCHICAL_MMDIT_RESIDUAL_SCALE_INIT="${HIERARCHICAL_MMDIT_RESIDUAL_SCALE_INIT:-0.05}"
export HIERARCHICAL_MMDIT_RESIDUAL_SCALE_MAX="${HIERARCHICAL_MMDIT_RESIDUAL_SCALE_MAX:-0.20}"
export OUT_DIR="${OUT_DIR:-runs/v80_binary_channel_mask_lora_d${HIERARCHICAL_MMDIT_DEPTH}_s${HIERARCHICAL_MMDIT_OPERATOR_STAGES}_b8}"

if (( HIERARCHICAL_MMDIT_OPERATOR_STAGES < HIERARCHICAL_MMDIT_DEPTH )); then
  printf '%s\n' 'V80 requires at least one operator stage per refinement block.' >&2
  exit 2
fi
if (( HIERARCHICAL_MMDIT_OPERATOR_GROUPS < 1 || HIERARCHICAL_MMDIT_OPERATOR_RANK % HIERARCHICAL_MMDIT_OPERATOR_GROUPS != 0 )); then
  printf '%s\n' 'V80 operator rank must be divisible by operator groups.' >&2
  exit 2
fi

exec bash "${SCRIPT_DIR}/current_v76a_owned_intent_mmdit.sh" \
  --hierarchical-mmdit-depth "${HIERARCHICAL_MMDIT_DEPTH}" \
  --hierarchical-mmdit-refine-steps "${HIERARCHICAL_MMDIT_REFINE_STEPS}" \
  --hierarchical-mmdit-stage-slots "${HIERARCHICAL_MMDIT_OPERATOR_STAGES}" \
  --hierarchical-mmdit-operator-stages "${HIERARCHICAL_MMDIT_OPERATOR_STAGES}" \
  --hierarchical-mmdit-operator-rank "${HIERARCHICAL_MMDIT_OPERATOR_RANK}" \
  --hierarchical-mmdit-operator-groups "${HIERARCHICAL_MMDIT_OPERATOR_GROUPS}" \
  --hierarchical-mmdit-operator-mask-logit-init "${HIERARCHICAL_MMDIT_MASK_LOGIT_INIT}" \
  --hierarchical-mmdit-operator-mask-threshold "${HIERARCHICAL_MMDIT_MASK_THRESHOLD}" \
  --hierarchical-mmdit-operator-mask-warmup-steps "${HIERARCHICAL_MMDIT_MASK_WARMUP_STEPS}" \
  --hierarchical-mmdit-operator-mask-transition-steps "${HIERARCHICAL_MMDIT_MASK_TRANSITION_STEPS}" \
  --hierarchical-mmdit-residual-scale-init "${HIERARCHICAL_MMDIT_RESIDUAL_SCALE_INIT}" \
  --hierarchical-mmdit-residual-scale-max "${HIERARCHICAL_MMDIT_RESIDUAL_SCALE_MAX}" \
  --hierarchical-mmdit-mask-usage-loss-weight "${HIERARCHICAL_MMDIT_MASK_USAGE_LOSS_WEIGHT}" \
  --hierarchical-mmdit-adapter-lr-scale "${HIERARCHICAL_MMDIT_ADAPTER_LR_SCALE}" \
  --hierarchical-mmdit-shared-base-lr-scale "${HIERARCHICAL_MMDIT_SHARED_BASE_LR_SCALE}" \
  --hierarchical-mmdit-residual-control-lr-scale "${HIERARCHICAL_MMDIT_RESIDUAL_LR_SCALE}" \
  --hierarchical-mmdit-schedule-mode fixed \
  --hierarchical-mmdit-random-prefix-probability 0 \
  --hierarchical-mmdit-exhaustion-mode off \
  "$@"
