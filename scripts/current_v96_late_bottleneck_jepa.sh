#!/usr/bin/env bash
# V96: single-stage coarse-to-fine Flow-addressed multi-horizon JEPA policy.
#
# Full-image correspondence stays on a bounded coarse chart.  Native DINO
# patches are consulted only by a differentiable late local reader whose fine
# residual is controlled by motion/confidence/uncertainty and JEPA mask hints.
# The far horizon is a spatial future-evidence chart, not a global stage token.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export OUT_DIR="${OUT_DIR:-runs/v96_late_bottleneck_jepa}"
export FLOW_JEPA_GRID_SIZE="${FLOW_JEPA_GRID_SIZE:-8}"
export FLOW_JEPA_FEATURE_DIM="${FLOW_JEPA_FEATURE_DIM:-96}"
export FLOW_JEPA_FLOW_ITERS="${FLOW_JEPA_FLOW_ITERS:-3}"
export FLOW_JEPA_CORR_LEVELS="${FLOW_JEPA_CORR_LEVELS:-3}"
export FLOW_JEPA_CORR_RADIUS="${FLOW_JEPA_CORR_RADIUS:-2}"
export FLOW_JEPA_DENSE_DEPTH="${FLOW_JEPA_DENSE_DEPTH:-2}"
export FLOW_JEPA_FINE_RADIUS="${FLOW_JEPA_FINE_RADIUS:-2}"
export FLOW_JEPA_READER_RADIUS="${FLOW_JEPA_READER_RADIUS:-1}"
export FLOW_JEPA_READER_HEADS="${FLOW_JEPA_READER_HEADS:-2}"
export FLOW_JEPA_MASK_RATIO="${FLOW_JEPA_MASK_RATIO:-0.375}"
export FLOW_JEPA_MASK_BLOCK_SIZE="${FLOW_JEPA_MASK_BLOCK_SIZE:-2}"
export FLOW_JEPA_MOTION_MASK_FRACTION="${FLOW_JEPA_MOTION_MASK_FRACTION:-0.60}"
export FLOW_JEPA_UNCERTAINTY_FLOOR="${FLOW_JEPA_UNCERTAINTY_FLOOR:-0.03}"
export FLOW_JEPA_HORIZONS="${FLOW_JEPA_HORIZONS:-4 12 24 48}"
export FLOW_JEPA_FUTURE_WEIGHT="${FLOW_JEPA_FUTURE_WEIGHT:-0.10}"
export FLOW_JEPA_WARP_WEIGHT="${FLOW_JEPA_WARP_WEIGHT:-0.03}"
export FLOW_JEPA_CYCLE_WEIGHT="${FLOW_JEPA_CYCLE_WEIGHT:-0.01}"
export FLOW_JEPA_SMOOTHNESS_WEIGHT="${FLOW_JEPA_SMOOTHNESS_WEIGHT:-0.002}"
export FLOW_JEPA_UNCERTAINTY_WEIGHT="${FLOW_JEPA_UNCERTAINTY_WEIGHT:-0.005}"
export FLOW_JEPA_SEQUENCE_WEIGHT="${FLOW_JEPA_SEQUENCE_WEIGHT:-0.02}"
export FLOW_JEPA_LR_SCALE="${FLOW_JEPA_LR_SCALE:-1.0}"
export ACTION_HISTORY_OFFSETS="${ACTION_HISTORY_OFFSETS:--24 -16 -12 -8 -6 -4 -2 -1}"
export ACTION_HISTORY_RECENT_TOKENS="${ACTION_HISTORY_RECENT_TOKENS:-4}"
export ACTION_HISTORY_SUMMARY_TOKENS="${ACTION_HISTORY_SUMMARY_TOKENS:-3}"
export GOAL_TOKEN_COUNT="${GOAL_TOKEN_COUNT:-4}"
export GOAL_RESAMPLER_DEPTH="${GOAL_RESAMPLER_DEPTH:-2}"
export GOAL_LANGUAGE_MAX_TOKENS="${GOAL_LANGUAGE_MAX_TOKENS:-32}"
export T5_CONDITION_PATH="${T5_CONDITION_PATH:-}"

FLOW_JEPA_LAUNCH_VERSION="${FLOW_JEPA_PARENT_VERSION:-v96}"
if [[ -z "${T5_CONDITION_PATH}" || ! -f "${T5_CONDITION_PATH}" ]]; then
  echo "[${FLOW_JEPA_LAUNCH_VERSION}] T5_CONDITION_PATH must point to the precomputed T5 .pt/.pth" >&2
  exit 2
fi
if [[ " ${FLOW_JEPA_HORIZONS} " != " 4 12 24 48 " ]]; then
  echo "[${FLOW_JEPA_LAUNCH_VERSION}] default contract requires FLOW_JEPA_HORIZONS='4 12 24 48'" >&2
  exit 2
fi

# The inherited V94 script appends these weights after user arguments.  Keep
# obsolete rollout surrogates off so JEPA future evidence and the action loss
# supervise one path without duplicate rollout targets.
export LAYER_CONTRACT_AUX_WEIGHT="${LAYER_CONTRACT_AUX_WEIGHT:-0.0}"
export ROLLOUT_DYNAMICS_WEIGHT="${ROLLOUT_DYNAMICS_WEIGHT:-0.0}"
export ROLLOUT_CONTRAST_WEIGHT="${ROLLOUT_CONTRAST_WEIGHT:-0.0}"
export ROLLOUT_VARIANCE_WEIGHT="${ROLLOUT_VARIANCE_WEIGHT:-0.0}"
export ROLLOUT_NORM_WEIGHT="${ROLLOUT_NORM_WEIGHT:-0.0}"
export ROLLOUT_MILESTONE_WEIGHT="${ROLLOUT_MILESTONE_WEIGHT:-0.0}"

if [[ -z "${FLOW_JEPA_PARENT_VERSION:-}" ]]; then
  printf '[v96] stage=single_end_to_end top=coarse_to_fine_flow_dino_jepa horizons=%s coarse_grid=%s native_grid=from_dino fine_radius=%s reader_radius=%s reader_heads=%s stage_token=0 flow_hard_selection=0 action_anchors=4/12/24 bottom=evidence_mmdit_native_execution language=precomputed_t5_pt\n' \
    "${FLOW_JEPA_HORIZONS}" \
    "${FLOW_JEPA_GRID_SIZE}" \
    "${FLOW_JEPA_FINE_RADIUS}" \
    "${FLOW_JEPA_READER_RADIUS}" \
    "${FLOW_JEPA_READER_HEADS}"
fi

exec bash "${SCRIPT_DIR}/current_v94_latent_ownership_execution.sh" \
  "$@" \
  --training-stage policy \
  --require-flow-jepa-stage1-checkpoint 0 \
  --stage1-reset-dirty-adapters 0 \
  --future-anchors 4 \
  --layer-consequence-steps 3 \
  --future-grid-size "${FLOW_JEPA_GRID_SIZE}" \
  --executed-action-offsets "${ACTION_HISTORY_OFFSETS}" \
  --action-history-enabled 1 \
  --action-history-recent-tokens "${ACTION_HISTORY_RECENT_TOKENS}" \
  --action-history-summary-tokens "${ACTION_HISTORY_SUMMARY_TOKENS}" \
  --goal-conditioning-enabled 1 \
  --goal-token-count "${GOAL_TOKEN_COUNT}" \
  --goal-resampler-depth "${GOAL_RESAMPLER_DEPTH}" \
  --goal-language-max-tokens "${GOAL_LANGUAGE_MAX_TOKENS}" \
  --t5-condition-path "${T5_CONDITION_PATH}" \
  --flow-jepa-enabled 1 \
  --flow-jepa-late-bottleneck 1 \
  --flow-jepa-grid-size "${FLOW_JEPA_GRID_SIZE}" \
  --flow-jepa-feature-dim "${FLOW_JEPA_FEATURE_DIM}" \
  --flow-jepa-flow-iters "${FLOW_JEPA_FLOW_ITERS}" \
  --flow-jepa-corr-levels "${FLOW_JEPA_CORR_LEVELS}" \
  --flow-jepa-corr-radius "${FLOW_JEPA_CORR_RADIUS}" \
  --flow-jepa-dense-depth "${FLOW_JEPA_DENSE_DEPTH}" \
  --flow-jepa-fine-radius "${FLOW_JEPA_FINE_RADIUS}" \
  --flow-jepa-reader-radius "${FLOW_JEPA_READER_RADIUS}" \
  --flow-jepa-reader-heads "${FLOW_JEPA_READER_HEADS}" \
  --flow-jepa-mask-ratio "${FLOW_JEPA_MASK_RATIO}" \
  --flow-jepa-mask-block-size "${FLOW_JEPA_MASK_BLOCK_SIZE}" \
  --flow-jepa-motion-mask-fraction "${FLOW_JEPA_MOTION_MASK_FRACTION}" \
  --flow-jepa-uncertainty-floor "${FLOW_JEPA_UNCERTAINTY_FLOOR}" \
  --flow-jepa-window-offsets "${FLOW_JEPA_HORIZONS}" \
  --flow-jepa-stage-offset 0 \
  --flow-jepa-directed-canvas-attention 1 \
  --flow-jepa-future-loss-weight "${FLOW_JEPA_FUTURE_WEIGHT}" \
  --flow-jepa-stage-loss-weight 0 \
  --flow-jepa-warp-loss-weight "${FLOW_JEPA_WARP_WEIGHT}" \
  --flow-jepa-cycle-loss-weight "${FLOW_JEPA_CYCLE_WEIGHT}" \
  --flow-jepa-smoothness-loss-weight "${FLOW_JEPA_SMOOTHNESS_WEIGHT}" \
  --flow-jepa-uncertainty-nll-weight "${FLOW_JEPA_UNCERTAINTY_WEIGHT}" \
  --flow-jepa-refinement-sequence-loss-weight "${FLOW_JEPA_SEQUENCE_WEIGHT}" \
  --flow-jepa-lr-scale "${FLOW_JEPA_LR_SCALE}"
