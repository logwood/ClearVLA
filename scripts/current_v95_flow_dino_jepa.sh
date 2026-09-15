#!/usr/bin/env bash
# V95: shared launcher for representation Stage1 and action-policy Stage2.
#
# This wrapper deliberately inherits V94's already-audited native Evidence
# execution path, but replaces the top visual contract:
#
#   frozen DINO grid -> semantic patch correspondence -> typed selector/value
#   evidence -> directed joint canvas (context -> action -> future query)
#
# A single far-stage token is updated before the local window tokens in every
# DiT block.  It predicts a global frozen-DINO delta at t+48 and conditions
# sparse patch-level windows at t+4/12/24 through a dedicated residual bridge.
# Future DINO remains a no-grad teacher only; no LSTM or cross-batch state is
# introduced.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export V95_TRAINING_STAGE="${V95_TRAINING_STAGE:-stage1}"
case "${V95_TRAINING_STAGE}" in
  stage1|policy) ;;
  *)
    echo "[v95] V95_TRAINING_STAGE must be stage1 or policy, got ${V95_TRAINING_STAGE}" >&2
    exit 2
    ;;
esac
if [[ "${V95_TRAINING_STAGE}" == "stage1" ]]; then
  export OUT_DIR="${OUT_DIR:-runs/v95_flow_dino_jepa_stage1}"
  export V95_REQUIRE_STAGE1_CONTRACT=0
  export V95_STAGE1_RESET_DIRTY_ADAPTERS="${V95_STAGE1_RESET_DIRTY_ADAPTERS:-1}"
  V95_FORWARD_CONTRACT="representation_only"
  V95_TARGET_ACTION_CONDITIONED=0
  V95_FINAL_ACTION_DECODER_EXECUTED=0
  V95_LAYER_CONTRACTS_EXECUTED=0
else
  export OUT_DIR="${OUT_DIR:-runs/v95_flow_dino_jepa_policy}"
  V95_STAGE1_CHECKPOINT="${V95_STAGE1_CHECKPOINT:-}"
  if [[ -z "${V95_STAGE1_CHECKPOINT}" || ! -f "${V95_STAGE1_CHECKPOINT}" ]]; then
    echo "[v95] policy stage requires V95_STAGE1_CHECKPOINT=.../best_stage1_representation.pt" >&2
    exit 2
  fi
  export STAGE1_CHECKPOINT="${V95_STAGE1_CHECKPOINT}"
  export V95_REQUIRE_STAGE1_CONTRACT=1
  export V95_STAGE1_RESET_DIRTY_ADAPTERS=0
  V95_FORWARD_CONTRACT="action_policy"
  V95_TARGET_ACTION_CONDITIONED=1
  V95_FINAL_ACTION_DECODER_EXECUTED=1
  V95_LAYER_CONTRACTS_EXECUTED=1
fi
export LAYER_CONTRACT_AUX_WEIGHT="${LAYER_CONTRACT_AUX_WEIGHT:-0.0}"
export ROLLOUT_DYNAMICS_WEIGHT="${ROLLOUT_DYNAMICS_WEIGHT:-0.0}"
export ROLLOUT_CONTRAST_WEIGHT="${ROLLOUT_CONTRAST_WEIGHT:-0.0}"
export ROLLOUT_VARIANCE_WEIGHT="${ROLLOUT_VARIANCE_WEIGHT:-0.0}"
export ROLLOUT_NORM_WEIGHT="${ROLLOUT_NORM_WEIGHT:-0.0}"
export ROLLOUT_MILESTONE_WEIGHT="${ROLLOUT_MILESTONE_WEIGHT:-0.0}"
export FLOW_JEPA_GRID_SIZE="${FLOW_JEPA_GRID_SIZE:-8}"
export FLOW_JEPA_FEATURE_DIM="${FLOW_JEPA_FEATURE_DIM:-96}"
export FLOW_JEPA_FLOW_ITERS="${FLOW_JEPA_FLOW_ITERS:-3}"
export FLOW_JEPA_CORR_LEVELS="${FLOW_JEPA_CORR_LEVELS:-3}"
export FLOW_JEPA_CORR_RADIUS="${FLOW_JEPA_CORR_RADIUS:-2}"
export FLOW_JEPA_MASK_RATIO="${FLOW_JEPA_MASK_RATIO:-0.375}"
export FLOW_JEPA_MASK_BLOCK_SIZE="${FLOW_JEPA_MASK_BLOCK_SIZE:-2}"
export FLOW_JEPA_MOTION_MASK_FRACTION="${FLOW_JEPA_MOTION_MASK_FRACTION:-0.60}"
export FLOW_JEPA_UNCERTAINTY_FLOOR="${FLOW_JEPA_UNCERTAINTY_FLOOR:-0.03}"
export FLOW_JEPA_WINDOW_OFFSETS="${FLOW_JEPA_WINDOW_OFFSETS:-4 12 24}"
export FLOW_JEPA_STAGE_OFFSET="${FLOW_JEPA_STAGE_OFFSET:-48}"
export FLOW_JEPA_FUTURE_WEIGHT="${FLOW_JEPA_FUTURE_WEIGHT:-0.10}"
export FLOW_JEPA_STAGE_WEIGHT="${FLOW_JEPA_STAGE_WEIGHT:-0.02}"
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

python - \
  "${FLOW_JEPA_GRID_SIZE}" \
  "${FLOW_JEPA_FEATURE_DIM}" \
  "${FLOW_JEPA_FLOW_ITERS}" \
  "${FLOW_JEPA_CORR_LEVELS}" \
  "${FLOW_JEPA_CORR_RADIUS}" \
  "${FLOW_JEPA_MASK_RATIO}" \
  "${FLOW_JEPA_MASK_BLOCK_SIZE}" \
  "${FLOW_JEPA_MOTION_MASK_FRACTION}" \
  "${FLOW_JEPA_UNCERTAINTY_FLOOR}" \
  "${FLOW_JEPA_FUTURE_WEIGHT}" \
  "${FLOW_JEPA_WARP_WEIGHT}" \
  "${FLOW_JEPA_CYCLE_WEIGHT}" \
  "${FLOW_JEPA_SMOOTHNESS_WEIGHT}" \
  "${FLOW_JEPA_UNCERTAINTY_WEIGHT}" \
  "${FLOW_JEPA_SEQUENCE_WEIGHT}" \
  "${FLOW_JEPA_LR_SCALE}" \
  "${FLOW_JEPA_WINDOW_OFFSETS}" \
  "${FLOW_JEPA_STAGE_OFFSET}" \
  "${FLOW_JEPA_STAGE_WEIGHT}" \
  "${ACTION_HISTORY_OFFSETS}" \
  "${ACTION_HISTORY_RECENT_TOKENS}" \
  "${ACTION_HISTORY_SUMMARY_TOKENS}" \
  "${GOAL_TOKEN_COUNT}" \
  "${GOAL_RESAMPLER_DEPTH}" \
  "${GOAL_LANGUAGE_MAX_TOKENS}" \
  "${T5_CONDITION_PATH}" <<'PY'
import math
import sys
from pathlib import Path

grid, feature_dim, iterations, levels, radius = map(int, sys.argv[1:6])
mask_ratio = float(sys.argv[6])
block = int(sys.argv[7])
motion_fraction = float(sys.argv[8])
uncertainty_floor = float(sys.argv[9])
weights = tuple(float(value) for value in sys.argv[10:16])
lr_scale = float(sys.argv[16])
window_offsets = tuple(int(value) for value in sys.argv[17].replace(",", " ").split())
stage_offset = int(sys.argv[18])
stage_weight = float(sys.argv[19])
action_offsets = tuple(int(value) for value in sys.argv[20].replace(",", " ").split())
action_recent = int(sys.argv[21])
action_summary = int(sys.argv[22])
goal_tokens = int(sys.argv[23])
goal_depth = int(sys.argv[24])
goal_max_tokens = int(sys.argv[25])
t5_condition_text = sys.argv[26].strip()
t5_condition = Path(t5_condition_text).expanduser()
if min(grid, feature_dim, iterations, levels, radius, block) < 1:
    raise SystemExit("[v95] grid/dim/iterations/correlation/block values must be positive")
if feature_dim % 8:
    raise SystemExit(f"[v95] flow feature dim must be divisible by 8, got {feature_dim}")
if grid < 2 ** (levels - 1):
    raise SystemExit(
        f"[v95] grid={grid} is too small for {levels} correlation levels"
    )
if not 0.0 < mask_ratio < 1.0:
    raise SystemExit(f"[v95] mask ratio must be in (0,1), got {mask_ratio}")
if not 0.0 <= motion_fraction <= 1.0:
    raise SystemExit(
        f"[v95] motion mask fraction must be in [0,1], got {motion_fraction}"
    )
if not math.isfinite(uncertainty_floor) or uncertainty_floor <= 0.0:
    raise SystemExit("[v95] uncertainty floor must be finite and positive")
if any(not math.isfinite(value) or value < 0.0 for value in weights):
    raise SystemExit(f"[v95] representation loss weights must be finite/non-negative: {weights}")
if weights[0] <= 0.0 or weights[1] <= 0.0:
    raise SystemExit("[v95] future prediction and final warp losses must remain active")
if len(window_offsets) != 3 or tuple(sorted(set(window_offsets))) != window_offsets:
    raise SystemExit("[v95] window offsets must be three strictly increasing values")
if window_offsets[0] <= 0 or window_offsets[-1] != 24:
    raise SystemExit("[v95] window offsets must be positive and end at policy horizon 24")
if stage_offset <= window_offsets[-1]:
    raise SystemExit("[v95] stage offset must be later than the window")
if not math.isfinite(stage_weight) or stage_weight <= 0.0:
    raise SystemExit("[v95] stage loss weight must be finite and positive")
if not math.isfinite(lr_scale) or lr_scale <= 0.0:
    raise SystemExit(f"[v95] Flow-DINO LR scale must be finite and positive, got {lr_scale}")
if len(action_offsets) < 2 or tuple(sorted(set(action_offsets))) != action_offsets:
    raise SystemExit("[v95] action history offsets must be strictly increasing")
if action_offsets[-1] >= 0:
    raise SystemExit("[v95] action history offsets must all precede the current frame")
if not 1 <= action_recent <= len(action_offsets):
    raise SystemExit("[v95] recent action token count must fit action history")
if action_summary < 1:
    raise SystemExit("[v95] action history summary token count must be positive")
if min(goal_tokens, goal_depth, goal_max_tokens) < 1:
    raise SystemExit("[v95] goal token/depth/max-token values must be positive")
if not t5_condition_text:
    raise SystemExit("[v95] T5_CONDITION_PATH must point to a precomputed .pt")
if t5_condition.suffix.lower() not in {".pt", ".pth"}:
    raise SystemExit(f"[v95] T5 condition must be .pt/.pth, got {t5_condition}")
if not t5_condition.is_file():
    raise SystemExit(f"[v95] T5 condition file does not exist: {t5_condition}")
PY

printf '[v95] experiment_stage=%s forward_contract=%s target_action_conditioned=%s final_action_decoder_executed=%s layer_contracts_executed=%s top=hierarchical_flow_dino_jepa flow=sea_raft_patch window_offsets=%s stage_offset=%s future_grid=%s flow_dim=%s flow_iters=%s corr_levels=%s corr_radius=%s mask_ratio=%s block=%s stage_then_window=1 stage_recurrent_state=0 future_teacher=frozen_dino_no_grad directed_canvas=1 old_rollout_objectives=0 action_history=%s action_memory=%s+%s goal_tokens=%s language_condition=precomputed_t5_pt t5_condition=%s\n' \
  "${V95_TRAINING_STAGE}" \
  "${V95_FORWARD_CONTRACT}" \
  "${V95_TARGET_ACTION_CONDITIONED}" \
  "${V95_FINAL_ACTION_DECODER_EXECUTED}" \
  "${V95_LAYER_CONTRACTS_EXECUTED}" \
  "${FLOW_JEPA_WINDOW_OFFSETS}" \
  "${FLOW_JEPA_STAGE_OFFSET}" \
  "${FLOW_JEPA_GRID_SIZE}" \
  "${FLOW_JEPA_FEATURE_DIM}" \
  "${FLOW_JEPA_FLOW_ITERS}" \
  "${FLOW_JEPA_CORR_LEVELS}" \
  "${FLOW_JEPA_CORR_RADIUS}" \
  "${FLOW_JEPA_MASK_RATIO}" \
  "${FLOW_JEPA_MASK_BLOCK_SIZE}" \
  "${ACTION_HISTORY_OFFSETS}" \
  "${ACTION_HISTORY_RECENT_TOKENS}" \
  "${ACTION_HISTORY_SUMMARY_TOKENS}" \
  "${GOAL_TOKEN_COUNT}" \
  "${T5_CONDITION_PATH}"

exec bash "${SCRIPT_DIR}/current_v94_latent_ownership_execution.sh" \
  "$@" \
  --training-stage "${V95_TRAINING_STAGE}" \
  --require-flow-jepa-stage1-checkpoint "${V95_REQUIRE_STAGE1_CONTRACT}" \
  --stage1-reset-dirty-adapters "${V95_STAGE1_RESET_DIRTY_ADAPTERS}" \
  --future-anchors 3 \
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
  --flow-jepa-grid-size "${FLOW_JEPA_GRID_SIZE}" \
  --flow-jepa-feature-dim "${FLOW_JEPA_FEATURE_DIM}" \
  --flow-jepa-flow-iters "${FLOW_JEPA_FLOW_ITERS}" \
  --flow-jepa-corr-levels "${FLOW_JEPA_CORR_LEVELS}" \
  --flow-jepa-corr-radius "${FLOW_JEPA_CORR_RADIUS}" \
  --flow-jepa-mask-ratio "${FLOW_JEPA_MASK_RATIO}" \
  --flow-jepa-mask-block-size "${FLOW_JEPA_MASK_BLOCK_SIZE}" \
  --flow-jepa-motion-mask-fraction "${FLOW_JEPA_MOTION_MASK_FRACTION}" \
  --flow-jepa-uncertainty-floor "${FLOW_JEPA_UNCERTAINTY_FLOOR}" \
  --flow-jepa-window-offsets "${FLOW_JEPA_WINDOW_OFFSETS}" \
  --flow-jepa-stage-offset "${FLOW_JEPA_STAGE_OFFSET}" \
  --flow-jepa-directed-canvas-attention 1 \
  --flow-jepa-future-loss-weight "${FLOW_JEPA_FUTURE_WEIGHT}" \
  --flow-jepa-stage-loss-weight "${FLOW_JEPA_STAGE_WEIGHT}" \
  --flow-jepa-warp-loss-weight "${FLOW_JEPA_WARP_WEIGHT}" \
  --flow-jepa-cycle-loss-weight "${FLOW_JEPA_CYCLE_WEIGHT}" \
  --flow-jepa-smoothness-loss-weight "${FLOW_JEPA_SMOOTHNESS_WEIGHT}" \
  --flow-jepa-uncertainty-nll-weight "${FLOW_JEPA_UNCERTAINTY_WEIGHT}" \
  --flow-jepa-refinement-sequence-loss-weight "${FLOW_JEPA_SEQUENCE_WEIGHT}" \
  --flow-jepa-lr-scale "${FLOW_JEPA_LR_SCALE}"
