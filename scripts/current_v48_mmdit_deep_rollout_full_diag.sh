#!/usr/bin/env bash
# Phase-1 rollout integration: final dynamics consumes the fully refined rollout
# canvas, MMDiT reads the complete detached rollout token grid, and the main
# rollout is calibrated for variance, norm, and inter-milestone change.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUT_DIR="${OUT_DIR:-runs/v64_mmdit_deep_rollout_parseval_b8}"

exec bash "${SCRIPT_DIR}/current_v48_mmdit_parseval_gripper_full_diag.sh" \
  --rollout-variance-loss-weight "${ROLLOUT_VARIANCE_LOSS_WEIGHT:-0.05}" \
  --rollout-norm-loss-weight "${ROLLOUT_NORM_LOSS_WEIGHT:-0.02}" \
  --rollout-milestone-delta-match-weight "${ROLLOUT_MILESTONE_DELTA_MATCH_WEIGHT:-0.15}" \
  "$@"
