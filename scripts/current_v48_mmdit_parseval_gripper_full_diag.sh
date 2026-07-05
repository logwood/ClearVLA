#!/usr/bin/env bash
# Two semantic output heads, one shared CVAE/MMDiT flow. The gripper is written
# in a causal local Parseval temporal frame and reconstructed from every field
# channel; no handcrafted event/delta/future coordinates or smoothing prior.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUT_DIR="${OUT_DIR:-runs/v48_mmdit_parseval_gripper_full_diag_b8}"

exec bash "${SCRIPT_DIR}/current_v48_mmdit_rollout_isolated_full_diag.sh" \
  --gripper-field-mode parseval_temporal \
  --gripper-field-dim "${GRIPPER_FIELD_DIM:-6}" \
  --latent-cvae-legacy-anchor-weight 0.0 \
  --latent-cvae-legacy-anchor-min-weight 0.0 \
  --gripper-fm-event-boost "${GRIPPER_FM_EVENT_BOOST:-0.0}" \
  --gripper-transition-l1-weight "${GRIPPER_TRANSITION_L1_WEIGHT:-0.0}" \
  --transition-gripper-flow-weight "${TRANSITION_GRIPPER_FLOW_WEIGHT:-0.0}" \
  --event-delta-consistency-weight "${EVENT_DELTA_CONSISTENCY_WEIGHT:-0.0}" \
  --event-magnitude-weight "${EVENT_MAGNITUDE_WEIGHT:-0.0}" \
  --event-off-delta-weight "${EVENT_OFF_DELTA_WEIGHT:-0.0}" \
  "$@"
