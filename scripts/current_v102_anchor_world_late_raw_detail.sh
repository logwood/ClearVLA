#!/usr/bin/env bash
# V102: anchor/camera world organization with a late, horizon-aligned raw read.
#
# This remains one end-to-end training stage and inherits V101's balanced
# sampling, action-horizon losses, action history, T5 goal tokens, JEPA targets,
# and native action decoder.  It changes only the ownership boundary:
#   1. world blocks may write per anchor/camera, never per xy cell;
#   2. high-frequency raw detail is compiled from observation/motion only;
#   3. final-world + per-horizon policy queries read each camera's detail chart;
#   4. T*basis policy tokens are lifted per basis and pooled inside each time
#      step, replacing flattened interpolation that corrupted event timing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export OUT_DIR="${OUT_DIR:-runs/v102_anchor_world_late_raw_detail}"
export V102_BATCH_SIZE="${V102_BATCH_SIZE:-8}"
export V101_BATCH_SIZE="${V102_BATCH_SIZE}"
export FLOW_JEPA_PARENT_VERSION=v102
export FLOW_JEPA_LATE_DETAIL_SCALE="${FLOW_JEPA_LATE_DETAIL_SCALE:-0.25}"

printf '[v102] stage=single_end_to_end top=anchor_world_late_raw_detail role_blocks=3/3/2 world_write=anchor_camera raw_compile=observation_only late_read=per_camera_horizon_basis detail_scale=%s policy_workspace=horizon_pool sampling=information_balanced history=tokens goal=t5 batch=%s stage1_init=off\n' \
  "${FLOW_JEPA_LATE_DETAIL_SCALE}" \
  "${V102_BATCH_SIZE}"

exec bash "${SCRIPT_DIR}/current_v101_information_balanced_long_horizon.sh" \
  "$@" \
  --batch-size "${V102_BATCH_SIZE}" \
  --flow-jepa-world-anchor-write-only 1 \
  --flow-jepa-late-policy-detail 1 \
  --flow-jepa-late-policy-detail-scale "${FLOW_JEPA_LATE_DETAIL_SCALE}" \
  --flow-jepa-policy-workspace-horizon-pool 1
