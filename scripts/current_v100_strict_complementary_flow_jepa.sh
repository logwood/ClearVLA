#!/usr/bin/env bash
# V100: strict grounding -> world -> policy ownership with complementary RGB detail.
#
# This is still one end-to-end stage.  Relative to V99 it changes only the four
# agreed structural items:
#   1. every scratch top block/shared adapter uses the base policy LR;
#   2. policy blocks and the final decoder cannot re-read raw visual memory;
#   3. pooled low-frequency content + flow-addressed high-frequency residual
#      are additive, never competing lanes in a router;
#   4. future DINO change receives a continuous change-weighted objective.
# Static flow is constrained only to be no worse than identity.  Gripper and
# native execution-controller objectives are inherited unchanged from V99/V96.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export OUT_DIR="${OUT_DIR:-runs/v100_strict_complementary_flow_jepa}"
export V100_BATCH_SIZE="${V100_BATCH_SIZE:-${V99_BATCH_SIZE:-8}}"
export V99_BATCH_SIZE="${V100_BATCH_SIZE}"
export FLOW_JEPA_IDENTITY_ADVANTAGE_WEIGHT="${FLOW_JEPA_IDENTITY_ADVANTAGE_WEIGHT:-0.02}"
export FLOW_JEPA_STATIC_IDENTITY_WEIGHT="${FLOW_JEPA_STATIC_IDENTITY_WEIGHT:-0.01}"
export FLOW_JEPA_FUTURE_CHANGE_WEIGHT="${FLOW_JEPA_FUTURE_CHANGE_WEIGHT:-0.02}"
export FLOW_JEPA_PARENT_VERSION="${FLOW_JEPA_PARENT_VERSION:-v100}"

if [[ "${FLOW_JEPA_PARENT_VERSION}" == "v100" ]]; then
  printf '[v100] stage=single_end_to_end top=strict_complementary_raw_flow_jepa role_blocks=3/3/2 visual_path=grounding_to_world_to_policy raw_fusion=latest_dino_plus_lowpass_plus_flow_highpass_fixed lr=uniform_top change_weight=%s moving_identity_weight=%s static_identity_weight=%s batch=%s gripper=unchanged execution=unchanged stage1_init=off\n' \
    "${FLOW_JEPA_FUTURE_CHANGE_WEIGHT}" \
    "${FLOW_JEPA_IDENTITY_ADVANTAGE_WEIGHT}" \
    "${FLOW_JEPA_STATIC_IDENTITY_WEIGHT}" \
    "${V100_BATCH_SIZE}"
fi

exec bash "${SCRIPT_DIR}/current_v99_observable_raw_flow_332_jepa.sh" \
  "$@" \
  --batch-size "${V100_BATCH_SIZE}" \
  --single-stage-role-lr 1 \
  --flow-jepa-strict-role-visual-path 1 \
  --flow-jepa-complementary-raw-detail 1 \
  --flow-jepa-future-change-loss-weight "${FLOW_JEPA_FUTURE_CHANGE_WEIGHT}" \
  --flow-jepa-identity-advantage-loss-weight "${FLOW_JEPA_IDENTITY_ADVANTAGE_WEIGHT}" \
  --flow-jepa-static-identity-loss-weight "${FLOW_JEPA_STATIC_IDENTITY_WEIGHT}"
