#!/usr/bin/env bash
# V99: observable-motion raw flow with an honest identity baseline.
#
# This keeps V98's 3/3/2 single-stage topology and 84x84 raw value chart, but
# removes the zero-flow shortcut:
#   * fixed RGB/census evidence supervises warp and defines motion balancing;
#   * uncertain DINO flow widens local search instead of shrinking to identity;
#   * moving pixels must beat the exact zero-flow warp baseline;
#   * the raw reader compares a flow-centred detail read with a pooled content
#     fallback rather than two coordinate-identical local banks.
# Static pixels are never forced to move and raw value amplitude is not gated.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export OUT_DIR="${OUT_DIR:-runs/v99_observable_raw_flow_332_jepa}"
export V99_BATCH_SIZE="${V99_BATCH_SIZE:-${V98_BATCH_SIZE:-8}}"
export FLOW_JEPA_RAW_BASE_CHANNELS="${FLOW_JEPA_RAW_BASE_CHANNELS:-32}"
export FLOW_JEPA_RAW_MID_RADIUS="${FLOW_JEPA_RAW_MID_RADIUS:-2}"
export FLOW_JEPA_RAW_HIGH_RADIUS="${FLOW_JEPA_RAW_HIGH_RADIUS:-1}"
export FLOW_JEPA_RAW_READER_RADIUS="${FLOW_JEPA_RAW_READER_RADIUS:-3}"
export FLOW_JEPA_RAW_READER_HEADS="${FLOW_JEPA_RAW_READER_HEADS:-4}"
export FLOW_JEPA_RAW_ACTIVATION_CHECKPOINT="${FLOW_JEPA_RAW_ACTIVATION_CHECKPOINT:-1}"
export FLOW_JEPA_GROUNDING_BLOCKS="${FLOW_JEPA_GROUNDING_BLOCKS:-3}"
export FLOW_JEPA_WORLD_BLOCKS="${FLOW_JEPA_WORLD_BLOCKS:-3}"
export FLOW_JEPA_POLICY_BLOCKS="${FLOW_JEPA_POLICY_BLOCKS:-2}"
export FLOW_JEPA_POLICY_WORKSPACE_SCALE="${FLOW_JEPA_POLICY_WORKSPACE_SCALE:-0.10}"
export FLOW_JEPA_IDENTITY_ADVANTAGE_WEIGHT="${FLOW_JEPA_IDENTITY_ADVANTAGE_WEIGHT:-0.02}"
export T5_CONDITION_PATH="${T5_CONDITION_PATH:-/home/sen.wang/workspace/robotics/clear/data/grasp_pen_embed.pt}"
export FLOW_JEPA_PARENT_VERSION="${FLOW_JEPA_PARENT_VERSION:-v99}"

if [[ "${FLOW_JEPA_PARENT_VERSION}" == "v99" ]]; then
  printf '[v99] stage=single_end_to_end top=observable_raw_flow_jepa role_blocks=3/3/2 raw_pyramid=1/4,1/8 raw_base=%s raw_reader_radius=%s raw_reader_heads=%s flow_address=continuous fallback=pooled_content motion_evidence=fixed_rgb_census identity_adv_weight=%s activation_checkpoint=%s bottom=evidence_mmdit_native_execution batch=%s stage1_init=off\n' \
    "${FLOW_JEPA_RAW_BASE_CHANNELS}" \
    "${FLOW_JEPA_RAW_READER_RADIUS}" \
    "${FLOW_JEPA_RAW_READER_HEADS}" \
    "${FLOW_JEPA_IDENTITY_ADVANTAGE_WEIGHT}" \
    "${FLOW_JEPA_RAW_ACTIVATION_CHECKPOINT}" \
    "${V99_BATCH_SIZE}"
fi

exec bash "${SCRIPT_DIR}/current_v96_late_bottleneck_jepa.sh" \
  "$@" \
  --batch-size "${V99_BATCH_SIZE}" \
  --stage1-initialization-enabled 0 \
  --require-flow-jepa-stage1-checkpoint 0 \
  --depth 8 \
  --midcut-layer 6 \
  --flow-jepa-raw-image-enabled 1 \
  --flow-jepa-role-hierarchy 1 \
  --flow-jepa-zero-flow-guard 1 \
  --flow-jepa-identity-advantage-loss-weight "${FLOW_JEPA_IDENTITY_ADVANTAGE_WEIGHT}" \
  --flow-jepa-raw-base-channels "${FLOW_JEPA_RAW_BASE_CHANNELS}" \
  --flow-jepa-raw-mid-radius "${FLOW_JEPA_RAW_MID_RADIUS}" \
  --flow-jepa-raw-high-radius "${FLOW_JEPA_RAW_HIGH_RADIUS}" \
  --flow-jepa-raw-reader-radius "${FLOW_JEPA_RAW_READER_RADIUS}" \
  --flow-jepa-raw-reader-heads "${FLOW_JEPA_RAW_READER_HEADS}" \
  --flow-jepa-raw-activation-checkpoint "${FLOW_JEPA_RAW_ACTIVATION_CHECKPOINT}" \
  --flow-jepa-grounding-blocks "${FLOW_JEPA_GROUNDING_BLOCKS}" \
  --flow-jepa-world-blocks "${FLOW_JEPA_WORLD_BLOCKS}" \
  --flow-jepa-policy-blocks "${FLOW_JEPA_POLICY_BLOCKS}" \
  --flow-jepa-policy-workspace-scale "${FLOW_JEPA_POLICY_WORKSPACE_SCALE}"
