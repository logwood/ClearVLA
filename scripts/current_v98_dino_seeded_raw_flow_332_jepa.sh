#!/usr/bin/env bash
# V98: one-stage DINO-seeded raw-flow + multi-horizon JEPA policy.
#
# Eight top DiT blocks form one serial path:
#   1-3 grounding: organize cached DINO semantics and identity-safe coarse flow;
#   4-6 world: read the late 1/4-resolution raw detail chart and predict the
#              multi-horizon future evidence;
#   7-8 policy: read the grounded world chart and write only the action canvas.
#
# DINO supplies the global address. Raw RGB only learns continuous 1/8 and 1/4
# residuals, so the 84x84 detail chart is preserved without a 21x21 raw global
# all-pairs graph. The reader router changes address precision, never raw value
# amplitude. There is no top-k route, detach, bypass, or hard compute gate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export OUT_DIR="${OUT_DIR:-runs/v98_dino_seeded_raw_flow_332_jepa}"
export V98_BATCH_SIZE="${V98_BATCH_SIZE:-${V97_BATCH_SIZE:-4}}"
export FLOW_JEPA_RAW_BASE_CHANNELS="${FLOW_JEPA_RAW_BASE_CHANNELS:-32}"
export FLOW_JEPA_RAW_MID_RADIUS="${FLOW_JEPA_RAW_MID_RADIUS:-2}"
export FLOW_JEPA_RAW_HIGH_RADIUS="${FLOW_JEPA_RAW_HIGH_RADIUS:-1}"
export FLOW_JEPA_RAW_READER_RADIUS="${FLOW_JEPA_RAW_READER_RADIUS:-3}"
export FLOW_JEPA_RAW_READER_HEADS="${FLOW_JEPA_RAW_READER_HEADS:-4}"
export FLOW_JEPA_RAW_ACTIVATION_CHECKPOINT="${FLOW_JEPA_RAW_ACTIVATION_CHECKPOINT:-1}"
export FLOW_JEPA_POLICY_WORKSPACE_SCALE="${FLOW_JEPA_POLICY_WORKSPACE_SCALE:-0.10}"
export FLOW_JEPA_PARENT_VERSION=v98

printf '[v98] stage=single_end_to_end top=dino_seeded_raw_flow_jepa role_blocks=3/3/2 raw_pyramid=1/4,1/8 raw_base=%s raw_reader_radius=%s raw_reader_heads=%s flow_address=continuous raw_value_gate=off activation_checkpoint=%s bottom=evidence_mmdit_native_execution batch=%s stage1_init=off\n' \
  "${FLOW_JEPA_RAW_BASE_CHANNELS}" \
  "${FLOW_JEPA_RAW_READER_RADIUS}" \
  "${FLOW_JEPA_RAW_READER_HEADS}" \
  "${FLOW_JEPA_RAW_ACTIVATION_CHECKPOINT}" \
  "${V98_BATCH_SIZE}"

exec bash "${SCRIPT_DIR}/current_v96_late_bottleneck_jepa.sh" \
  "$@" \
  --batch-size "${V98_BATCH_SIZE}" \
  --stage1-initialization-enabled 0 \
  --require-flow-jepa-stage1-checkpoint 0 \
  --depth 8 \
  --midcut-layer 6 \
  --flow-jepa-raw-image-enabled 1 \
  --flow-jepa-role-hierarchy 1 \
  --flow-jepa-raw-base-channels "${FLOW_JEPA_RAW_BASE_CHANNELS}" \
  --flow-jepa-raw-mid-radius "${FLOW_JEPA_RAW_MID_RADIUS}" \
  --flow-jepa-raw-high-radius "${FLOW_JEPA_RAW_HIGH_RADIUS}" \
  --flow-jepa-raw-reader-radius "${FLOW_JEPA_RAW_READER_RADIUS}" \
  --flow-jepa-raw-reader-heads "${FLOW_JEPA_RAW_READER_HEADS}" \
  --flow-jepa-raw-activation-checkpoint "${FLOW_JEPA_RAW_ACTIVATION_CHECKPOINT}" \
  --flow-jepa-grounding-blocks 3 \
  --flow-jepa-world-blocks 3 \
  --flow-jepa-policy-blocks 2 \
  --flow-jepa-policy-workspace-scale "${FLOW_JEPA_POLICY_WORKSPACE_SCALE}"
