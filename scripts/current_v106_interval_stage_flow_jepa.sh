#!/usr/bin/env bash
# V106: spatially aligned interval-stage increments with variance-safe routing.
#
# The four online W charts retain camera/8x8 ownership.  At the W->P boundary
# a causal per-cell organizer reads only the chronological horizon axis and
# writes a bounded coarse delta.  Frozen teachers use real observations from
# [4,8], [8,16], [16,32], and [32,48] to expose robust interval content,
# signed least-squares progression, and endpoint displacement.  No future
# teacher enters the forward path and the organizer never rewrites the
# protected high-resolution address/value bank.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export OUT_DIR="${OUT_DIR:-runs/v106_interval_stage_flow_jepa}"
export V106_BATCH_SIZE="${V106_BATCH_SIZE:-8}"
export V105_BATCH_SIZE="${V106_BATCH_SIZE}"
export FLOW_JEPA_PARENT_VERSION="${FLOW_JEPA_PARENT_VERSION:-v106}"
export CLEARVLA_REQUIRED_MODEL_CONTRACT="${CLEARVLA_REQUIRED_MODEL_CONTRACT:-v106}"
export FLOW_JEPA_ROUTING_NORM_FLOOR="${FLOW_JEPA_ROUTING_NORM_FLOOR:-0.25}"
export FLOW_JEPA_CORRELATION_RMS_FLOOR="${FLOW_JEPA_CORRELATION_RMS_FLOOR:-0.10}"
export FLOW_JEPA_VISIBILITY_TRANSITION_FRACTION="${FLOW_JEPA_VISIBILITY_TRANSITION_FRACTION:-0.10}"
export FLOW_JEPA_HORIZON_VALUE_MAX_RMS="${FLOW_JEPA_HORIZON_VALUE_MAX_RMS:-0.50}"
export FLOW_JEPA_INTERVAL_STAGE_SCALE="${FLOW_JEPA_INTERVAL_STAGE_SCALE:-0.10}"
export FLOW_JEPA_INTERVAL_STAGE_WEIGHT="${FLOW_JEPA_INTERVAL_STAGE_WEIGHT:-0.02}"

printf '[v106] base=v105 stage=single_end_to_end stage_target=spatial_interval_increment intervals=4-8,8-16,16-32,32-48 teacher=robust_content_plus_signed_progression_plus_endpoint organizer=causal_per_cell_w2p fine_bank=protected numerics=complete role_floor:%s corr_floor:%s visibility_width:%s interval_scale=%s interval_weight=%s batch=%s\n' \
  "${FLOW_JEPA_ROUTING_NORM_FLOOR}" \
  "${FLOW_JEPA_CORRELATION_RMS_FLOOR}" \
  "${FLOW_JEPA_VISIBILITY_TRANSITION_FRACTION}" \
  "${FLOW_JEPA_INTERVAL_STAGE_SCALE}" \
  "${FLOW_JEPA_INTERVAL_STAGE_WEIGHT}" \
  "${V106_BATCH_SIZE}"

exec bash "${SCRIPT_DIR}/current_v105_horizon_addressed_flow_jepa.sh" \
  "$@" \
  --batch-size "${V106_BATCH_SIZE}" \
  --flow-jepa-variance-safe-routing 1 \
  --flow-jepa-complete-numerical-contract 1 \
  --flow-jepa-routing-norm-floor "${FLOW_JEPA_ROUTING_NORM_FLOOR}" \
  --flow-jepa-correlation-rms-floor "${FLOW_JEPA_CORRELATION_RMS_FLOOR}" \
  --flow-jepa-visibility-transition-fraction "${FLOW_JEPA_VISIBILITY_TRANSITION_FRACTION}" \
  --flow-jepa-horizon-value-max-rms "${FLOW_JEPA_HORIZON_VALUE_MAX_RMS}" \
  --flow-jepa-interval-stage-delta 1 \
  --flow-jepa-interval-boundaries "4,8,16,32,48" \
  --flow-jepa-interval-support-offsets "4,8,12,16,20,24,28,32,36,40,44,48" \
  --flow-jepa-interval-stage-update-scale "${FLOW_JEPA_INTERVAL_STAGE_SCALE}" \
  --flow-jepa-interval-stage-loss-weight "${FLOW_JEPA_INTERVAL_STAGE_WEIGHT}"
