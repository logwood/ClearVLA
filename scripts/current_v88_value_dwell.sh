#!/usr/bin/env bash
# V88: horizon-resolved candidate value learning with monotonic dwell.
# All refinement steps execute. The first 200 optimizer steps keep the exact
# V87 assignment while the detached candidate-value reader calibrates.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HIERARCHICAL_MMDIT_OPERATION_VALUE_LOSS_WEIGHT="${HIERARCHICAL_MMDIT_OPERATION_VALUE_LOSS_WEIGHT:-0.05}"
export HIERARCHICAL_MMDIT_OPERATION_VALUE_HUBER_DELTA="${HIERARCHICAL_MMDIT_OPERATION_VALUE_HUBER_DELTA:-0.10}"
export HIERARCHICAL_MMDIT_OPERATION_VALUE_RELIABILITY_SCALE="${HIERARCHICAL_MMDIT_OPERATION_VALUE_RELIABILITY_SCALE:-0.0}"
export HIERARCHICAL_MMDIT_OPERATION_VALUE_WARMUP_STEPS="${HIERARCHICAL_MMDIT_OPERATION_VALUE_WARMUP_STEPS:-200}"
export HIERARCHICAL_MMDIT_DWELL_MODE="${HIERARCHICAL_MMDIT_DWELL_MODE:-learned}"
export HIERARCHICAL_MMDIT_OPERATION_CANDIDATE_PROBES=1
export HIERARCHICAL_MMDIT_OPERATION_ROUTE_LOSS_WEIGHT=0
export HIERARCHICAL_MMDIT_EXHAUSTION_MODE=off
export OUT_DIR="${OUT_DIR:-runs/v88_value_dwell}"

printf '[v88] dwell=%s warmup=%s value_weight=%s huber=%s reliability_scale=%s\n' \
  "${HIERARCHICAL_MMDIT_DWELL_MODE}" \
  "${HIERARCHICAL_MMDIT_OPERATION_VALUE_WARMUP_STEPS}" \
  "${HIERARCHICAL_MMDIT_OPERATION_VALUE_LOSS_WEIGHT}" \
  "${HIERARCHICAL_MMDIT_OPERATION_VALUE_HUBER_DELTA}" \
  "${HIERARCHICAL_MMDIT_OPERATION_VALUE_RELIABILITY_SCALE}"

exec bash "${SCRIPT_DIR}/current_v87_spectral_controller.sh" \
  --hierarchical-mmdit-operation-candidate-probes 1 \
  --hierarchical-mmdit-operation-route-loss-weight 0 \
  --hierarchical-mmdit-oracle-route-loss-weight 0 \
  --hierarchical-mmdit-exhaustion-mode off \
  --hierarchical-mmdit-dwell-mode "${HIERARCHICAL_MMDIT_DWELL_MODE}" \
  --hierarchical-mmdit-operation-value-warmup-steps \
    "${HIERARCHICAL_MMDIT_OPERATION_VALUE_WARMUP_STEPS}" \
  --hierarchical-mmdit-operation-value-loss-weight \
    "${HIERARCHICAL_MMDIT_OPERATION_VALUE_LOSS_WEIGHT}" \
  --hierarchical-mmdit-operation-value-huber-delta \
    "${HIERARCHICAL_MMDIT_OPERATION_VALUE_HUBER_DELTA}" \
  --hierarchical-mmdit-operation-value-reliability-scale \
    "${HIERARCHICAL_MMDIT_OPERATION_VALUE_RELIABILITY_SCALE}" \
  "$@"
