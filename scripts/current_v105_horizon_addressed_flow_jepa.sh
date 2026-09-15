#!/usr/bin/env bash
# V105: horizon-specific, observation-only soft address for predictive JEPA.
#
# V104 remains reproducible. This wrapper adds:
#   1. one soft address posterior per real +4/+12/+24/+48 W chart;
#   2. continuous high-resolution values from the existing Flow-DINO lattice;
#   3. teacher-only spatial relevance supervision with no forward leakage;
#   4. a raw delta anchor plus reliability-weighted normalized delta loss.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export OUT_DIR="${OUT_DIR:-runs/v105_horizon_addressed_flow_jepa}"
export V105_BATCH_SIZE="${V105_BATCH_SIZE:-8}"
export V104_BATCH_SIZE="${V105_BATCH_SIZE}"
export FLOW_JEPA_PARENT_VERSION="${FLOW_JEPA_PARENT_VERSION:-v105}"
export CLEARVLA_REQUIRED_MODEL_CONTRACT="${CLEARVLA_REQUIRED_MODEL_CONTRACT:-v105}"
export FLOW_JEPA_HORIZON_ADDRESS_SCALE="${FLOW_JEPA_HORIZON_ADDRESS_SCALE:-0.10}"
export FLOW_JEPA_HORIZON_ADDRESS_WEIGHT="${FLOW_JEPA_HORIZON_ADDRESS_WEIGHT:-0.02}"

if [[ "${FLOW_JEPA_PARENT_VERSION}" == "v105" ]]; then
  printf '[v105] base=v104 future_address=horizon_soft_observation_only update_scale=%s address_weight=%s future_delta=raw_plus_reliable_normalized batch=%s\n' \
    "${FLOW_JEPA_HORIZON_ADDRESS_SCALE}" \
    "${FLOW_JEPA_HORIZON_ADDRESS_WEIGHT}" \
    "${V105_BATCH_SIZE}"
fi

exec bash "${SCRIPT_DIR}/current_v104_sequential_bounded_flow_jepa.sh" \
  "$@" \
  --batch-size "${V105_BATCH_SIZE}" \
  --flow-jepa-horizon-soft-address 1 \
  --flow-jepa-horizon-address-update-scale "${FLOW_JEPA_HORIZON_ADDRESS_SCALE}" \
  --flow-jepa-future-reliable-normalization 1 \
  --flow-jepa-horizon-address-loss-weight "${FLOW_JEPA_HORIZON_ADDRESS_WEIGHT}"
