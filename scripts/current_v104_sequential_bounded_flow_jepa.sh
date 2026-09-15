#!/usr/bin/env bash
# V104: structural repair of the complete V103 model path.
#
# V103 remains reproducible.  This wrapper adds three mechanism contracts:
#   1. every learned flow is a smooth in-image source-relative coordinate;
#   2. G/W/P residual writes and typed AttnRes values have bounded normalized
#      RMS, without hard clipping, detach, quotas, or gradient patches;
#   3. +4/+12/+24/+48 are produced by one observation-history memory in
#      chronological order inside a single stateless forward call.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export OUT_DIR="${OUT_DIR:-runs/v104_sequential_bounded_flow_jepa}"
export V104_BATCH_SIZE="${V104_BATCH_SIZE:-8}"
export V103_BATCH_SIZE="${V104_BATCH_SIZE}"
export FLOW_JEPA_PARENT_VERSION="${FLOW_JEPA_PARENT_VERSION:-v104}"
export CLEARVLA_REQUIRED_MODEL_CONTRACT="${CLEARVLA_REQUIRED_MODEL_CONTRACT:-v104}"
export ROLE_RESIDUAL_MAX_UPDATE_RMS="${ROLE_RESIDUAL_MAX_UPDATE_RMS:-0.50}"
export ROLE_ATTNRES_MAX_VALUE_RMS="${ROLE_ATTNRES_MAX_VALUE_RMS:-1.00}"

printf '[v104] base=v103 geometry=bounded_source_relative motion_units=normalized role_residual=soft_rms(max=%s) attnres_value=soft_rms(max=%s) horizon_memory=sequential_history batch=%s\n' \
  "${ROLE_RESIDUAL_MAX_UPDATE_RMS}" \
  "${ROLE_ATTNRES_MAX_VALUE_RMS}" \
  "${V104_BATCH_SIZE}"

exec bash "${SCRIPT_DIR}/current_v103_typed_predictive_flow_jepa.sh" \
  "$@" \
  --batch-size "${V104_BATCH_SIZE}" \
  --flow-jepa-bounded-flow-coordinates 1 \
  --flow-jepa-sequential-horizon-memory 1 \
  --role-residual-amplitude-contract 1 \
  --role-residual-max-update-rms "${ROLE_RESIDUAL_MAX_UPDATE_RMS}" \
  --role-attnres-max-value-rms "${ROLE_ATTNRES_MAX_VALUE_RMS}"
