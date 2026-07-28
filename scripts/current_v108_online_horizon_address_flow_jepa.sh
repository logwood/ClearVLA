#!/usr/bin/env bash
# V108: move the horizon-address read into the deployed G3 -> W1 carrier.
#
# The existing observation-only bank is read exactly once.  Its bounded result
# updates the ordinary rollout consumed by W/P/action; the final JEPA predictor
# reuses that final carrier and does not perform another address read.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export OUT_DIR="${OUT_DIR:-runs/v108_online_horizon_address_flow_jepa}"
export V108_BATCH_SIZE="${V108_BATCH_SIZE:-8}"
export V107_BATCH_SIZE="${V108_BATCH_SIZE}"
export FLOW_JEPA_PARENT_VERSION="${FLOW_JEPA_PARENT_VERSION:-v108}"
export CLEARVLA_REQUIRED_MODEL_CONTRACT="${CLEARVLA_REQUIRED_MODEL_CONTRACT:-v108}"

printf '[v108] base=v107 stage=single_end_to_end horizon_address=online_g3_to_w1 future_read=single carrier=existing_rollout batch=%s\n' \
  "${V108_BATCH_SIZE}"

exec bash "${SCRIPT_DIR}/current_v107_complete_top_path_flow_jepa.sh" \
  "$@" \
  --batch-size "${V108_BATCH_SIZE}" \
  --flow-jepa-online-horizon-address 1
