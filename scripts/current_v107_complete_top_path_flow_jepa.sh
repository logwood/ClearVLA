#!/usr/bin/env bash
# V107: complete the observable top-to-bottom address and write path.
#
# - four factual soft policy glimpses survive until the P-side value read;
# - auxiliary horizon fine addresses retain every target 8x8 cell;
# - the signed interval increment is an explicit typed W->P value;
# - G/W/P residual bounds apply to the actual gated write proposal.
#
# All routes remain soft and differentiable.  Future teachers remain loss-only,
# the precision bank remains observation-owned, and V106 arithmetic is retained
# when these four V107 flags are disabled.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export OUT_DIR="${OUT_DIR:-runs/v107_complete_top_path_flow_jepa}"
export V107_BATCH_SIZE="${V107_BATCH_SIZE:-8}"
export V106_BATCH_SIZE="${V107_BATCH_SIZE}"
export FLOW_JEPA_PARENT_VERSION="${FLOW_JEPA_PARENT_VERSION:-v107}"
export CLEARVLA_REQUIRED_MODEL_CONTRACT="${CLEARVLA_REQUIRED_MODEL_CONTRACT:-v107}"

printf '[v107] base=v106 stage=single_end_to_end policy_address=multi_glimpse:%s horizon_fine=target_cell_specific interval_stage=typed_w2p residual_contract=post_gate batch=%s\n' \
  "${FLOW_JEPA_RAW_READER_HEADS:-4}" \
  "${V107_BATCH_SIZE}"

exec bash "${SCRIPT_DIR}/current_v106_interval_stage_flow_jepa.sh" \
  "$@" \
  --batch-size "${V107_BATCH_SIZE}" \
  --flow-jepa-raw-reader-heads "${FLOW_JEPA_RAW_READER_HEADS:-4}" \
  --flow-jepa-policy-multi-glimpse-address 1 \
  --flow-jepa-horizon-cell-fine-address 1 \
  --flow-jepa-interval-stage-typed-value 1 \
  --role-residual-contract-after-gate 1
