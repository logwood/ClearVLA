#!/usr/bin/env bash
# V89: typed execution ownership on top of V88 value-supervised dwell.
#
# - six stage slots remain typed memory/retrieval evidence;
# - three parameter-distinct MMDiT blocks are the executable repertoire;
# - host LayerScale is the only residual-amplitude owner;
# - the unified controller selects dwell and nested branch capacity.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HIERARCHICAL_MMDIT_EXECUTION_CONTRACT="${HIERARCHICAL_MMDIT_EXECUTION_CONTRACT:-typed_block_budget}"
export OUT_DIR="${OUT_DIR:-runs/v89_typed_block_budget}"

printf '[v89] execution_contract=%s dwell=%s warmup=%s\n' \
  "${HIERARCHICAL_MMDIT_EXECUTION_CONTRACT}" \
  "${HIERARCHICAL_MMDIT_DWELL_MODE:-learned}" \
  "${HIERARCHICAL_MMDIT_OPERATION_VALUE_WARMUP_STEPS:-200}"

exec bash "${SCRIPT_DIR}/current_v88_value_dwell.sh" \
  --hierarchical-mmdit-execution-contract \
    "${HIERARCHICAL_MMDIT_EXECUTION_CONTRACT}" \
  "$@"
