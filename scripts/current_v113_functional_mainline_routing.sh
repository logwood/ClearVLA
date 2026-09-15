#!/usr/bin/env bash
# V113: functional typed ownership from G3 through W/P1/P2.
#
# Owner innovations are selected in the compact route space and reconstructed
# to hidden width once.  W appearance is a required P1 verifier; P2 preserves
# its policy carrier and routes typed innovations.  Goal, executed history and
# phase remain separate per-horizon selector contexts.  Raw values are still
# read exactly once and the bottom CVAE/3x2 MMDiT/workspace/execution stack is
# unchanged.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export OUT_DIR="${OUT_DIR:-runs/v113_functional_mainline_routing}"
export V113_BATCH_SIZE="${V113_BATCH_SIZE:-1}"
export V112_BATCH_SIZE="${V113_BATCH_SIZE}"
export FLOW_JEPA_PARENT_VERSION="${FLOW_JEPA_PARENT_VERSION:-v113}"
export CLEARVLA_REQUIRED_MODEL_CONTRACT="${CLEARVLA_REQUIRED_MODEL_CONTRACT:-v113}"

printf '[v113] base=v112 W=route_space_owner_select+single_hidden_write P1=mandatory_W_appearance P2=protected_policy+routed_typed_delta horizon=phase+goal+history_per_anchor raw_read=once bottom=unchanged batch=%s\n' \
  "${V113_BATCH_SIZE}"

exec bash "${SCRIPT_DIR}/current_v112_pre_value_owner_routing.sh" \
  "$@" \
  --batch-size "${V113_BATCH_SIZE}" \
  --flow-jepa-functional-mainline-routing 1
