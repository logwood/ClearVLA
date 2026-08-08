#!/usr/bin/env bash
# V112: explicit public/private selector ownership before the single P value read.
#
# G3 public memory is projected from clean query + owner-neutral geometry.
# Semantic/appearance/geometry/interval selector states then advance at the
# G3->W entry and after W1/W2/W3.  P1 consumes the W appearance posterior in
# its actual joint source/slot/fine factor.  Raw values are still read exactly
# once at W->P; P2, the bottom CVAE/3x2 MMDiT, workspace and execution stack are
# unchanged.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export OUT_DIR="${OUT_DIR:-runs/v112_pre_value_owner_routing}"
export V112_BATCH_SIZE="${V112_BATCH_SIZE:-8}"
export V111_BATCH_SIZE="${V112_BATCH_SIZE}"
# Standalone default: CLEARVLA_REQUIRED_MODEL_CONTRACT=v112.  A descendant
# launcher may set a stricter contract before chaining through this parent.
export FLOW_JEPA_PARENT_VERSION="${FLOW_JEPA_PARENT_VERSION:-v112}"
export CLEARVLA_REQUIRED_MODEL_CONTRACT="${CLEARVLA_REQUIRED_MODEL_CONTRACT:-v112}"

printf '[v112] base=v111 G3=explicit_public+private_sidecars W=pre_value_semantic+appearance+geometry+interval P1=joint_source_slot_fine bottom=unchanged batch=%s\n' \
  "${V112_BATCH_SIZE}"

exec bash "${SCRIPT_DIR}/current_v111_structured_ownership_bottleneck.sh" \
  "$@" \
  --batch-size "${V112_BATCH_SIZE}" \
  --flow-jepa-pre-value-owner-routing 1 \
  --flow-jepa-pre-value-owner-update-scale 0.10
