#!/usr/bin/env bash
# V111: functional evidence ownership on the complete V110 path.
#
# G2 separates semantic hypothesis, appearance verification and geometric
# rectification. G3 exposes one public camera-spatial chart while retaining
# typed sidecars. W composes chronological interval innovations. P factorizes
# source and fine posteriors, then performs typed local RGB/detail operations.
# The bottom CVAE, 3x2 MMDiT and execution machinery are unchanged.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export OUT_DIR="${OUT_DIR:-runs/v111_structured_ownership_bottleneck}"
export V111_BATCH_SIZE="${V111_BATCH_SIZE:-8}"
export V110_BATCH_SIZE="${V111_BATCH_SIZE}"
export FLOW_JEPA_PARENT_VERSION="${FLOW_JEPA_PARENT_VERSION:-v111}"
# Standalone default is V111.  Descendant launchers own stricter contracts and
# must survive the complete parent chain (V112/V113 and future extensions).
export CLEARVLA_REQUIRED_MODEL_CONTRACT="${CLEARVLA_REQUIRED_MODEL_CONTRACT:-v111}"

printf '[v111] base=v110 stage=single_end_to_end G=public_chart+typed_sidecars W=interval_innovations P=factorized_source_fine+typed_local_ops bottom=unchanged batch=%s\n' \
  "${V111_BATCH_SIZE}"

exec bash "${SCRIPT_DIR}/current_v110_coordinate_typed_raw_jepa.sh" \
  "$@" \
  --batch-size "${V111_BATCH_SIZE}" \
  --flow-jepa-structured-ownership-bottleneck 1
