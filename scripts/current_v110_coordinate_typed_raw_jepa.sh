#!/usr/bin/env bash
# V110: coordinate-typed current RGB/detail ownership on the stable V109
# progressive G1/G2/G3 -> W -> P topology.
#
# Native-resolution current RGB is sampled only at explicit current-anchor
# coordinates. Future evidence remains a soft transport distribution over
# those observed anchors;
# no future RGB crop is fabricated.  The W->P ingress preserves a structured
# 3x3 micro-grid and performs the learned local value read before the existing
# P1/P2 policy blocks.  The lower policy stack is deliberately inherited
# unchanged.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export OUT_DIR="${OUT_DIR:-runs/v110_coordinate_typed_raw_jepa}"
export V110_BATCH_SIZE="${V110_BATCH_SIZE:-8}"
export V109_BATCH_SIZE="${V110_BATCH_SIZE}"
export FLOW_JEPA_PARENT_VERSION="${FLOW_JEPA_PARENT_VERSION:-v110}"
if [[ -z "${CLEARVLA_REQUIRED_MODEL_CONTRACT:-}" ]]; then
  export CLEARVLA_REQUIRED_MODEL_CONTRACT=v110
fi

printf '[v110] base=v109+v107_numerics stage=single_end_to_end address=typed_g1_g2_g3+w_transport policy=typed_ingress_microgrid+local_refiner_then_p1_p2 bottom=unchanged batch=%s\n' \
  "${V110_BATCH_SIZE}"

exec bash "${SCRIPT_DIR}/current_v109_progressive_grounding_address_flow_jepa.sh" \
  "$@" \
  --batch-size "${V110_BATCH_SIZE}" \
  --flow-jepa-coordinate-typed-raw-detail 1 \
  --flow-jepa-raw-micro-grid 3
