#!/usr/bin/env bash
# V109: typed progressive address formation across the existing 3/3/2 trunk.
#
# Pre-G compiles deterministic observation geometry. G1 establishes complete
# coarse hypotheses, G2 rectifies continuous fine support, and G3 compiles a
# selector-only canonical handoff. W remains horizon-query owned. The existing
# W->P late reader performs the first and only high-resolution raw value read.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export OUT_DIR="${OUT_DIR:-runs/v109_progressive_grounding_address_flow_jepa}"
export V109_BATCH_SIZE="${V109_BATCH_SIZE:-8}"
export V108_BATCH_SIZE="${V109_BATCH_SIZE}"
export FLOW_JEPA_PARENT_VERSION="${FLOW_JEPA_PARENT_VERSION:-v109}"
export CLEARVLA_REQUIRED_MODEL_CONTRACT="${CLEARVLA_REQUIRED_MODEL_CONTRACT:-v109}"

printf '[v109] base=v108 stage=single_end_to_end address=pre_g_scaffold+g1_hypothesis+g2_rectification+g3_canonical world=horizon_posterior policy=first_highres_read batch=%s\n' \
  "${V109_BATCH_SIZE}"

exec bash "${SCRIPT_DIR}/current_v108_online_horizon_address_flow_jepa.sh" \
  "$@" \
  --batch-size "${V109_BATCH_SIZE}" \
  --flow-jepa-progressive-grounding-address 1
