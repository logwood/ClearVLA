#!/usr/bin/env bash
# =============================================================================
# V73A: light workspace partition on top of V72.
#
# Single structural variable:
#   - scan/lateral stay in the CVAE global condition / z path, but are removed
#     from workspace values.
#
# This tests whether the old shelf duplicated the same global summary twice:
# once through cond/z and once as ordinary evidence.  Layer, progress,
# trajectory, rollout, transition and routed_layer behavior stays V72-like.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ARM_NOISE_TEMPORAL_RHO="${ARM_NOISE_TEMPORAL_RHO:-0.0}"
export OUT_DIR="${OUT_DIR:-runs/v73a_workspace_light_partition_rho${ARM_NOISE_TEMPORAL_RHO}_b8}"

exec bash "${SCRIPT_DIR}/current_v72_progress_isolation.sh" \
  --latent-cvae-workspace-global-sources 0 \
  "$@"
