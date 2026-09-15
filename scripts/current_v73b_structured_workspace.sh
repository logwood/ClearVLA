#!/usr/bin/env bash
# =============================================================================
# V73B: structured, time-aware workspace on top of V72.
#
# This is the stronger semantic partition:
#   - scan/lateral are global memory only: they form cond/z and do not become
#     workspace values.
#   - full layer_stack is not a static shelf menu; layer information reaches
#     workspace through per-step routed_layer/capsules.
#   - progress is query/state, not evidence value.
#   - workspace slots explicitly receive the existing MMDiT primary condition
#     (z + time_lift(time_emb)).  No new time embedding is introduced, so flow
#     time semantics stay aligned with training, sampling and eval.
#
# Read against V73A:
#   - wscan/wlat should be zero in both arms.
#   - wroute/wcaps/trajectory/rollout decide whether structured evidence can
#     replace the old static soup.
#   - wgstate should be nonzero only here.
#   - wpq should be nonzero while wprog should be zero.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ARM_NOISE_TEMPORAL_RHO="${ARM_NOISE_TEMPORAL_RHO:-0.0}"
export OUT_DIR="${OUT_DIR:-runs/v73b_structured_workspace_rho${ARM_NOISE_TEMPORAL_RHO}_b8}"

exec bash "${SCRIPT_DIR}/current_v72_progress_isolation.sh" \
  --latent-cvae-workspace-global-sources 0 \
  --latent-cvae-workspace-layer-source 0 \
  --latent-cvae-workspace-progress-value 0 \
  --latent-cvae-workspace-time-state 1 \
  "$@"
