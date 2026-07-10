#!/usr/bin/env bash
# =============================================================================
# V74B: central WorkspaceController on top of V74A.
#
# This version keeps the V74A MemoryBank / WorkspaceController split and turns
# on the central controller:
#   - starts from the V72 evidence-source topology; global/layer/progress
#     workspace values remain present in this baseline;
#   - role-aware key bias, read capacity, delay gate, and query modulation come
#     from one controller state;
#   - workspace seeds and controller state both use the existing primary
#     condition (z + time_lift(t)); workspace seed injection is slot-aware
#     instead of one identical vector broadcast to all slots;
#   - route time is added to layer/progress/capsule route queries only;
#   - memory K/V and MMDiT's external 24-token workspace interface are unchanged.
#
# Important diagnostic semantics:
#   - wtrn = transition/consequence memory consumption.  This includes the
#            current transition_event slot, which is still derived from
#            controlled_delta and is not trusted as true event evidence yet.
#   - wevt = true event role. It should stay zero until a real event evidence
#            source is added.
#   - ctrlcap/ctrldly/ctrlent/ctrlq show whether the controller is delaying,
#     allocating capacity, or collapsing.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ARM_NOISE_TEMPORAL_RHO="${ARM_NOISE_TEMPORAL_RHO:-0.0}"
export OUT_DIR="${OUT_DIR:-runs/v74b_workspace_controller_from_v72_rho${ARM_NOISE_TEMPORAL_RHO}_b8}"

exec bash "${SCRIPT_DIR}/current_v74a_memory_bank_refactor.sh" \
  --latent-cvae-workspace-time-state 1 \
  --latent-cvae-workspace-slot-time-state 1 \
  --latent-cvae-workspace-slot-time-scale 0.10 \
  --latent-cvae-workspace-controller 1 \
  --adaptive-cvae-route-time-query 1 \
  "$@"
