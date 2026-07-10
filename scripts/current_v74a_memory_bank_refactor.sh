#!/usr/bin/env bash
# =============================================================================
# V74A: MemoryBank / WorkspaceController structure split.
#
# This is intentionally a structural refactor, not a stronger retrieval model:
#   - EvidenceMemoryBank owns source normalization, type embeddings, source
#     validation, role accounting, and static K/V cache preparation.
#   - SemanticEvidenceWorkspace remains the controller that queries the bank and
#     emits the same 24 workspace tokens to MMDiT.
#   - Retrieval math and MMDiT external token interface stay aligned with the
#     V72 evidence topology.
#
# Read against V72:
#   - pflow/full metrics should stay close; large drift means the refactor
#     changed behavior unintentionally.
#   - This script intentionally does NOT remove workspace global/layer/progress
#     value sources.  Cutting those sources is a later structural ablation, not
#     part of the safe memory-bank refactor baseline.
#   - wgeom/wtrn/wevt/wstate/wrlay/wglob are new role-level diagnostics only;
#     wevt is reserved for future true event evidence and should be zero now.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ARM_NOISE_TEMPORAL_RHO="${ARM_NOISE_TEMPORAL_RHO:-0.0}"
export OUT_DIR="${OUT_DIR:-runs/v74a_memory_bank_refactor_from_v72_rho${ARM_NOISE_TEMPORAL_RHO}_b8}"

exec bash "${SCRIPT_DIR}/current_v72_progress_isolation.sh" \
  --latent-cvae-workspace-global-sources 1 \
  --latent-cvae-workspace-layer-source 1 \
  --latent-cvae-workspace-progress-value 1 \
  --latent-cvae-workspace-time-state 0 \
  "$@"
