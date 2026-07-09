#!/usr/bin/env bash
# =============================================================================
# V74A: MemoryBank / WorkspaceController structure split.
#
# This is intentionally a structural refactor, not a stronger retrieval model:
#   - EvidenceMemoryBank owns source normalization, type embeddings, source
#     validation, role accounting, and static K/V cache preparation.
#   - SemanticEvidenceWorkspace remains the controller that queries the bank and
#     emits the same 24 workspace tokens to MMDiT.
#   - Retrieval math and MMDiT external token interface stay aligned with V73B.
#
# Read against V73B:
#   - pflow/full metrics should stay close; large drift means the refactor
#     changed behavior unintentionally.
#   - wgeom/wevt/wstate/wrlay/wglob are new role-level diagnostics only.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ARM_NOISE_TEMPORAL_RHO="${ARM_NOISE_TEMPORAL_RHO:-0.0}"
export OUT_DIR="${OUT_DIR:-runs/v74a_memory_bank_refactor_rho${ARM_NOISE_TEMPORAL_RHO}_b8}"

exec bash "${SCRIPT_DIR}/current_v73b_structured_workspace.sh" "$@"
