#!/usr/bin/env bash
# =============================================================================
# V77: identifiable residual amplitude for the serial-owned MMDiT decoder.
#
# This keeps the V76 ownership/intent/workspace topology unchanged. Every
# self/noisy/stage/low/FFN residual is normalized over its per-sample token
# field before bounded gates are applied. Projection weights therefore own
# direction/content only; gates own amplitude. Stage promotion follows the
# same order. Dynamic orthogonal cancellation diagnostics are enabled in code.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HIERARCHICAL_MMDIT_DEPTH="${HIERARCHICAL_MMDIT_DEPTH:-3}"
export HIERARCHICAL_MMDIT_REFINE_STEPS="${HIERARCHICAL_MMDIT_REFINE_STEPS:-3}"
export OUT_DIR="${OUT_DIR:-runs/v77_gate_identifiable_mmdit_d${HIERARCHICAL_MMDIT_DEPTH}_s${HIERARCHICAL_MMDIT_REFINE_STEPS}_b8}"

exec bash "${SCRIPT_DIR}/current_v76a_owned_intent_mmdit.sh" "$@"
