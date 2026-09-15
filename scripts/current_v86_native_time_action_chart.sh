#!/usr/bin/env bash
# V86: keep physical flow integration in the redundant manifold coordinates,
# but move every learned action-token operation into the native-time chart.
# Input fields are synthesized before token projection; the MMDiT receives an
# explicit horizon position; native 24x7 velocity is deterministically expanded
# back into the arm/gripper tangent fields. No smoothing or learned correction
# adapter is introduced.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUT_DIR="${OUT_DIR:-runs/v86_native_time_action_chart_b8}"

printf '%s\n' '[v86] native-time input chart + positioned action tokens + tangent-only output'

exec bash "${SCRIPT_DIR}/current_v85_unified_controller.sh" \
  "$@" \
  --arm-flow-mode manifold_native \
  --gripper-field-mode parseval_temporal
