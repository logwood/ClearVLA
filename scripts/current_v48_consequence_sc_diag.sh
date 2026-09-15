#!/usr/bin/env bash
# V48 baseline plus the first recovered v56-v59 mechanism:
# a no-grad consequence self-condition input and zero-base shortcut diagnostic.
#
# This intentionally does not enable trajectory manifold/coefficient writers,
# query-direction heads, or detail-micro branches.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUT_DIR="${OUT_DIR:-runs/v48_consequence_sc_diag_b8}"

exec bash "${SCRIPT_DIR}/current_v48_justok.sh" \
  --action-consequence-self-condition "${ACTION_CONSEQUENCE_SELF_CONDITION:-1}" \
  --layer-zero-base-diagnostic "${LAYER_ZERO_BASE_DIAGNOSTIC:-1}" \
  "$@"
