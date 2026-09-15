#!/usr/bin/env bash
# V117: V116 ancestry with observable stateless intent, a supervised
# near/mid/late WindowEffectBank, and the only structured effect read in P2.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

for argument in "$@"; do
  case "${argument}" in
    --resume|--resume=*)
      echo "[v117] direct resume is rejected; the changed S/W/P topology starts fresh" >&2
      exit 2
      ;;
  esac
done

export OUT_DIR="${OUT_DIR:-runs/v117_window_effect_intent_p2}"
export V117_BATCH_SIZE="${V117_BATCH_SIZE:-8}"
export V116_BATCH_SIZE="${V117_BATCH_SIZE}"
# Standalone default remains CLEARVLA_REQUIRED_MODEL_CONTRACT=v117; descendants
# may provide a stricter capability identity before entering this parent graph.
export FLOW_JEPA_PARENT_VERSION="${FLOW_JEPA_PARENT_VERSION:-v117}"
export CLEARVLA_REQUIRED_MODEL_CONTRACT="${CLEARVLA_REQUIRED_MODEL_CONTRACT:-v117}"

printf '[v117] base=v116 topology=3-2-3 S=three_block_stateless_intent W=near+mid+late_window_effect P2=single_structured_effect_read P3=typed_plan_compiler batch=%s\n' \
  "${V117_BATCH_SIZE}"

exec bash "${SCRIPT_DIR}/current_v116_supervised_effect_mainline.sh" \
  "$@" \
  --flow-jepa-stateless-intent-controller 1 \
  --flow-jepa-window-effect-bank 1 \
  --flow-jepa-future-slots 3 \
  --flow-jepa-effect-read-in-p2 1
