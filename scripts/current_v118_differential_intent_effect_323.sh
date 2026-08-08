#!/usr/bin/env bash
# Differential intent/effect 3-2-3 mainline.
#
# The numerical V118 label is experiment bookkeeping.  The required model
# contract is the capability identity below and validates the coherent graph
# directly instead of replaying every historical vXXX validator.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

for argument in "$@"; do
  case "${argument}" in
    --resume|--resume=*)
      echo "[differential-intent-effect-323] changed S/W/P ownership starts fresh" >&2
      exit 2
      ;;
  esac
done

export OUT_DIR="${OUT_DIR:-runs/v118_differential_intent_effect_323}"
export V118_BATCH_SIZE="${V118_BATCH_SIZE:-8}"
export V117_BATCH_SIZE="${V118_BATCH_SIZE}"
export FLOW_JEPA_PARENT_VERSION=v118_differential_intent_effect_323
export CLEARVLA_REQUIRED_MODEL_CONTRACT=differential_intent_effect_323

printf '[v118] architecture=differential_intent_effect_323 topology=3-2-3 S=four_token_state_bank W=near_mid_causal+late_typed P2=effect_specific_consequence_base P3=precision+temporal batch=%s\n' \
  "${V118_BATCH_SIZE}"

exec bash "${SCRIPT_DIR}/current_v117_window_effect_intent_p2.sh" \
  "$@" \
  --flow-jepa-differential-intent-effect-mainline 1
