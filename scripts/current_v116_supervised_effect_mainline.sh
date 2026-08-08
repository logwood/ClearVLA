#!/usr/bin/env bash
# V116: V115's 3-2-3 graph with one fully supervised W effect interface,
# structured P2 effect read, non-terminal four-state phase belief, and a
# separate small-prior execution terminal signal.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

for argument in "$@"; do
  case "${argument}" in
    --resume|--resume=*)
      echo "[v116] direct resume is rejected; V116 starts fresh" >&2
      exit 2
      ;;
  esac
done

export OUT_DIR="${OUT_DIR:-runs/v116_supervised_effect_mainline}"
export V116_BATCH_SIZE="${V116_BATCH_SIZE:-8}"
export V115_BATCH_SIZE="${V116_BATCH_SIZE}"
# V116 provides the standalone defaults, while descendants (currently V117)
# must be able to preserve their stricter serialized/source contract through
# this ancestry wrapper.
export FLOW_JEPA_PARENT_VERSION="${FLOW_JEPA_PARENT_VERSION:-v116}"
export CLEARVLA_REQUIRED_MODEL_CONTRACT="${CLEARVLA_REQUIRED_MODEL_CONTRACT:-v116}"

printf '[v116] base=v115 topology=3-2-3 W=supervised_current_successor_effect P2=structured_spatial_effect P3=precision+effect+temporal terminal=execution_only phase=4state_nonterminal flow_time=beta_1_5_1 batch=%s\n' \
  "${V116_BATCH_SIZE}"

exec bash "${SCRIPT_DIR}/current_v115_g_aligned_goal_phase_323.sh" \
  "$@" \
  --flow-jepa-supervised-effect-mainline 1 \
  --flow-matching-time-distribution beta_1_5_1
