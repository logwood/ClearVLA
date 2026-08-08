#!/usr/bin/env bash
# V115: G-aligned future consequences, observable stateless goal phases, and
# a 3-G / 2-W / 3-P top schedule whose P3 is a typed plan compiler.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

for argument in "$@"; do
  case "${argument}" in
    --resume|--resume=*)
      echo "[v115] direct resume is rejected; V115 starts from fresh model state" >&2
      exit 2
      ;;
  esac
done

export OUT_DIR="${OUT_DIR:-runs/v115_g_aligned_goal_phase_323}"
export V115_BATCH_SIZE="${V115_BATCH_SIZE:-8}"
export V114_BATCH_SIZE="${V115_BATCH_SIZE}"
export FLOW_JEPA_PARENT_VERSION="${FLOW_JEPA_PARENT_VERSION:-v115}"
# Default contract: CLEARVLA_REQUIRED_MODEL_CONTRACT=v115. A child launcher
# may set a stricter contract before entering this exact V115 parent graph.
export CLEARVLA_REQUIRED_MODEL_CONTRACT="${CLEARVLA_REQUIRED_MODEL_CONTRACT:-v115}"
export FLOW_JEPA_GROUNDING_BLOCKS=3
export FLOW_JEPA_WORLD_BLOCKS=2
export FLOW_JEPA_POLICY_BLOCKS=3
# V105's fixed-chart W posterior is superseded by the G-aligned soft-track
# teacher and the one online FutureEffectField.  Leaving its auxiliary loss on
# would train a tensor that V115 deliberately does not let P consume.
export FLOW_JEPA_HORIZON_ADDRESS_WEIGHT=0

# Preserve complete validation while bounding only the optional multi-sample
# diagnostics.  Frozen causal interventions remain standalone probes.
export V114_EVAL_SAMPLING_DIAGNOSTIC_BATCHES="${V115_EVAL_SAMPLING_DIAGNOSTIC_BATCHES:-4}"
export V114_EVAL_PROPOSAL_ABLATION_BATCHES="${V115_EVAL_PROPOSAL_ABLATION_BATCHES:-2}"
export V114_EVAL_EXECUTION_ABLATION_BATCHES="${V115_EVAL_EXECUTION_ABLATION_BATCHES:-1}"
export V114_EVAL_REPRESENTATION_BATCHES="${V115_EVAL_REPRESENTATION_BATCHES:-4}"

printf '[v115] base=v114 topology=3-2-3 teacher=g_aligned_soft_tracks W=single_supervised_future_effect phase=stateless_goal_program P1=shared_factual_once P2=basis_organizer P3=typed_plan_compiler bottom=unchanged batch=%s\n' \
  "${V115_BATCH_SIZE}"

exec bash "${SCRIPT_DIR}/current_v114_shared_factual_utility_precision.sh" \
  "$@" \
  --flow-jepa-shared-factual-glimpse-bank 1 \
  --flow-jepa-g-aligned-future-effect 1 \
  --flow-jepa-teacher-g-ema-decay 0.995 \
  --flow-jepa-stateless-goal-phase-machine 1 \
  --flow-jepa-top-role-schedule 3-2-3 \
  --flow-jepa-policy-plan-compiler 1
