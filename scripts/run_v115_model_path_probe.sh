#!/usr/bin/env bash
# Frozen paired causal probe for the complete V115 G -> W -> P mainline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

declare -a REQUIRED_SOURCE_MARKERS=(
  "${REPO_ROOT}/clearvla/policy/goal_conditioning.py|class StatelessGoalPhaseMachine"
  "${REPO_ROOT}/clearvla/policy/flow_dino_evidence.py|class FutureTeacherTrackPack"
  "${REPO_ROOT}/clearvla/policy/trunk.py|class PolicyPlanCompiler"
  "${REPO_ROOT}/clearvla/experiments/observed_state_lab/policy_runtime_v39.py|clearvla-v115-model-path-intervention-v15"
)
source_has_marker() {
  local marker="$1"
  local source_file="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -q --fixed-strings "${marker}" "${source_file}"
  else
    grep -Fq -- "${marker}" "${source_file}"
  fi
}
for requirement in "${REQUIRED_SOURCE_MARKERS[@]}"; do
  source_file="${requirement%%|*}"
  source_marker="${requirement#*|}"
  if [[ ! -f "${source_file}" ]] || ! source_has_marker "${source_marker}" "${source_file}"; then
    echo "[v115-model-path-probe] source mismatch: missing ${source_marker} in ${source_file}" >&2
    echo "[v115-model-path-probe] sync the complete V115 source snapshot before rerunning" >&2
    exit 2
  fi
done

export CHECKPOINT="${CHECKPOINT:-runs/v115_g_aligned_goal_phase_323/checkpoints/latest.pt}"
export DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-runs/diagnostics/v115_model_path}"
export PROBE_BATCHES="${PROBE_BATCHES:-4}"
export BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-115}"
export RESULT_JSON="${RESULT_JSON:-${DIAGNOSTICS_DIR}/probe_v115_${PROBE_BATCHES}b.json}"
export MODEL_PATH_PROBE_LABEL=v115
export MODEL_PATH_REQUIRED_CONTRACT=v115

exec bash "${SCRIPT_DIR}/run_v103_model_path_probe.sh" "$@"
