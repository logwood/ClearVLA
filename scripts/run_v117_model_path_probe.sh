#!/usr/bin/env bash
# Frozen paired causal probe for V117's intent -> window-effect -> P2 path.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

declare -a REQUIRED_SOURCE_MARKERS=(
  "${REPO_ROOT}/clearvla/policy/goal_conditioning.py|class StatelessIntentController"
  "${REPO_ROOT}/clearvla/policy/flow_dino_evidence.py|class WindowEffectBank"
  "${REPO_ROOT}/clearvla/policy/trunk.py|class StructuredFutureEffectReader"
  "${REPO_ROOT}/clearvla/experiments/observed_state_lab/policy_runtime_v39.py|clearvla-v117-model-path-intervention-v17"
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
    echo "[v117-model-path-probe] source mismatch: missing ${source_marker} in ${source_file}" >&2
    exit 2
  fi
done

export CHECKPOINT="${CHECKPOINT:-runs/v117_window_effect_intent_p2/checkpoints/latest.pt}"
export DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-runs/diagnostics/v117_model_path}"
export PROBE_BATCHES="${PROBE_BATCHES:-4}"
export BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-117}"
export RESULT_JSON="${RESULT_JSON:-${DIAGNOSTICS_DIR}/probe_v117_${PROBE_BATCHES}b.json}"
export MODEL_PATH_PROBE_LABEL=v117
export MODEL_PATH_REQUIRED_CONTRACT=v117
# Default to the causal boundaries that can accept/reject V117.  The inherited
# V103 probe contains dozens of historical modes and is intentionally available
# only through MODEL_PATH_MODES=all when an exhaustive ancestry audit is needed.
export MODEL_PATH_MODES="${MODEL_PATH_MODES:-policy_temporal_shuffle flow_zero dino_key_spatial_shuffle literal_current_rgb_zero protected_detail_zero future_effect_zero future_effect_semantic_zero future_effect_transport_zero future_effect_reliability_zero intent_window_selector_uniform intent_window_selector_episode_shuffle intent_temporal_zero intent_temporal_episode_shuffle p3_precision_delta_zero p3_effect_delta_zero p3_temporal_delta_zero goal_episode_shuffle action_history_episode_shuffle}"

exec bash "${SCRIPT_DIR}/run_v103_model_path_probe.sh" "$@"
