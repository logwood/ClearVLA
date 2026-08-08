#!/usr/bin/env bash
# Frozen paired causal probe for the differential intent/effect 3-2-3 graph.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

declare -a REQUIRED_SOURCE_MARKERS=(
  "${REPO_ROOT}/clearvla/policy/differential_intent_effect.py|class IntentStateBank"
  "${REPO_ROOT}/clearvla/policy/differential_intent_effect.py|class DifferentialWindowEffectBank"
  "${REPO_ROOT}/clearvla/policy/differential_intent_effect.py|class ConsequenceAwarePlanState"
  "${REPO_ROOT}/clearvla/experiments/observed_state_lab/policy_runtime_v39.py|clearvla-differential-intent-effect-model-path-v18"
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
    echo "[v118-model-path-probe] source mismatch: missing ${source_marker} in ${source_file}" >&2
    exit 2
  fi
done

export CHECKPOINT="${CHECKPOINT:-runs/v118_differential_intent_effect_323/checkpoints/latest.pt}"
export DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-runs/diagnostics/v118_model_path}"
export PROBE_BATCHES="${PROBE_BATCHES:-4}"
export BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-118}"
export RESULT_JSON="${RESULT_JSON:-${DIAGNOSTICS_DIR}/probe_v118_${PROBE_BATCHES}b.json}"
export MODEL_PATH_PROBE_LABEL=v118
export MODEL_PATH_REQUIRED_CONTRACT=differential_intent_effect_323
# Each mode owns one named boundary.  There is deliberately no uniform
# selector or P3 effect-lane probe: neither object exists in this capability.
export MODEL_PATH_MODES="${MODEL_PATH_MODES:-policy_temporal_shuffle flow_zero flow_spatial_shuffle dino_key_spatial_shuffle literal_current_rgb_zero literal_current_rgb_spatial_shuffle source_raw_key_zero source_raw_key_spatial_shuffle protected_detail_zero protected_detail_episode_shuffle future_effect_zero future_effect_spatial_shuffle future_effect_current_zero future_effect_current_spatial_shuffle future_effect_semantic_zero future_effect_semantic_spatial_shuffle future_effect_transport_zero future_effect_transport_spatial_shuffle future_effect_reliability_zero future_effect_reliability_spatial_shuffle future_effect_near_zero future_effect_near_shuffle future_effect_mid_zero future_effect_mid_shuffle future_effect_late_zero future_effect_late_shuffle intent_state_zero intent_state_episode_shuffle intent_window_near_zero intent_window_near_shuffle intent_window_mid_zero intent_window_mid_shuffle intent_window_late_zero intent_window_late_shuffle intent_temporal_zero intent_temporal_episode_shuffle p3_precision_delta_zero p3_precision_delta_episode_shuffle p3_temporal_delta_zero p3_temporal_delta_episode_shuffle goal_episode_shuffle action_history_episode_shuffle}"

exec bash "${SCRIPT_DIR}/run_v103_model_path_probe.sh" "$@"
