#!/usr/bin/env bash
# Frozen causal probe for the grounded_intent_effect_323 capability.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

declare -a REQUIRED_SOURCE_MARKERS=(
  "${REPO_ROOT}/clearvla/policy/grounded_intent_effect.py|class GroundedFactSet"
  "${REPO_ROOT}/clearvla/policy/grounded_intent_effect.py|class FutureEffectField"
  "${REPO_ROOT}/clearvla/policy/grounded_intent_effect.py|class PolicyPlanDeltaBank"
  "${REPO_ROOT}/clearvla/experiments/observed_state_lab/policy_runtime_v39.py|clearvla-grounded-intent-effect-323-model-path-v1"
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
    echo "[grounded-intent-effect-323-probe] source mismatch: missing ${source_marker} in ${source_file}" >&2
    exit 2
  fi
done

# ``full`` preserves the broad frozen-checkpoint audit.  The focused profile
# follows up the two unresolved causal questions from the first valid probe:
# whether reliability suppresses useful W content, and whether G3 object-slot
# identity (rather than its public scaffold) reaches S/W/P2.  An explicitly
# supplied MODEL_PATH_MODES always wins over either profile.
export CLEARVLA_DATA_CACHE_ROOT="${CLEARVLA_DATA_CACHE_ROOT:-/data/senwang/data}"
export CLEARVLA_CHECKPOINT_ROOT="${CLEARVLA_CHECKPOINT_ROOT:-/data/senwang/checkpoint}"
export DATA_ROOT="${DATA_ROOT:-/data/liang.zhang/dataset/grab_pen_single/grab_pen_single}"
export CACHE_DIR="${CACHE_DIR:-${CLEARVLA_DATA_CACHE_ROOT}/cache_336}"
export DINO_CACHE_DIR="${DINO_CACHE_DIR:-${CLEARVLA_DATA_CACHE_ROOT}/dinov2_cache_336}"
export CHECKPOINT="${CHECKPOINT:-${CLEARVLA_CHECKPOINT_ROOT}/v119_grounded_intent_effect_323_b8/checkpoints/latest.pt}"
export DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-runs/diagnostics/grounded_intent_effect_323}"
export PROBE_BATCHES="${PROBE_BATCHES:-4}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-119}"
export RESULT_JSON="${RESULT_JSON:-${DIAGNOSTICS_DIR}/probe_grounded_323_${PROBE_BATCHES}b.json}"
export MODEL_PATH_PROBE_LABEL=grounded-intent-effect-323
export MODEL_PATH_REQUIRED_CONTRACT=grounded_intent_effect_323
PROBE_PROFILE="${PROBE_PROFILE:-full}"
FULL_MODEL_PATH_MODES="goal_zero goal_episode_shuffle action_history_episode_shuffle address_g3_zero address_g3_episode_shuffle address_g3_slot_permute address_g3_slot_mean dino_key_spatial_shuffle literal_current_rgb_zero raw_value_zero flow_zero flow_spatial_shuffle intent_state_zero intent_state_episode_shuffle intent_interval_h4_8_zero intent_interval_h4_8_shuffle intent_interval_h8_16_zero intent_interval_h8_16_shuffle intent_interval_h16_32_zero intent_interval_h16_32_shuffle intent_interval_h32_48_zero intent_interval_h32_48_shuffle future_effect_zero future_effect_spatial_shuffle future_effect_current_zero future_effect_current_spatial_shuffle future_effect_semantic_zero future_effect_semantic_spatial_shuffle future_effect_transport_zero future_effect_transport_spatial_shuffle future_effect_reliability_zero future_effect_reliability_spatial_shuffle future_effect_reliability_one future_effect_h4_8_zero future_effect_h4_8_shuffle future_effect_h8_16_zero future_effect_h8_16_shuffle future_effect_h16_32_zero future_effect_h16_32_shuffle future_effect_h32_48_zero future_effect_h32_48_shuffle protected_detail_zero protected_detail_episode_shuffle p3_precision_delta_zero p3_precision_delta_episode_shuffle p3_temporal_delta_zero p3_temporal_delta_episode_shuffle"
INTENT_EFFECT_FOLLOWUP_MODEL_PATH_MODES="goal_zero goal_episode_shuffle action_history_episode_shuffle intent_state_zero intent_state_episode_shuffle intent_goal_set_zero intent_goal_set_episode_shuffle intent_achieved_zero intent_achieved_episode_shuffle intent_remaining_zero intent_remaining_episode_shuffle intent_temporal_zero intent_temporal_episode_shuffle intent_interval_h16_32_zero intent_interval_h16_32_shuffle future_effect_zero future_effect_reliability_zero future_effect_reliability_one address_g3_zero address_g3_slot_permute address_g3_slot_mean protected_detail_episode_shuffle p3_precision_delta_zero p3_temporal_delta_zero"
if [[ -z "${MODEL_PATH_MODES:-}" ]]; then
  case "${PROBE_PROFILE}" in
    full)
      MODEL_PATH_MODES="${FULL_MODEL_PATH_MODES}"
      ;;
    intent_effect_followup)
      MODEL_PATH_MODES="${INTENT_EFFECT_FOLLOWUP_MODEL_PATH_MODES}"
      ;;
    *)
      echo "[grounded-intent-effect-323-probe] unknown PROBE_PROFILE=${PROBE_PROFILE}; expected full or intent_effect_followup" >&2
      exit 2
      ;;
  esac
fi
export MODEL_PATH_MODES
echo "[grounded-intent-effect-323-probe] profile=${PROBE_PROFILE} modes=${MODEL_PATH_MODES}" >&2

exec bash "${SCRIPT_DIR}/run_v103_model_path_probe.sh" "$@"
