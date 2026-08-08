#!/usr/bin/env bash
# Paired pre-value ownership/action probe for a completed V112 checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CHECKPOINT="${CHECKPOINT:-runs/v112_long_run_01/checkpoints/latest.pt}"
export DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-runs/diagnostics/v112_model_path}"
export BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-112}"
export RESULT_JSON="${RESULT_JSON:-${DIAGNOSTICS_DIR}/probe_v112_${PROBE_BATCHES:-10}b.json}"
export MODEL_PATH_PROBE_LABEL=v112
export MODEL_PATH_REQUIRED_CONTRACT=v112
# V112 has an unusually valuable but expensive frozen checkpoint.  The
# default subset answers three concrete questions from its completed log:
# whether W helped, whether current fine evidence reached action, and whether
# the long-horizon condition lanes were used. Set MODEL_PATH_MODES=all for the
# historical exhaustive contract.
export MODEL_PATH_MODES="${MODEL_PATH_MODES:-policy_zero policy_temporal_shuffle world_residual_zero world_residual_anchor_shuffle flow_zero flow_spatial_shuffle raw_value_zero raw_value_spatial_shuffle source_raw_match_spatial_shuffle dino_key_spatial_shuffle literal_current_rgb_zero future_transport_neutral semantic_owner_zero appearance_owner_zero geometry_owner_zero goal_zero goal_episode_shuffle action_history_zero action_history_episode_shuffle phase_belief_zero address_g1_zero address_g2_zero address_g3_zero interval_stage_zero g3_delta_zero w1_delta_zero w2_delta_zero w3_delta_zero world_to_policy_delta_zero p1_delta_zero p2_delta_zero p2_delta_episode_shuffle protected_detail_zero}"

exec bash "${SCRIPT_DIR}/run_v103_model_path_probe.sh" "$@"
