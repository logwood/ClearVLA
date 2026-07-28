#!/usr/bin/env bash
# Paired online-address/action model-path probe for a completed V108 checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CHECKPOINT="${CHECKPOINT:-runs/v108_online_horizon_address_flow_jepa/checkpoints/latest.pt}"
export DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-runs/diagnostics/v108_model_path}"
export BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-108}"
export RESULT_JSON="${RESULT_JSON:-${DIAGNOSTICS_DIR}/probe_v7_${PROBE_BATCHES:-10}b.json}"
export MODEL_PATH_PROBE_LABEL=v108
export MODEL_PATH_REQUIRED_CONTRACT=v108

exec bash "${SCRIPT_DIR}/run_v103_model_path_probe.sh" "$@"
