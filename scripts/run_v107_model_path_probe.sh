#!/usr/bin/env bash
# Paired causal model-path probe for a completed V107 checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CHECKPOINT="${CHECKPOINT:-runs/v107_complete_top_path_flow_jepa/checkpoints/latest.pt}"
export DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-runs/diagnostics/v107_model_path}"
export BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-107}"
export RESULT_JSON="${RESULT_JSON:-${DIAGNOSTICS_DIR}/probe_v6_${PROBE_BATCHES:-10}b.json}"
export MODEL_PATH_PROBE_LABEL=v107
export MODEL_PATH_REQUIRED_CONTRACT=v107

exec bash "${SCRIPT_DIR}/run_v103_model_path_probe.sh" "$@"
