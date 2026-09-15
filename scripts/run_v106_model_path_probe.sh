#!/usr/bin/env bash
# Paired causal model-path probe for a completed V106 checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CHECKPOINT="${CHECKPOINT:-runs/v106_interval_stage_flow_jepa/checkpoints/latest.pt}"
export DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-runs/diagnostics/v106_model_path}"
export BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-106}"
export RESULT_JSON="${RESULT_JSON:-${DIAGNOSTICS_DIR}/probe_v5_${PROBE_BATCHES:-10}b.json}"
export MODEL_PATH_PROBE_LABEL=v106
export MODEL_PATH_REQUIRED_CONTRACT=v106

exec bash "${SCRIPT_DIR}/run_v103_model_path_probe.sh" "$@"
