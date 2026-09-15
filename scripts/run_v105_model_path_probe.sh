#!/usr/bin/env bash
# Frozen-checkpoint matched causal probe for the complete V105 model path.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CHECKPOINT="${CHECKPOINT:-runs/v105_horizon_addressed_flow_jepa/checkpoints/latest.pt}"
export DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-runs/diagnostics/v105_model_path}"
export BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-105}"
export RESULT_JSON="${RESULT_JSON:-${DIAGNOSTICS_DIR}/probe_v4_${PROBE_BATCHES:-10}b.json}"
export MODEL_PATH_PROBE_LABEL=v105
export MODEL_PATH_REQUIRED_CONTRACT=v105

exec bash "${SCRIPT_DIR}/run_v103_model_path_probe.sh" "$@"
