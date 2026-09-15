#!/usr/bin/env bash
# Frozen-checkpoint matched causal probe for the complete V104 model path.
# The evaluator reads the serialized policy configuration and requires all
# three V104 structural contracts before assigning the V104 result schema.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CHECKPOINT="${CHECKPOINT:-runs/v104_sequential_bounded_flow_jepa/checkpoints/latest.pt}"
export DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-runs/diagnostics/v104_model_path}"
export BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-104}"
export RESULT_JSON="${RESULT_JSON:-${DIAGNOSTICS_DIR}/probe_v3_${PROBE_BATCHES:-10}b.json}"
export MODEL_PATH_PROBE_LABEL=v104
export MODEL_PATH_REQUIRED_CONTRACT=v104

exec bash "${SCRIPT_DIR}/run_v103_model_path_probe.sh" "$@"
