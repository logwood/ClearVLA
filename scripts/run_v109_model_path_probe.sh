#!/usr/bin/env bash
# Paired progressive-address/action model-path probe for a completed V109 checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CHECKPOINT="${CHECKPOINT:-runs/v109_progressive_grounding_address_flow_jepa/checkpoints/latest.pt}"
export DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-runs/diagnostics/v109_model_path}"
export BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-109}"
export RESULT_JSON="${RESULT_JSON:-${DIAGNOSTICS_DIR}/probe_v8_${PROBE_BATCHES:-10}b.json}"
export MODEL_PATH_PROBE_LABEL=v109
export MODEL_PATH_REQUIRED_CONTRACT=v109

exec bash "${SCRIPT_DIR}/run_v103_model_path_probe.sh" "$@"
