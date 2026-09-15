#!/usr/bin/env bash
# Paired functional-ownership/action probe for a completed V111 checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CHECKPOINT="${CHECKPOINT:-runs/v111_structured_ownership_bottleneck/checkpoints/latest.pt}"
export DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-runs/diagnostics/v111_model_path}"
export BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-111}"
export RESULT_JSON="${RESULT_JSON:-${DIAGNOSTICS_DIR}/probe_v10_${PROBE_BATCHES:-10}b.json}"
export MODEL_PATH_PROBE_LABEL=v111
export MODEL_PATH_REQUIRED_CONTRACT=v111

exec bash "${SCRIPT_DIR}/run_v103_model_path_probe.sh" "$@"
