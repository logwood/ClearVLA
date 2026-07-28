#!/usr/bin/env bash
# Paired typed-address/action model-path probe for a completed V110 checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CHECKPOINT="${CHECKPOINT:-runs/v110_coordinate_typed_raw_jepa/checkpoints/latest.pt}"
export DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-runs/diagnostics/v110_model_path}"
export BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-110}"
export RESULT_JSON="${RESULT_JSON:-${DIAGNOSTICS_DIR}/probe_v9_${PROBE_BATCHES:-10}b.json}"
export MODEL_PATH_PROBE_LABEL=v110
export MODEL_PATH_REQUIRED_CONTRACT=v110

exec bash "${SCRIPT_DIR}/run_v103_model_path_probe.sh" "$@"
