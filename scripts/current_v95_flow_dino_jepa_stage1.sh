#!/usr/bin/env bash
# Explicit entry point for the new V95 top-representation Stage1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export V95_TRAINING_STAGE=stage1
export OUT_DIR="${OUT_DIR:-runs/v95_flow_dino_jepa_stage1}"

exec bash "${SCRIPT_DIR}/current_v95_flow_dino_jepa.sh" "$@"
