#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MAINLINE_CONFIG="${MAINLINE_CONFIG:-configs/mainline/object_intent_dynamics_323_rdt_shared_v1.json}"
exec bash "${SCRIPT_DIR}/train_rdt_multitask.sh" "$@"
