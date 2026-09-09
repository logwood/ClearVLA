#!/usr/bin/env bash
# Isolated Pen training-only persistence control; one ordinary fresh run.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MAINLINE_CONFIG="${MAINLINE_CONFIG:-configs/mainline/object_intent_dynamics_323_pen_anchored_v1.json}"
export OUT_DIR="${OUT_DIR:-/data/senwang/clearvla/experiments/pen/20260909-pen-anchored-persistence}"
exec bash "${SCRIPT_DIR}/train_mainline.sh" "$@"
