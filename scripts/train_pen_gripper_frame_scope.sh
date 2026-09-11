#!/usr/bin/env bash
# One training-objective scope control; no model or deployed-codec changes.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MAINLINE_CONFIG="${MAINLINE_CONFIG:-configs/mainline/object_intent_dynamics_323_pen_gripper_frame_scope_v1.json}"
export OUT_DIR="${OUT_DIR:-/data/senwang/clearvla/experiments/pen/20260911-pen-gripper-frame-scope}"
exec bash "${SCRIPT_DIR}/train_mainline.sh" "$@"
