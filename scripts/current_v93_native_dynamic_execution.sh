#!/usr/bin/env bash
# V93: V92 native execution plane with typed current/next block dispatch.
#
# The controller selects a physical candidate value field. The host block
# still owns residual amplitude, and nested contraction only removes ordered
# operator groups. Warmup remains the exact V92 fixed path.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUT_DIR="${OUT_DIR:-runs/v93_native_dynamic_execution}"

exec bash "${SCRIPT_DIR}/current_v92_native_execution_controller.sh" \
  --latent-cvae-mmdit-dynamic-block-route 1 \
  "$@"
