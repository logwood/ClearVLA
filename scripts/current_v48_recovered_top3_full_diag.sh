#!/usr/bin/env bash
# V48 recovered-top3 experiment with all currently useful diagnostics enabled.
#
# Enabled mechanisms:
# - direct condition residual with learned strength
# - t-gated noisy-action branch
# - recurrent layer-scan condition
#
# Enabled diagnostics:
# - zero-rollout-token consequence shortcut probe
#
# Kept off by default:
# - action-consequence self-condition. It changes the consequence-cell action
#   coordinate through a preview-action side path and can fight the main flow.
#
# Still disabled:
# - trajectory denoise/manifold
# - coefficient heads
# - block-action denoise
# - canvas cross-attention
# - serial writers
# - layer boost
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUT_DIR="${OUT_DIR:-runs/v48_recovered_top3_full_diag_b8}"

exec bash "${SCRIPT_DIR}/current_v48_recovered_top3.sh" \
  --action-consequence-self-condition "${ACTION_CONSEQUENCE_SELF_CONDITION:-0}" \
  --layer-zero-base-diagnostic "${LAYER_ZERO_BASE_DIAGNOSTIC:-1}" \
  "$@"
