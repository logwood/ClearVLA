#!/usr/bin/env bash
# V48 baseline with the first three recovered mechanisms enabled:
# 1) direct condition residual with learned strength,
# 2) t-gated noisy-action branch,
# 3) recurrent layer-scan condition.
#
# This still does not enable trajectory denoise/manifold, coefficient heads,
# block-action denoise, canvas cross-attention, serial writers, or layer boost.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUT_DIR="${OUT_DIR:-runs/v48_recovered_top3_b8}"

exec bash "${SCRIPT_DIR}/current_v48_justok.sh" \
  --adaptive-cvae-direct-condition-residual "${ADAPTIVE_CVAE_DIRECT_CONDITION_RESIDUAL:-1}" \
  --adaptive-cvae-condition-strength "${ADAPTIVE_CVAE_CONDITION_STRENGTH:-1}" \
  --adaptive-cvae-condition-strength-init "${ADAPTIVE_CVAE_CONDITION_STRENGTH_INIT:-0.35}" \
  --adaptive-cvae-condition-strength-min "${ADAPTIVE_CVAE_CONDITION_STRENGTH_MIN:-0.03}" \
  --adaptive-cvae-condition-strength-max "${ADAPTIVE_CVAE_CONDITION_STRENGTH_MAX:-1.5}" \
  --latent-cvae-noisy-gate "${LATENT_CVAE_NOISY_GATE:-1}" \
  --latent-cvae-noisy-gate-min "${LATENT_CVAE_NOISY_GATE_MIN:-0.10}" \
  --latent-cvae-noisy-gate-power "${LATENT_CVAE_NOISY_GATE_POWER:-1.0}" \
  --latent-cvae-layer-scan "${LATENT_CVAE_LAYER_SCAN:-1}" \
  --latent-cvae-layer-scan-alpha "${LATENT_CVAE_LAYER_SCAN_ALPHA:-0.2}" \
  "$@"
