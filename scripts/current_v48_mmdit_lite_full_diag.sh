#!/usr/bin/env bash
# V48 baseline with MMDiT-lite taking over the adaptive CVAE refine block.
#
# The CVAE prior/posterior, progress memory, layer routing, and policy losses
# are unchanged.  When --latent-cvae-mmdit-decoder=1 is set, each adaptive
# refine step uses an MMDiT action-condition block instead of the legacy refine
# or micro controller update.  Noisy action enters as masked condition tokens,
# not as a direct residual stream into the velocity head.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUT_DIR="${OUT_DIR:-runs/v48_mmdit_lite_full_diag_b8}"

exec bash "${SCRIPT_DIR}/current_v48_justok.sh" \
  --latent-cvae-mmdit-decoder "${LATENT_CVAE_MMDIT_DECODER:-1}" \
  --latent-cvae-mmdit-depth "${LATENT_CVAE_MMDIT_DEPTH:-3}" \
  --latent-cvae-mmdit-cond-update "${LATENT_CVAE_MMDIT_COND_UPDATE:-0}" \
  --latent-cvae-mmdit-noisy-causal "${LATENT_CVAE_MMDIT_NOISY_CAUSAL:-1}" \
  --latent-cvae-noisy-gate "${LATENT_CVAE_NOISY_GATE:-1}" \
  --latent-cvae-noisy-gate-min "${LATENT_CVAE_NOISY_GATE_MIN:-0.08}" \
  --latent-cvae-noisy-gate-power "${LATENT_CVAE_NOISY_GATE_POWER:-1.0}" \
  --latent-cvae-layer-scan "${LATENT_CVAE_LAYER_SCAN:-1}" \
  --latent-cvae-layer-scan-alpha "${LATENT_CVAE_LAYER_SCAN_ALPHA:-0.2}" \
  --adaptive-cvae-function-adapters 0 \
  --adaptive-cvae-micro-control 0 \
  --adaptive-cvae-direct-condition-residual 0 \
  --adaptive-cvae-condition-strength 0 \
  --action-consequence-self-condition 0 \
  --layer-zero-base-diagnostic "${LAYER_ZERO_BASE_DIAGNOSTIC:-1}" \
  "$@"
