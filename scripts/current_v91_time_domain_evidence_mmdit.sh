#!/usr/bin/env bash
# V91 phase-1 migration: V48's useful native-time, distinct-block bottom path
# behind the current typed EvidenceBank interface.
#
# This entry point deliberately starts from the stable V65/V48 data and
# optimizer command, then replaces only the final action decoder. It does not
# enable the later hierarchical workspace/controller, spectral/DCT state,
# adaptive refine, micro controller, posterior, or learned dwell paths.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OUT_DIR="${OUT_DIR:-runs/v91_time_domain_evidence_mmdit}"
export LATENT_CVAE_MMDIT_DEPTH="${LATENT_CVAE_MMDIT_DEPTH:-3}"
export LATENT_CVAE_LAYER_SCAN="${LATENT_CVAE_LAYER_SCAN:-1}"
export LATENT_CVAE_LAYER_SCAN_ALPHA="${LATENT_CVAE_LAYER_SCAN_ALPHA:-0.2}"

printf '[v91] decoder=evidence_latent_mmdit_action depth=%s layer_scan=%s alpha=%s x_t=action_stream evidence_fusion=mmdit native_time=1 posterior=0 adaptive=0 spectral=0\n' \
  "${LATENT_CVAE_MMDIT_DEPTH}" \
  "${LATENT_CVAE_LAYER_SCAN}" \
  "${LATENT_CVAE_LAYER_SCAN_ALPHA}"

exec bash "${SCRIPT_DIR}/current_v65_z_workspace_full_diag.sh" \
  --final-action-decoder evidence_latent_mmdit_action \
  --latent-cvae-mmdit-depth "${LATENT_CVAE_MMDIT_DEPTH}" \
  --latent-cvae-mmdit-cond-update 0 \
  --latent-cvae-mmdit-residual-scale-max "${LATENT_CVAE_MMDIT_RESIDUAL_SCALE_MAX:-0.25}" \
  --latent-cvae-mmdit-evidence-scale "${LATENT_CVAE_MMDIT_EVIDENCE_SCALE:-1.0}" \
  --latent-cvae-mmdit-noisy-scale "${LATENT_CVAE_MMDIT_NOISY_SCALE:-1.0}" \
  --latent-cvae-layer-scan "${LATENT_CVAE_LAYER_SCAN}" \
  --latent-cvae-layer-scan-alpha "${LATENT_CVAE_LAYER_SCAN_ALPHA}" \
  --latent-cvae-inference-sample 0 \
  --latent-cvae-variational 0 \
  --latent-cvae-hierarchical-workspace 0 \
  --hierarchical-mmdit-spectral-state 0 \
  --arm-flow-mode legacy_independent \
  --gripper-field-mode legacy_handcrafted \
  --adaptive-cvae-function-adapters 0 \
  --adaptive-cvae-micro-control 0 \
  --adaptive-cvae-micro-refine-block 0 \
  --adaptive-cvae-direct-condition-residual 0 \
  --adaptive-cvae-condition-strength 0 \
  --action-consequence-self-condition 0 \
  "$@"
