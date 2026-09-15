#!/usr/bin/env bash
# MMDiT-lite experiment with the rollout/consequence condition kept forward
# visible but prevented from absorbing the final action loss through the trunk.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUT_DIR="${OUT_DIR:-runs/v48_mmdit_rollout_isolated_full_diag_b8}"

# Preserve the legacy default for older entry points, while allowing V94's
# explicit LATENT_CVAE_TRANSITION_DETACH=0 to survive this wrapper chain.
exec bash "${SCRIPT_DIR}/current_v48_mmdit_lite_full_diag.sh" \
  --latent-cvae-layer-detach 1 \
  --latent-cvae-layer-grad-scale 0.0 \
  --latent-cvae-transition-detach "${LATENT_CVAE_TRANSITION_DETACH:-1}" \
  --latent-cvae-condition-source-norm "${LATENT_CVAE_CONDITION_SOURCE_NORM:-1}" \
  --latent-cvae-bounded-consequence-fusion "${LATENT_CVAE_BOUNDED_CONSEQUENCE_FUSION:-1}" \
  --latent-cvae-consequence-scale-init "${LATENT_CVAE_CONSEQUENCE_SCALE_INIT:-0.10}" \
  --latent-cvae-consequence-scale-max "${LATENT_CVAE_CONSEQUENCE_SCALE_MAX:-0.50}" \
  --layer-zero-base-diagnostic "${LAYER_ZERO_BASE_DIAGNOSTIC:-1}" \
  "$@"
