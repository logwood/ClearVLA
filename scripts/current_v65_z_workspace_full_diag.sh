#!/usr/bin/env bash
# Z-primary MMDiT denoising with a typed, AdaLN-modulated evidence workspace.
# All non-z semantics are summarized once per refine step before action update.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUT_DIR="${OUT_DIR:-runs/v65_z_workspace_h${LATENT_CVAE_HORIZON_TOKENS:-24}_b8}"

exec bash "${SCRIPT_DIR}/current_v48_mmdit_deep_rollout_full_diag.sh" \
  --latent-cvae-horizon-tokens "${LATENT_CVAE_HORIZON_TOKENS:-24}" \
  --latent-cvae-mmdit-cond-update 0 \
  --latent-cvae-grad-clip "${LATENT_CVAE_GRAD_CLIP:-1.0}" \
  "$@"
