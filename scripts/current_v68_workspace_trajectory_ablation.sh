#!/usr/bin/env bash
# V68 diagnostic: V67 identifiable rollout with the full-resolution trajectory
# canvas removed from workspace values. Noisy/action tokens remain in the MMDiT
# action stream and can still select layer/transition/rollout evidence.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUT_DIR="${OUT_DIR:-runs/v68_no_workspace_trajectory_h${LATENT_CVAE_HORIZON_TOKENS:-24}_b8}"

exec bash "${SCRIPT_DIR}/current_v67_identifiable_rollout.sh" \
  --latent-cvae-workspace-trajectory-source 0 \
  "$@"
