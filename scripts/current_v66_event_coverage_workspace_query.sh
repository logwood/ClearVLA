#!/usr/bin/env bash
# V66: preserve the V65 single workspace path while reallocating the existing
# one-dimensional gripper loss budget toward sparse transitions. Workspace
# evidence queries also inspect x_t, but x_t never enters workspace values.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUT_DIR="${OUT_DIR:-runs/v66_event_coverage_workspace_query_h${LATENT_CVAE_HORIZON_TOKENS:-24}_b8}"
export GRIPPER_FM_EVENT_BOOST="${GRIPPER_FM_EVENT_BOOST:-6.0}"

exec bash "${SCRIPT_DIR}/current_v65_z_workspace_full_diag.sh" \
  --latent-cvae-workspace-noisy-query "${LATENT_CVAE_WORKSPACE_NOISY_QUERY:-1}" \
  "$@"
