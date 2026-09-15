#!/usr/bin/env bash
# V67: keep V66's normalized gripper event emphasis, disable the inconclusive
# noisy workspace query by default, and make rollout residuals identifiable by
# fixing the action-independent baseline at the physical no-change origin.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUT_DIR="${OUT_DIR:-runs/v67_identifiable_rollout_event_h${LATENT_CVAE_HORIZON_TOKENS:-24}_b8}"
export GRIPPER_FM_EVENT_BOOST="${GRIPPER_FM_EVENT_BOOST:-6.0}"
export LATENT_CVAE_WORKSPACE_NOISY_QUERY="${LATENT_CVAE_WORKSPACE_NOISY_QUERY:-0}"

exec bash "${SCRIPT_DIR}/current_v66_event_coverage_workspace_query.sh" \
  --controlled-base-mode "${CONTROLLED_BASE_MODE:-fixed_zero}" \
  "$@"
