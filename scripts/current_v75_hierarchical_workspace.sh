#!/usr/bin/env bash
# =============================================================================
# V75: hierarchical evidence workspace.
#
# Structure per MMDiT refine step:
#   raw EvidenceBank -> temporary low slots -> persistent stage content
#                                      \----> [low, stage, noisy] -> action
#
# Hard boundaries:
#   - action/noisy tokens never enter workspace retrieval or stage update;
#   - stage affects low reads only through selector queries and role logits;
#   - low values come only from learned low seeds plus raw EvidenceBank values;
#   - stage role identity and recurrent stage content remain separate tensors;
#   - V74B time broadcast, controller, route-time query, and external
#     progress/capsule/layer routing are disabled on this mainline.
#
# Key diagnostics:
#   mdla/mdsa  : MMDiT attention to low/stage groups
#   mdle/mdse  : length-fair low/stage enrichment
#   hlsel      : effective stage slots used by low selectors
#   hsrole/*   : fixed stage-role geometry
#   hscont/*   : dynamic stage-content geometry
#   hsrcos     : role/content alignment (collapse warning)
#   hsupd/hsret: recurrent stage update and retain behavior
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ARM_NOISE_TEMPORAL_RHO="${ARM_NOISE_TEMPORAL_RHO:-0.0}"
export STAGE_SLOTS="${STAGE_SLOTS:-6}"
export STAGE_PROMOTE_SCALE_INIT="${STAGE_PROMOTE_SCALE_INIT:-0.05}"
export OUT_DIR="${OUT_DIR:-runs/v75_hierarchical_workspace_s${STAGE_SLOTS}_rho${ARM_NOISE_TEMPORAL_RHO}_b8}"

exec bash "${SCRIPT_DIR}/current_v74a_memory_bank_refactor.sh" \
  --latent-cvae-hierarchical-workspace 1 \
  --latent-cvae-stage-slots "${STAGE_SLOTS}" \
  --latent-cvae-stage-promote-scale-init "${STAGE_PROMOTE_SCALE_INIT}" \
  --latent-cvae-workspace-global-sources 1 \
  --latent-cvae-workspace-layer-source 1 \
  --latent-cvae-workspace-progress-value 0 \
  --latent-cvae-workspace-noisy-query 0 \
  --latent-cvae-workspace-time-state 0 \
  --latent-cvae-workspace-slot-time-state 0 \
  --latent-cvae-workspace-controller 0 \
  --adaptive-cvae-progress-memory 0 \
  --adaptive-cvae-layer-routing 0 \
  --adaptive-cvae-context-capsules 0 \
  --adaptive-cvae-route-time-query 0 \
  "$@"
