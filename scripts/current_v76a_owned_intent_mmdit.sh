#!/usr/bin/env bash
# =============================================================================
# V76A / phase 1: fix semantic ownership before adaptive depth.
#
# This run changes two coupled structures and deliberately leaves depth fixed:
#   1. Replace CVAE prior/posterior z with deterministic global/stage/read
#      intent contracts compiled from ordered layer and typed world evidence.
#   2. Give the workspace manager ownership of retrieval/promotion only. The
#      action decoder consumes noisy -> stage -> low through dedicated serial
#      functions; no shared condition-group market or manager output gate exists.
#
# Evidence values have five physically isolated roles:
#   geom=(deploy-safe proposal, rollout), transition, event, state, layer.
# Pure global summaries feed intent/stage values; dynamic scan summaries are
# selector-only and cannot re-enter the workspace as anonymous value tokens.
# Action/noisy tokens are read-only MMDiT KV and never enter workspace values.
# V76B exhaustion/random-dwell/adaptive depth is intentionally not active.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HIERARCHICAL_MMDIT_DEPTH="${HIERARCHICAL_MMDIT_DEPTH:-3}"
export HIERARCHICAL_MMDIT_REFINE_STEPS="${HIERARCHICAL_MMDIT_REFINE_STEPS:-3}"
export HIERARCHICAL_MMDIT_LOW_SLOTS="${HIERARCHICAL_MMDIT_LOW_SLOTS:-25}"
export HIERARCHICAL_MMDIT_STAGE_SLOTS="${HIERARCHICAL_MMDIT_STAGE_SLOTS:-6}"
export OUT_DIR="${OUT_DIR:-runs/v76a_serial_owned_mmdit_d${HIERARCHICAL_MMDIT_DEPTH}_s${HIERARCHICAL_MMDIT_REFINE_STEPS}_b8}"

if (( HIERARCHICAL_MMDIT_LOW_SLOTS < 5 || HIERARCHICAL_MMDIT_LOW_SLOTS % 5 != 0 )); then
  printf 'HIERARCHICAL_MMDIT_LOW_SLOTS must be a positive multiple of five (got %s)\n' \
    "${HIERARCHICAL_MMDIT_LOW_SLOTS}" >&2
  exit 2
fi

exec bash "${SCRIPT_DIR}/current_v75_hierarchical_workspace.sh" \
  --final-action-decoder hierarchical_mmdit_action \
  --hierarchical-mmdit-depth "${HIERARCHICAL_MMDIT_DEPTH}" \
  --hierarchical-mmdit-refine-steps "${HIERARCHICAL_MMDIT_REFINE_STEPS}" \
  --hierarchical-mmdit-low-slots "${HIERARCHICAL_MMDIT_LOW_SLOTS}" \
  --hierarchical-mmdit-stage-slots "${HIERARCHICAL_MMDIT_STAGE_SLOTS}" \
  --hierarchical-mmdit-ffn-expansion 2.0 \
  --hierarchical-mmdit-layer-grad-scale 0.0 \
  --hierarchical-mmdit-source-grad-scale 0.0 \
  --hierarchical-mmdit-consequence-scale-init 0.10 \
  --hierarchical-mmdit-consequence-scale-max 0.50 \
  --hierarchical-mmdit-noisy-causal 1 \
  --hierarchical-mmdit-noisy-gate-min 0.05 \
  --hierarchical-mmdit-noisy-gate-power 1.5 \
  --hierarchical-mmdit-stage-promote-scale-init 0.05 \
  --hierarchical-mmdit-output-init-std 1e-3 \
  --hierarchical-mmdit-residual-scale-max 0.20 \
  --hierarchical-mmdit-output-contract 0 \
  --hierarchical-mmdit-noisy-market-bias 0 \
  --hierarchical-mmdit-noisy-gate-mode 0 \
  --latent-cvae-variational 0 \
  --latent-cvae-z-probe 0 \
  --latent-cvae-mmdit-decoder 0 \
  --latent-cvae-hierarchical-workspace 0 \
  --latent-cvae-workspace-global-sources 0 \
  --latent-cvae-workspace-layer-source 0 \
  --latent-cvae-workspace-progress-value 0 \
  --latent-cvae-workspace-noisy-query 0 \
  --latent-cvae-workspace-time-state 0 \
  --latent-cvae-workspace-controller 0 \
  --adaptive-cvae-progress-memory 0 \
  --adaptive-cvae-layer-routing 0 \
  --adaptive-cvae-context-capsules 0 \
  --adaptive-cvae-route-time-query 0 \
  --action-consequence-self-condition 0 \
  --latent-cvae-kl-weight 0 \
  --latent-cvae-posterior-recon-weight 0 \
  --latent-cvae-adaptive-regularizer-weight 0 \
  --latent-cvae-adaptive-route-entropy-weight 0 \
  --latent-cvae-micro-supervision-weight 0 \
  --latent-cvae-micro-event-weight 0 \
  --latent-cvae-micro-monotonic-weight 0 \
  --latent-cvae-micro-weight-kl-weight 0 \
  --latent-cvae-micro-coverage-smooth-weight 0 \
  --latent-cvae-micro-coverage-floor-weight 0 \
  "$@"
