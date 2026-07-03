#!/usr/bin/env bash
# =============================================================================
# V54 REBASE: v48-proven base + V53 components, v51 baggage dropped.
#
# Rationale (2026-07-03 forensic audit):
#   v48 is the last configuration proven to learn (pflow 0.39 -> 0.11 by
#   epoch 2 -> 0.039 by epoch 8; full_mse 0.0068).  v51 introduced the
#   trajectory manifold with a position-collapse bug (adjacent-position
#   distinguishability ~1.7%), dropped the v50 per-channel Gaussian bridge,
#   and silently broke the eval noise distribution.  Everything since was
#   built and tuned on that broken substrate.
#
# This rebase:
#   TIER 0 (correctness, always on): block bridge back on; eval noise and
#     codec-boundary fixes are in code (V53.5) and need no flags here.
#   TIER 1 (V53 components, in): boosting layer contracts, depth-scan
#     condition, monotonic layer routing, canvas cross-attention (C1),
#     serial writers (C2), zero-base diagnostic, gate/keep diagnostics.
#   TIER 2 (recalibrated): x_t t-gate softened (min 0.10, p=1.0) and hinge
#     relaxed (0.50 @ 0.02) -- the old values were tuned against a
#     position-blind model whose x_t hunger was partly legitimate.
#   TIER 3 (dropped): the v51 trajectory manifold itself (TRAJECTORY_DENOISE=0).
#     Its correct version (pos-exempt) stays implemented; re-enable A/B with:
#       LATENT_CVAE_TRAJECTORY_DENOISE=1 bash scripts/current_v54_rebase.sh
#     (POS_EXEMPT is already 1 downstream.)
#
# GATE CRITERION: train pflow must drop below ~0.15 by epoch 2 (v48 did 0.11;
#   allow headroom for the softened t-gate).  If it does not, peel back
#   Tier 1 components one by one before touching anything else.
# =============================================================================
set -euo pipefail

export OUT_DIR=${OUT_DIR:-runs/v54_rebase_b8}

# --- Tier 3: v51 trajectory manifold OFF (kill-switch documented above) ---
export LATENT_CVAE_TRAJECTORY_DENOISE=${LATENT_CVAE_TRAJECTORY_DENOISE:-0}

# --- Tier 2: anti-shortcut constraints, recalibrated for a healthy system ---
export LATENT_CVAE_NOISY_GATE=${LATENT_CVAE_NOISY_GATE:-1}
export LATENT_CVAE_NOISY_GATE_MIN=${LATENT_CVAE_NOISY_GATE_MIN:-0.10}
export LATENT_CVAE_NOISY_GATE_POWER=${LATENT_CVAE_NOISY_GATE_POWER:-1.0}
export LATENT_CVAE_NOISY_RATIO_MAX=${LATENT_CVAE_NOISY_RATIO_MAX:-0.50}
export LATENT_CVAE_NOISY_RATIO_WEIGHT=${LATENT_CVAE_NOISY_RATIO_WEIGHT:-0.02}

# --- Tier 1: V53-C structural components join the party ---
export LATENT_CVAE_CANVAS_CROSS_ATTENTION=${LATENT_CVAE_CANVAS_CROSS_ATTENTION:-1}
export ADAPTIVE_CVAE_SERIAL_WRITERS=${ADAPTIVE_CVAE_SERIAL_WRITERS:-1}

# --- v48 fidelity: function adapters were ON in the last working config;
#     their apparent death (cfunc=0.000) was observed only on the broken
#     substrate and gets a fresh trial here ---
export ADAPTIVE_CVAE_FUNCTION_ADAPTERS=${ADAPTIVE_CVAE_FUNCTION_ADAPTERS:-1}

# Everything else (boost, depth-scan, monotonic routing, zero-base diag,
# block bridge, layer detach/grad-scale, refine steps 6, lr 8e-5, epochs 8,
# v52 proposal-residual weights) inherits the v53 script defaults, which
# already match the v48 anchors where it matters.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/current_v53_full.sh" "$@"
