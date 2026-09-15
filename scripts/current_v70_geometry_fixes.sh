#!/usr/bin/env bash
# =============================================================================
# V70: geometry & instrumentation fixes on top of the v69 arm-manifold A/B.
#
# Unconditional correctness fixes (in code, no flags; v69 reproducible via
# the v69-epoch1-baseline git tag):
#   - H3: every training-time clean estimate is manifold-projected before
#     decode, matching deployment geometry and closing the arm-null
#     arbitrage channel through the decode blend.
#   - Codec consistency arithmetic exempted from bf16 autocast; the
#     afmproj/afmnoise hygiene canaries should now read ~1e-12.
#   - Metric overhaul: null gauges get stable denominators (null_rms +
#     null_output_fraction); gripper_arm_flow_ratio is target-normalized;
#     xratio retired in favor of volpar (volume parity) and xinfl
#     (attention x volume influence ratio); gripper null decomposed by
#     event/hold steps (gfnehr) as the H4 timing-uncertainty test.
#
# Flag-gated structural change (ON here):
#   - Noisy condition tokens LayerNormed + t-gate moved to an additive
#     log g(t) attention-logit bias.  Volume degree of freedom closed;
#     attention shares become honest influence readings.
#
# Expected reading changes vs v69: volpar pins near 1.0 by construction;
# mdna becomes directly comparable across runs; afmnrms/gfmnrms replace the
# denominator-collapsing null ratios (healthy range: compare against data
# delta std ~0.03 and the fp32 floor).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ARM_NOISE_TEMPORAL_RHO="${ARM_NOISE_TEMPORAL_RHO:-0.0}"
export OUT_DIR="${OUT_DIR:-runs/v70_geometry_fixes_rho${ARM_NOISE_TEMPORAL_RHO}_b8}"

exec bash "${SCRIPT_DIR}/current_v69_arm_manifold_flow.sh" \
  --latent-cvae-mmdit-noisy-logit-gate 1 \
  "$@"
