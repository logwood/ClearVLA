#!/usr/bin/env bash
# =============================================================================
# V71: isolation run — v70 unconditional fixes WITHOUT the noisy LN+logit gate.
#
# No code changes vs v70; this is a pure flag flip.  Code state should be the
# same commit that ran v70a.
#
# What is active (unconditional, in code):
#   - H3: training-time clean estimates manifold-projected before decode
#     (arm-only bite: gripper decode is synthesis-based and already
#     null-invariant).
#   - Codec autocast exemption (afmproj/gfmproj at fp32 zero, ~1e-14).
#   - Full V70 instrumentation: null_rms / null_output_fraction,
#     target-normalized gfar, volpar / xinfl, gfnehr event/hold split.
#
# What is OFF (the reverted structural experiment):
#   - latent-cvae-mmdit-noisy-logit-gate 0 -> noisy branch returns to the
#     v69 multiplicative t-gate; per-token value volume is FREE again.
#     v70a showed the LayerNorm force-fed x_t at ~2.5x the revealed-optimal
#     volume (mdnt equilibrium ~8-9 vs pinned 22.6), costing train flow on
#     all channels and +12% gripper val rmse.
#
# Comparisons this run settles (all at rho=0, paired seed):
#   v69a vs v71 : effect of H3 + autocast + instruments alone.
#                 If val ~= v69a  -> LN convicted, H3 exonerated.
#                 If val worse    -> H3 honesty-tax is real; decide if the
#                                    train/deploy geometry unification is
#                                    worth it.
#   v70a vs v71 : effect of the LN+logit gate in isolation.
#
# Expected reading changes vs v70a:
#   - volpar unpins (expect ~0.1 early E1, drifting up as mdnt grows 1.6->9).
#   - xinfl drops back to O(1); afmproj/gfmproj stay ~1e-14.
#   - gfnehr becomes the UNCONTAMINATED H4 gauge (v70a's was tainted by the
#     LN injury to x_t timing evidence; re-baseline H4 from this run).
#   - sample_arm_null_preproject_rate: read against v69a's natural decay
#     curve (0.315 -> 0.178 -> 0.140 over E1-3), not as a raw level.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ARM_NOISE_TEMPORAL_RHO="${ARM_NOISE_TEMPORAL_RHO:-0.0}"
export OUT_DIR="${OUT_DIR:-runs/v71_h3_isolation_rho${ARM_NOISE_TEMPORAL_RHO}_b8}"

exec bash "${SCRIPT_DIR}/current_v69_arm_manifold_flow.sh" \
  --latent-cvae-mmdit-noisy-logit-gate 0 \
  "$@"
