#!/usr/bin/env bash
# V69: keep V67 rollout/workspace semantics while making the redundant arm
# [absolute, delta] flow geometrically consistent. Noise is sampled as one
# temporally correlated native trajectory, encoded into the physical manifold,
# and every inference update is projected back onto the same support.
#
# Default rho is 0.0 (white native noise): step 1 of the plan isolates the
# manifold-consistency benefit before temporal correlation enters.  Effective
# temporal rank over the 24-step window vs rho (measured):
#   rho=0.00 -> 24.0/24   rho=0.70 -> 12.7/24
#   rho=0.90 ->  5.5/24   rho=0.95 ->  3.7/24  (near-collapse, avoid)
# Use current_v69b_arm_manifold_rho07.sh for the correlated-noise arm of the
# A/B; do not exceed rho ~0.9 -- scale matching beyond that costs rank while
# still missing the data delta scale by >3x.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUT_DIR="${OUT_DIR:-runs/v69_arm_manifold_rho${ARM_NOISE_TEMPORAL_RHO:-0.0}_b8}"

exec bash "${SCRIPT_DIR}/current_v67_identifiable_rollout.sh" \
  --arm-flow-mode manifold_native \
  --arm-noise-temporal-rho "${ARM_NOISE_TEMPORAL_RHO:-0.0}" \
  --arm-manifold-null-weight "${ARM_MANIFOLD_NULL_WEIGHT:-1.0}" \
  "$@"
