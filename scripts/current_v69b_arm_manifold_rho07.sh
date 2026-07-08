#!/usr/bin/env bash
# V69b: correlated-noise arm of the manifold A/B.  rho=0.7 keeps effective
# temporal rank at 12.7/24 (vs 3.7/24 at rho=0.95) while cutting delta-channel
# noise std from 1.41 to 0.77.  Run AFTER v69 (rho=0) so the manifold benefit
# and the correlation benefit stay separately attributable.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ARM_NOISE_TEMPORAL_RHO="${ARM_NOISE_TEMPORAL_RHO:-0.7}"
export OUT_DIR="${OUT_DIR:-runs/v69b_arm_manifold_rho${ARM_NOISE_TEMPORAL_RHO}_b8}"

exec bash "${SCRIPT_DIR}/current_v69_arm_manifold_flow.sh" "$@"
