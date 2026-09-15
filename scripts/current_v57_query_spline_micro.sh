#!/usr/bin/env bash
# =============================================================================
# V57 QUERY-SPLINE MICRO: fix the V56 coordinate mismatch.
#
# B-spline remains an output/action-space writer, not a hidden-token compressor:
# learned control queries cross-attend the full refined horizon tokens, then a
# direction+magnitude head writes physical coefficients that are expanded by the
# spline basis.  The recurrent refine/micro stack keeps the full token stream
# and is audited as base-spline proposal -> final refined velocity.
#
# No process manifold, no smoothing loss, no clean-target shortcut.  The first
# run intentionally writes the full typed physical velocity (arm_abs, arm_delta,
# grip_value, grip_delta) through the same query-spline writer, while proposal
# residual coefficient supervision stays arm-only.
# =============================================================================
set -euo pipefail

export OUT_DIR=${OUT_DIR:-runs/v57_query_spline_micro_b8}

# Keep the failed trajectory-process manifold off.
export LATENT_CVAE_TRAJECTORY_DENOISE=${LATENT_CVAE_TRAJECTORY_DENOISE:-0}

# Output/action-space spline writer.
export LATENT_CVAE_ARM_COEFF_OUTPUT=${LATENT_CVAE_ARM_COEFF_OUTPUT:-1}
export LATENT_CVAE_ARM_COEFF_POINTS=${LATENT_CVAE_ARM_COEFF_POINTS:-8}
export LATENT_CVAE_ARM_COEFF_BASIS=${LATENT_CVAE_ARM_COEFF_BASIS:-bspline}
export LATENT_CVAE_ARM_COEFF_DEGREE=${LATENT_CVAE_ARM_COEFF_DEGREE:-2}
export LATENT_CVAE_ARM_COEFF_RIDGE=${LATENT_CVAE_ARM_COEFF_RIDGE:-1e-2}
export LATENT_CVAE_ARM_COEFF_WRITER=${LATENT_CVAE_ARM_COEFF_WRITER:-query_direction}

# First trial follows the user's hypothesis: one low-frequency typed-physical
# writer.  Set this to 0 if gripper/event metrics degrade.
export LATENT_CVAE_COEFF_INCLUDE_GRIPPER=${LATENT_CVAE_COEFF_INCLUDE_GRIPPER:-1}
export LATENT_CVAE_COEFF_MAGNITUDE_GROUPS=${LATENT_CVAE_COEFF_MAGNITUDE_GROUPS:-2}
export LATENT_CVAE_SPLINE_BASE_DIAGNOSTICS=${LATENT_CVAE_SPLINE_BASE_DIAGNOSTICS:-1}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/current_v54_rebase.sh" "$@"
