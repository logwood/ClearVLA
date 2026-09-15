#!/usr/bin/env bash
# =============================================================================
# V56 B-SPLINE HEAD: keep the V54/V55 healthy denoise geometry, only change the
# arm output writer from a global DCT basis to a local clamped B-spline basis.
#
# This deliberately does NOT re-enable LATENT_CVAE_TRAJECTORY_DENOISE.  The
# recurrent refine state stays in full action-token space; only the final arm
# velocity is written through B-spline control coefficients.  The B-spline
# analysis operator is a ridge pseudo-inverse, so non-orthogonal control points
# are not treated as DCT-like coordinates.  Gripper value/delta remain
# per-horizon-token outputs.
#
# No extra smoothing loss is added here.  Smoothness/locality comes only from
# the output parameterization; primary training still uses physical action-space
# flow/decode losses.
# =============================================================================
set -euo pipefail

export OUT_DIR=${OUT_DIR:-runs/v56_bspline_head_b8}

# Keep the process manifold off.  This is the root-cause kill-switch from V54.
export LATENT_CVAE_TRAJECTORY_DENOISE=${LATENT_CVAE_TRAJECTORY_DENOISE:-0}

# Output-only arm coefficient parameterization.
export LATENT_CVAE_ARM_COEFF_OUTPUT=${LATENT_CVAE_ARM_COEFF_OUTPUT:-1}
export LATENT_CVAE_ARM_COEFF_POINTS=${LATENT_CVAE_ARM_COEFF_POINTS:-8}
export LATENT_CVAE_ARM_COEFF_BASIS=${LATENT_CVAE_ARM_COEFF_BASIS:-bspline}
export LATENT_CVAE_ARM_COEFF_DEGREE=${LATENT_CVAE_ARM_COEFF_DEGREE:-2}
export LATENT_CVAE_ARM_COEFF_RIDGE=${LATENT_CVAE_ARM_COEFF_RIDGE:-1e-2}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/current_v54_rebase.sh" "$@"
