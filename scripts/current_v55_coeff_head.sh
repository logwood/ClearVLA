#!/usr/bin/env bash
# =============================================================================
# V55 COEFF HEAD: keep the V54 healthy denoise geometry, only change the arm
# output writer.
#
# This deliberately does NOT re-enable LATENT_CVAE_TRAJECTORY_DENOISE.  The
# recurrent refine state stays in full action-token space; only the final arm
# velocity is written through an orthonormal DCT coefficient head.  Gripper
# value/delta remain per-horizon-token outputs, so event-like signals do not
# enter a smooth trajectory basis.
#
# No extra smoothing loss is added here.  The only prior is the output
# parameterization itself; primary training still uses physical action-space
# flow/decode losses.
# =============================================================================
set -euo pipefail

export OUT_DIR=${OUT_DIR:-runs/v55_coeff_head_b8}

# Keep the process manifold off.  This is the root-cause kill-switch from V54.
export LATENT_CVAE_TRAJECTORY_DENOISE=${LATENT_CVAE_TRAJECTORY_DENOISE:-0}

# Output-only arm coefficient parameterization.
export LATENT_CVAE_ARM_COEFF_OUTPUT=${LATENT_CVAE_ARM_COEFF_OUTPUT:-1}
export LATENT_CVAE_ARM_COEFF_POINTS=${LATENT_CVAE_ARM_COEFF_POINTS:-8}
export LATENT_CVAE_ARM_COEFF_BASIS=${LATENT_CVAE_ARM_COEFF_BASIS:-dct}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/current_v54_rebase.sh" "$@"
