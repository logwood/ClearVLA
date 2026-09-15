#!/usr/bin/env bash
# =============================================================================
# V58 DETAIL MICRO: v57 query-spline writer + full-action-token detail unfolding.
#
# This keeps the B-spline writer as an output/action-space parameterization and
# adds a lightweight residual detail writer inside each refine step.  It reuses
# the useful just_ok idea, intermediate full-token supervision, without bringing
# back the old PD/damping/Heun micro controller.
#
# No process trajectory manifold, no smoothing loss, no target-action shortcut.
# =============================================================================
set -euo pipefail

export OUT_DIR=${OUT_DIR:-runs/v58_detail_micro_b8}

# Keep the failed process manifold off.  Detail operates on full horizon tokens.
export LATENT_CVAE_TRAJECTORY_DENOISE=${LATENT_CVAE_TRAJECTORY_DENOISE:-0}

# Output/action-space spline writer from v57.
export LATENT_CVAE_ARM_COEFF_OUTPUT=${LATENT_CVAE_ARM_COEFF_OUTPUT:-1}
export LATENT_CVAE_ARM_COEFF_POINTS=${LATENT_CVAE_ARM_COEFF_POINTS:-8}
export LATENT_CVAE_ARM_COEFF_BASIS=${LATENT_CVAE_ARM_COEFF_BASIS:-bspline}
export LATENT_CVAE_ARM_COEFF_DEGREE=${LATENT_CVAE_ARM_COEFF_DEGREE:-2}
export LATENT_CVAE_ARM_COEFF_RIDGE=${LATENT_CVAE_ARM_COEFF_RIDGE:-1e-2}
export LATENT_CVAE_ARM_COEFF_WRITER=${LATENT_CVAE_ARM_COEFF_WRITER:-query_direction}
export LATENT_CVAE_COEFF_INCLUDE_GRIPPER=${LATENT_CVAE_COEFF_INCLUDE_GRIPPER:-1}
export LATENT_CVAE_COEFF_MAGNITUDE_GROUPS=${LATENT_CVAE_COEFF_MAGNITUDE_GROUPS:-2}
export LATENT_CVAE_SPLINE_BASE_DIAGNOSTICS=${LATENT_CVAE_SPLINE_BASE_DIAGNOSTICS:-1}

# Concentrate residual capacity in the new detail path, not the old function
# bank side path.
export ADAPTIVE_CVAE_FUNCTION_ADAPTERS=${ADAPTIVE_CVAE_FUNCTION_ADAPTERS:-0}

# Detail residual unfolding.
export ADAPTIVE_CVAE_DETAIL_MICRO=${ADAPTIVE_CVAE_DETAIL_MICRO:-1}
export ADAPTIVE_CVAE_DETAIL_MICRO_SUPERVISION=${ADAPTIVE_CVAE_DETAIL_MICRO_SUPERVISION:-1}
export ADAPTIVE_CVAE_DETAIL_MICRO_SCALE=${ADAPTIVE_CVAE_DETAIL_MICRO_SCALE:-0.30}
export ADAPTIVE_CVAE_DETAIL_MICRO_GATE_INIT=${ADAPTIVE_CVAE_DETAIL_MICRO_GATE_INIT:-0.45}
export LATENT_CVAE_MICRO_SUPERVISION_WEIGHT=${LATENT_CVAE_MICRO_SUPERVISION_WEIGHT:-0.06}
export LATENT_CVAE_MICRO_EVENT_WEIGHT=${LATENT_CVAE_MICRO_EVENT_WEIGHT:-0.01}
export LATENT_CVAE_MICRO_MONOTONIC_WEIGHT=${LATENT_CVAE_MICRO_MONOTONIC_WEIGHT:-0.01}
export LATENT_CVAE_MICRO_WEIGHT_KL_WEIGHT=${LATENT_CVAE_MICRO_WEIGHT_KL_WEIGHT:-0.0005}
export LATENT_CVAE_MICRO_COVERAGE_SMOOTH_WEIGHT=${LATENT_CVAE_MICRO_COVERAGE_SMOOTH_WEIGHT:-0.001}
export LATENT_CVAE_MICRO_COVERAGE_FLOOR_WEIGHT=${LATENT_CVAE_MICRO_COVERAGE_FLOOR_WEIGHT:-0.001}
export LATENT_CVAE_MICRO_LEARNED_WEIGHT_MAX=${LATENT_CVAE_MICRO_LEARNED_WEIGHT_MAX:-0.35}
export LATENT_CVAE_MICRO_LEARNED_RAMP_STEPS=${LATENT_CVAE_MICRO_LEARNED_RAMP_STEPS:-2000}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/current_v54_rebase.sh" "$@"
