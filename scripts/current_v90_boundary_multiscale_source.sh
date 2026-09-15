#!/usr/bin/env bash
# V90: replace only the native arm bridge source. The field/DCT chart,
# gripper source, MMDiT, controller, losses, and sampler remain on V89.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ARM_SOURCE_MODE="${ARM_SOURCE_MODE:-boundary_multiscale}"
# Trace-match the conditional stochastic energy of the rho=0.95 AR baseline
# (mean horizon RMS ~= 0.804) so the first A/B isolates temporal geometry.
export ARM_SOURCE_SCALE="${ARM_SOURCE_SCALE:-0.80}"
export ARM_SOURCE_INNOVATION_WEIGHT="${ARM_SOURCE_INNOVATION_WEIGHT:-0.50}"
export ARM_SOURCE_VELOCITY_WEIGHT="${ARM_SOURCE_VELOCITY_WEIGHT:-0.35}"
export ARM_SOURCE_ACCELERATION_WEIGHT="${ARM_SOURCE_ACCELERATION_WEIGHT:-0.15}"
export ARM_NOISE_TEMPORAL_RHO="${ARM_NOISE_TEMPORAL_RHO:-0.95}"
export HIERARCHICAL_MMDIT_DWELL_MODE=fixed
export HIERARCHICAL_MMDIT_OPERATION_VALUE_LOSS_WEIGHT=0
export OUT_DIR="${OUT_DIR:-runs/v90_${ARM_SOURCE_MODE}}"

printf '[v90] arm_source=%s scale=%s weights=%s/%s/%s rho_if_ar1=%s control=fixed\n' \
  "${ARM_SOURCE_MODE}" \
  "${ARM_SOURCE_SCALE}" \
  "${ARM_SOURCE_INNOVATION_WEIGHT}" \
  "${ARM_SOURCE_VELOCITY_WEIGHT}" \
  "${ARM_SOURCE_ACCELERATION_WEIGHT}" \
  "${ARM_NOISE_TEMPORAL_RHO}"

# Fixed dwell and disabled candidate probes keep the source A/B independent of
# learned execution routing. Both modes still use the same V89 typed blocks.
exec bash "${SCRIPT_DIR}/current_v89_typed_block_budget.sh" \
  --arm-source-mode "${ARM_SOURCE_MODE}" \
  --arm-source-scale "${ARM_SOURCE_SCALE}" \
  --arm-source-innovation-weight "${ARM_SOURCE_INNOVATION_WEIGHT}" \
  --arm-source-velocity-weight "${ARM_SOURCE_VELOCITY_WEIGHT}" \
  --arm-source-acceleration-weight "${ARM_SOURCE_ACCELERATION_WEIGHT}" \
  --arm-noise-temporal-rho "${ARM_NOISE_TEMPORAL_RHO}" \
  --hierarchical-mmdit-dwell-mode fixed \
  --hierarchical-mmdit-operation-candidate-probes 0 \
  --hierarchical-mmdit-operation-value-loss-weight 0 \
  "$@"
