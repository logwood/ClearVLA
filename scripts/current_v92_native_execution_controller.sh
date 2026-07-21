#!/usr/bin/env bash
# V92: native-time evidence MMDiT with one unified execution controller.
#
# The host operation is still the V91 full-rank MMDiT block.  Capacity is a
# nested direction aperture and dwell is repeated execution of that same
# block; neither is an independent residual-amplitude or noisy-source gate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OUT_DIR="${OUT_DIR:-runs/v92_native_execution_controller}"
export LATENT_CVAE_MMDIT_DEPTH="${LATENT_CVAE_MMDIT_DEPTH:-3}"
export LATENT_CVAE_MMDIT_OPERATOR_RANK="${LATENT_CVAE_MMDIT_OPERATOR_RANK:-32}"
export LATENT_CVAE_MMDIT_OPERATOR_GROUPS="${LATENT_CVAE_MMDIT_OPERATOR_GROUPS:-4}"
export LATENT_CVAE_MMDIT_CONTROL_TOKENS="${LATENT_CVAE_MMDIT_CONTROL_TOKENS:-8}"
export LATENT_CVAE_MMDIT_CONTROLLER_DEPTH="${LATENT_CVAE_MMDIT_CONTROLLER_DEPTH:-2}"
export LATENT_CVAE_MMDIT_CONTROLLER_HEADS="${LATENT_CVAE_MMDIT_CONTROLLER_HEADS:-8}"
export LATENT_CVAE_MMDIT_MAX_DWELL="${LATENT_CVAE_MMDIT_MAX_DWELL:-2}"
export LATENT_CVAE_MMDIT_DWELL_MODE="${LATENT_CVAE_MMDIT_DWELL_MODE:-learned}"

printf '[v92] decoder=evidence_latent_mmdit_action depth=%s capacity=1 rank=%s groups=%s controller=1 tokens=%s dwell=%s max_dwell=%s warmup=200\n' \
  "${LATENT_CVAE_MMDIT_DEPTH}" \
  "${LATENT_CVAE_MMDIT_OPERATOR_RANK}" \
  "${LATENT_CVAE_MMDIT_OPERATOR_GROUPS}" \
  "${LATENT_CVAE_MMDIT_CONTROL_TOKENS}" \
  "${LATENT_CVAE_MMDIT_DWELL_MODE}" \
  "${LATENT_CVAE_MMDIT_MAX_DWELL}"

exec bash "${SCRIPT_DIR}/current_v91_time_domain_evidence_mmdit.sh" \
  --latent-cvae-mmdit-depth "${LATENT_CVAE_MMDIT_DEPTH}" \
  --latent-cvae-mmdit-operator-capacity 1 \
  --latent-cvae-mmdit-operator-rank "${LATENT_CVAE_MMDIT_OPERATOR_RANK}" \
  --latent-cvae-mmdit-operator-groups "${LATENT_CVAE_MMDIT_OPERATOR_GROUPS}" \
  --latent-cvae-mmdit-operator-depth-logit-init 4.0 \
  --latent-cvae-mmdit-execution-controller 1 \
  --latent-cvae-mmdit-control-tokens "${LATENT_CVAE_MMDIT_CONTROL_TOKENS}" \
  --latent-cvae-mmdit-controller-depth "${LATENT_CVAE_MMDIT_CONTROLLER_DEPTH}" \
  --latent-cvae-mmdit-controller-heads "${LATENT_CVAE_MMDIT_CONTROLLER_HEADS}" \
  --latent-cvae-mmdit-controller-ffn-expansion 2.0 \
  --latent-cvae-mmdit-max-dwell "${LATENT_CVAE_MMDIT_MAX_DWELL}" \
  --latent-cvae-mmdit-dwell-mode "${LATENT_CVAE_MMDIT_DWELL_MODE}" \
  --latent-cvae-mmdit-execution-warmup-steps 200 \
  --latent-cvae-mmdit-execution-transition-steps 1000 \
  --latent-cvae-mmdit-execution-value-loss-weight "${EXECUTION_VALUE_WEIGHT:-0.05}" \
  "$@"
