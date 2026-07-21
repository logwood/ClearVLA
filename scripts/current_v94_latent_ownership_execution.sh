#!/usr/bin/env bash
# V94: active Evidence-path ownership + native execution experiment.
#
# This is the complete, auditable entry point for the implementation that is
# currently in the workspace.  It keeps the V91 data/decoder contract and
# enables exactly these live mechanisms:
#
#   1. Evidence layer-contract and transition values remain end-to-end attached;
#   2. native execution owns capacity, dwell, monotonic block dispatch, and exit;
#   3. the native candidate value reader receives its physical candidate loss;
#   4. evaluation emits an active-path z-zero/z-shuffle condition probe.
#
# The native Evidence path keeps upstream execution context attached by default;
# transition-detach remains an explicit legacy compatibility switch.
# No workspace controller, posterior sampling, spectral state, adaptive refine,
# micro control, or hierarchical upper controller is enabled in this arm.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "clearvla/cli/train_v40_policy.py" ]]; then
  echo "[v94] repository root check failed: clearvla/cli/train_v40_policy.py is missing" >&2
  exit 2
fi

# Never default back into the historical V94 directory: the runtime manifest
# also rejects accidental JSONL append/checkpoint overwrite.
export OUT_DIR="${OUT_DIR:-runs/v94_integrated_execution_fix2}"
export LATENT_CVAE_MMDIT_DEPTH="${LATENT_CVAE_MMDIT_DEPTH:-3}"
export LATENT_CVAE_MMDIT_OPERATOR_RANK="${LATENT_CVAE_MMDIT_OPERATOR_RANK:-32}"
export LATENT_CVAE_MMDIT_OPERATOR_GROUPS="${LATENT_CVAE_MMDIT_OPERATOR_GROUPS:-32}"
# logit(29/32): the fully opened raw policy starts from a modest, explicit
# three-channel contraction. Warm-up still begins at the exact rank-32 host.
export LATENT_CVAE_MMDIT_OPERATOR_DEPTH_LOGIT_INIT="${LATENT_CVAE_MMDIT_OPERATOR_DEPTH_LOGIT_INIT:-2.268683541}"
export LATENT_CVAE_MMDIT_CONTROL_TOKENS="${LATENT_CVAE_MMDIT_CONTROL_TOKENS:-8}"
export LATENT_CVAE_MMDIT_CONTROLLER_DEPTH="${LATENT_CVAE_MMDIT_CONTROLLER_DEPTH:-2}"
export LATENT_CVAE_MMDIT_CONTROLLER_HEADS="${LATENT_CVAE_MMDIT_CONTROLLER_HEADS:-8}"
export LATENT_CVAE_MMDIT_MAX_DWELL="${LATENT_CVAE_MMDIT_MAX_DWELL:-2}"
export LATENT_CVAE_MMDIT_DWELL_MODE="${LATENT_CVAE_MMDIT_DWELL_MODE:-learned}"
export LATENT_CVAE_MMDIT_EXECUTION_SOFT_TEMPERATURE="${LATENT_CVAE_MMDIT_EXECUTION_SOFT_TEMPERATURE:-1.0}"
export LATENT_CVAE_MMDIT_IDENTITY_CANDIDATE="${LATENT_CVAE_MMDIT_IDENTITY_CANDIDATE:-1}"
export LATENT_CVAE_MMDIT_TERMINAL_PRIOR_WEIGHT="${LATENT_CVAE_MMDIT_TERMINAL_PRIOR_WEIGHT:-0.25}"
export LATENT_CVAE_MMDIT_EXECUTION_EVAL_POLICY="${LATENT_CVAE_MMDIT_EXECUTION_EVAL_POLICY:-soft}"
export LATENT_CVAE_MMDIT_EXECUTION_WARMUP="${LATENT_CVAE_MMDIT_EXECUTION_WARMUP:-200}"
export LATENT_CVAE_MMDIT_EXECUTION_TRANSITION="${LATENT_CVAE_MMDIT_EXECUTION_TRANSITION:-1000}"
export LATENT_CVAE_LAYER_CONTRACT_GRAD_SCALE="${LATENT_CVAE_LAYER_CONTRACT_GRAD_SCALE:-1.0}"
export LATENT_CVAE_LAYER_GRAD_SCALE="${LATENT_CVAE_LAYER_GRAD_SCALE:-1.0}"
export LATENT_CVAE_LAYER_DETACH="${LATENT_CVAE_LAYER_DETACH:-0}"
export LATENT_CVAE_TRANSITION_DETACH="${LATENT_CVAE_TRANSITION_DETACH:-0}"
export LATENT_CVAE_Z_PROBE="${LATENT_CVAE_Z_PROBE:-1}"
export EXECUTION_VALUE_WEIGHT="${EXECUTION_VALUE_WEIGHT:-0.05}"
export LAYER_CONTRACT_AUX_WEIGHT="${LAYER_CONTRACT_AUX_WEIGHT:-0.03}"

case "${LATENT_CVAE_MMDIT_DWELL_MODE}" in
  learned) ;;
  *)
    echo "[v94] dwell mode must be learned; learned_shadow bypasses the V94 execution path, got ${LATENT_CVAE_MMDIT_DWELL_MODE}" >&2
    exit 2
    ;;
esac

python - \
  "${LATENT_CVAE_LAYER_CONTRACT_GRAD_SCALE}" \
  "${LATENT_CVAE_LAYER_GRAD_SCALE}" \
  "${LATENT_CVAE_LAYER_DETACH}" \
  "${LATENT_CVAE_TRANSITION_DETACH}" \
  "${LATENT_CVAE_Z_PROBE}" \
  "${EXECUTION_VALUE_WEIGHT}" \
  "${LATENT_CVAE_MMDIT_EXECUTION_SOFT_TEMPERATURE}" \
  "${LATENT_CVAE_MMDIT_OPERATOR_DEPTH_LOGIT_INIT}" \
  "${LATENT_CVAE_MMDIT_DEPTH}" \
  "${LATENT_CVAE_MMDIT_OPERATOR_RANK}" \
  "${LATENT_CVAE_MMDIT_OPERATOR_GROUPS}" \
  "${LATENT_CVAE_MMDIT_MAX_DWELL}" \
  "${LATENT_CVAE_MMDIT_EXECUTION_WARMUP}" \
  "${LATENT_CVAE_MMDIT_EXECUTION_TRANSITION}" \
  "${LAYER_CONTRACT_AUX_WEIGHT}" \
  "${LATENT_CVAE_MMDIT_IDENTITY_CANDIDATE}" \
  "${LATENT_CVAE_MMDIT_TERMINAL_PRIOR_WEIGHT}" \
  "${LATENT_CVAE_MMDIT_EXECUTION_EVAL_POLICY}" <<'PY'
import math
import sys

grad_scale = float(sys.argv[1])
layer_grad_scale = float(sys.argv[2])
layer_detach = int(sys.argv[3])
transition_detach = int(sys.argv[4])
z_probe = int(sys.argv[5])
value_weight = float(sys.argv[6])
soft_temperature = float(sys.argv[7])
depth_logit_init = float(sys.argv[8])
depth = int(sys.argv[9])
rank = int(sys.argv[10])
groups = int(sys.argv[11])
max_dwell = int(sys.argv[12])
warmup = int(sys.argv[13])
transition = int(sys.argv[14])
layer_aux_weight = float(sys.argv[15])
identity_candidate = int(sys.argv[16])
terminal_prior_weight = float(sys.argv[17])
execution_eval_policy = str(sys.argv[18])
if grad_scale != 1.0 or layer_grad_scale != 1.0:
    raise SystemExit(
        "[v94] the mainline requires full upstream gradients: "
        f"contract/layer={grad_scale}/{layer_grad_scale}"
    )
if layer_detach != 0 or transition_detach != 0:
    raise SystemExit(
        "[v94] the mainline forbids layer/transition detach: "
        f"{layer_detach}/{transition_detach}"
    )
if z_probe != 1:
    raise SystemExit(f"[v94] the active-path z probe must remain enabled, got {z_probe}")
if value_weight <= 0.0:
    raise SystemExit(f"[v94] execution value loss weight must be positive, got {value_weight}")
if soft_temperature <= 0.0:
    raise SystemExit(f"[v94] execution soft temperature must be positive, got {soft_temperature}")
if not math.isfinite(depth_logit_init) or depth_logit_init <= 0.0:
    raise SystemExit(f"[v94] operator depth logit init must be finite and positive, got {depth_logit_init}")
if depth < 1 or rank < 1 or groups < 1 or max_dwell < 1:
    raise SystemExit(
        "[v94] depth, rank, groups, and max dwell must all be positive, got "
        f"{depth}, {rank}, {groups}, {max_dwell}"
    )
if rank % groups:
    raise SystemExit(f"[v94] operator rank must be divisible by groups, got {rank}/{groups}")
if warmup < 0 or transition < 1:
    raise SystemExit(f"[v94] warmup must be >=0 and transition >=1, got {warmup}/{transition}")
if not math.isfinite(layer_aux_weight) or layer_aux_weight <= 0.0:
    raise SystemExit(
        "[v94] layer contract auxiliary weight must be finite and positive, "
        f"got {layer_aux_weight}"
    )
if identity_candidate != 1:
    raise SystemExit("[v94] the mainline requires one explicit identity candidate")
if not math.isfinite(terminal_prior_weight) or not 0.0 < terminal_prior_weight <= 1.0:
    raise SystemExit(
        f"[v94] terminal prior weight must be in (0,1], got {terminal_prior_weight}"
    )
if execution_eval_policy != "soft":
    raise SystemExit(
        "[v94] the mainline evaluates the trained soft contract; "
        f"hard/neutral are ablations, got {execution_eval_policy}"
    )
PY

printf '[v94] decoder=evidence_latent_mmdit_action depth=%s contract_grad_scale=%s layer_grad_scale=%s layer_detach=%s z_probe=%s transition_detach=%s capacity_gate=1 rank=%s groups=%s depth_logit_init=%s dynamic_route=1 controller_tokens=%s dwell=%s max_dwell=%s soft_temperature=%s terminal_candidate=%s terminal_prior=%s eval_policy=%s warmup=%s transition=%s value_weight=%s layer_aux_weight=%s duplicate_delta_weights=0 workspace_controller=0 posterior=0 spectral=0 adaptive=0 micro=0\n' \
  "${LATENT_CVAE_MMDIT_DEPTH}" \
  "${LATENT_CVAE_LAYER_CONTRACT_GRAD_SCALE}" \
  "${LATENT_CVAE_LAYER_GRAD_SCALE}" \
  "${LATENT_CVAE_LAYER_DETACH}" \
  "${LATENT_CVAE_Z_PROBE}" \
  "${LATENT_CVAE_TRANSITION_DETACH}" \
  "${LATENT_CVAE_MMDIT_OPERATOR_RANK}" \
  "${LATENT_CVAE_MMDIT_OPERATOR_GROUPS}" \
  "${LATENT_CVAE_MMDIT_OPERATOR_DEPTH_LOGIT_INIT}" \
  "${LATENT_CVAE_MMDIT_CONTROL_TOKENS}" \
  "${LATENT_CVAE_MMDIT_DWELL_MODE}" \
  "${LATENT_CVAE_MMDIT_MAX_DWELL}" \
  "${LATENT_CVAE_MMDIT_EXECUTION_SOFT_TEMPERATURE}" \
  "${LATENT_CVAE_MMDIT_IDENTITY_CANDIDATE}" \
  "${LATENT_CVAE_MMDIT_TERMINAL_PRIOR_WEIGHT}" \
  "${LATENT_CVAE_MMDIT_EXECUTION_EVAL_POLICY}" \
  "${LATENT_CVAE_MMDIT_EXECUTION_WARMUP}" \
  "${LATENT_CVAE_MMDIT_EXECUTION_TRANSITION}" \
  "${EXECUTION_VALUE_WEIGHT}" \
  "${LAYER_CONTRACT_AUX_WEIGHT}"

if [[ "${V94_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[v94] preflight-only: configuration accepted; training was not started"
  exit 0
fi

# Forward dataset/optimizer overrides first, then append the V94 structural
# contract. With argparse's last-value rule, a stale command-line flag cannot
# silently route this entry point back through a legacy execution branch.
# ``rollout_delta`` and the layer ``midcut_rollout_delta`` both read the same
# milestone_step_delta prediction/target as the retained milestone objective;
# V94 disables those duplicate aliases and keeps one explicit contract weight.
exec bash "${SCRIPT_DIR}/current_v91_time_domain_evidence_mmdit.sh" \
  "$@" \
  --final-action-decoder evidence_latent_mmdit_action \
  --latent-cvae-mmdit-depth "${LATENT_CVAE_MMDIT_DEPTH}" \
  --latent-cvae-mmdit-cond-update 0 \
  --latent-cvae-mmdit-residual-scale-max "${LATENT_CVAE_MMDIT_RESIDUAL_SCALE_MAX:-0.25}" \
  --latent-cvae-mmdit-evidence-scale "${LATENT_CVAE_MMDIT_EVIDENCE_SCALE:-1.0}" \
  --latent-cvae-mmdit-noisy-scale "${LATENT_CVAE_MMDIT_NOISY_SCALE:-1.0}" \
  --latent-cvae-layer-scan "${LATENT_CVAE_LAYER_SCAN:-1}" \
  --latent-cvae-layer-scan-alpha "${LATENT_CVAE_LAYER_SCAN_ALPHA:-0.2}" \
  --latent-cvae-condition-source-norm 1 \
  --latent-cvae-bounded-consequence-fusion 1 \
  --layer-contract-grad-scale "${LATENT_CVAE_LAYER_CONTRACT_GRAD_SCALE}" \
  --latent-cvae-layer-memory 1 \
  --latent-cvae-layer-detach "${LATENT_CVAE_LAYER_DETACH}" \
  --latent-cvae-layer-grad-scale "${LATENT_CVAE_LAYER_GRAD_SCALE}" \
  --latent-cvae-transition-memory 1 \
  --latent-cvae-transition-detach "${LATENT_CVAE_TRANSITION_DETACH}" \
  --latent-cvae-workspace-trajectory-source 1 \
  --latent-cvae-workspace-global-sources 1 \
  --latent-cvae-workspace-layer-source 1 \
  --latent-cvae-workspace-progress-value 1 \
  --latent-cvae-workspace-time-state 0 \
  --latent-cvae-workspace-slot-time-state 1 \
  --latent-cvae-workspace-controller 0 \
  --latent-cvae-hierarchical-workspace 0 \
  --latent-cvae-z-probe "${LATENT_CVAE_Z_PROBE}" \
  --latent-cvae-mmdit-operator-capacity 1 \
  --latent-cvae-mmdit-operator-rank "${LATENT_CVAE_MMDIT_OPERATOR_RANK}" \
  --latent-cvae-mmdit-operator-groups "${LATENT_CVAE_MMDIT_OPERATOR_GROUPS}" \
  --latent-cvae-mmdit-operator-depth-logit-init "${LATENT_CVAE_MMDIT_OPERATOR_DEPTH_LOGIT_INIT}" \
  --latent-cvae-mmdit-execution-controller 1 \
  --latent-cvae-mmdit-dynamic-block-route 1 \
  --latent-cvae-mmdit-control-tokens "${LATENT_CVAE_MMDIT_CONTROL_TOKENS}" \
  --latent-cvae-mmdit-controller-depth "${LATENT_CVAE_MMDIT_CONTROLLER_DEPTH}" \
  --latent-cvae-mmdit-controller-heads "${LATENT_CVAE_MMDIT_CONTROLLER_HEADS}" \
  --latent-cvae-mmdit-controller-ffn-expansion 2.0 \
  --latent-cvae-mmdit-max-dwell "${LATENT_CVAE_MMDIT_MAX_DWELL}" \
  --latent-cvae-mmdit-dwell-mode "${LATENT_CVAE_MMDIT_DWELL_MODE}" \
  --latent-cvae-mmdit-execution-soft-temperature "${LATENT_CVAE_MMDIT_EXECUTION_SOFT_TEMPERATURE}" \
  --latent-cvae-mmdit-identity-candidate "${LATENT_CVAE_MMDIT_IDENTITY_CANDIDATE}" \
  --latent-cvae-mmdit-terminal-prior-weight "${LATENT_CVAE_MMDIT_TERMINAL_PRIOR_WEIGHT}" \
  --latent-cvae-mmdit-execution-eval-policy "${LATENT_CVAE_MMDIT_EXECUTION_EVAL_POLICY}" \
  --latent-cvae-mmdit-execution-warmup-steps "${LATENT_CVAE_MMDIT_EXECUTION_WARMUP}" \
  --latent-cvae-mmdit-execution-transition-steps "${LATENT_CVAE_MMDIT_EXECUTION_TRANSITION}" \
  --latent-cvae-mmdit-execution-value-loss-weight "${EXECUTION_VALUE_WEIGHT}" \
  --eval-execution-ablation-batches "${EVAL_EXECUTION_ABLATION_BATCHES:-8}" \
  --rollout-delta-loss-weight 0 \
  --midcut-rollout-delta-loss-weight 0 \
  --rollout-dynamics-loss-weight 0.02 \
  --rollout-contrast-loss-weight 0.03 \
  --rollout-variance-loss-weight 0.03 \
  --rollout-norm-loss-weight 0.01 \
  --rollout-milestone-delta-match-weight 0.08 \
  --first-weight 1.20 \
  --first4-weight 1.15 \
  --first8-weight 1.10 \
  --tail-weight 1.10 \
  --event-loss-weight 0.03 \
  --layer-contract-loss-weight 1.0 \
  --layer-contract-aux-loss-weight "${LAYER_CONTRACT_AUX_WEIGHT}" \
  --layer-contract-aux-final-ratio 1.0 \
  --layer-contract-aux-decay-epochs 0 \
  --latent-cvae-inference-sample 0 \
  --latent-cvae-variational 0 \
  --hierarchical-mmdit-spectral-state 0 \
  --hierarchical-mmdit-unified-controller 0 \
  --hierarchical-mmdit-operation-candidate-probes 0 \
  --hierarchical-mmdit-operation-value-loss-weight 0 \
  --arm-flow-mode legacy_independent \
  --gripper-field-mode legacy_handcrafted \
  --adaptive-cvae-function-adapters 0 \
  --adaptive-cvae-micro-control 0 \
  --adaptive-cvae-micro-refine-block 0 \
  --adaptive-cvae-direct-condition-residual 0 \
  --adaptive-cvae-condition-strength 0 \
  --action-consequence-self-condition 0
