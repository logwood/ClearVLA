#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/data/liang.zhang/dataset/grab_pen_single/grab_pen_single}
CACHE_DIR=${CACHE_DIR:-/home/sen.wang/workspace/robotics/clear/data/cache_336}
DINO_CACHE_DIR=${DINO_CACHE_DIR:-/home/sen.wang/workspace/robotics/clear/data/dinov2_cache_336}
OUT_DIR=${OUT_DIR:-runs/v52_proposal_residual_coeff_b8}
LATENT_CVAE_LAYER_DETACH=${LATENT_CVAE_LAYER_DETACH:-1}
LATENT_CVAE_LAYER_GRAD_SCALE=${LATENT_CVAE_LAYER_GRAD_SCALE:-0.15}
LATENT_CVAE_TRANSITION_DETACH=${LATENT_CVAE_TRANSITION_DETACH:-1}
LATENT_CVAE_MU_BOUND=${LATENT_CVAE_MU_BOUND:-1.25}
LATENT_CVAE_MIN_STD=${LATENT_CVAE_MIN_STD:-0.6}
LATENT_CVAE_CAUSAL_ATTENTION=${LATENT_CVAE_CAUSAL_ATTENTION:-1}
LATENT_CVAE_TRAJECTORY_DENOISE=${LATENT_CVAE_TRAJECTORY_DENOISE:-1}
LATENT_CVAE_TRAJECTORY_CONTROL_POINTS=${LATENT_CVAE_TRAJECTORY_CONTROL_POINTS:-8}
LATENT_CVAE_TRAJECTORY_CONTEXT=${LATENT_CVAE_TRAJECTORY_CONTEXT:-1}
LATENT_CVAE_TRAJECTORY_MID_SUPERVISION=${LATENT_CVAE_TRAJECTORY_MID_SUPERVISION:-1}
LATENT_CVAE_TRAJECTORY_UPDATE_SCALE=${LATENT_CVAE_TRAJECTORY_UPDATE_SCALE:-1.0}
LATENT_CVAE_TRAJECTORY_CONTEXT_SCALE=${LATENT_CVAE_TRAJECTORY_CONTEXT_SCALE:-0.50}
ADAPTIVE_CVAE_REFINE_STEPS=${ADAPTIVE_CVAE_REFINE_STEPS:-6}
ADAPTIVE_CVAE_PROGRESS_MEMORY=${ADAPTIVE_CVAE_PROGRESS_MEMORY:-1}
ADAPTIVE_CVAE_PROGRESS_STEPS=${ADAPTIVE_CVAE_PROGRESS_STEPS:-6}
ADAPTIVE_CVAE_PREFIX_MEMORY=${ADAPTIVE_CVAE_PREFIX_MEMORY:-0}
ADAPTIVE_CVAE_LAYER_ROUTING=${ADAPTIVE_CVAE_LAYER_ROUTING:-1}
ADAPTIVE_CVAE_ROUTE_COSINE=${ADAPTIVE_CVAE_ROUTE_COSINE:-1}
ADAPTIVE_CVAE_ROUTE_TEMPERATURE=${ADAPTIVE_CVAE_ROUTE_TEMPERATURE:-1.0}
ADAPTIVE_CVAE_PREFIX_DETACH=${ADAPTIVE_CVAE_PREFIX_DETACH:-1}
ADAPTIVE_CVAE_PROGRESS_Z_INJECTION=${ADAPTIVE_CVAE_PROGRESS_Z_INJECTION:-1}
ADAPTIVE_CVAE_ROUTE_QUERY_BIAS=${ADAPTIVE_CVAE_ROUTE_QUERY_BIAS:-1}
ADAPTIVE_CVAE_TOKEN_SEMANTIC_ADAPTER=${ADAPTIVE_CVAE_TOKEN_SEMANTIC_ADAPTER:-1}
ADAPTIVE_CVAE_CONTEXT_DROPOUT=${ADAPTIVE_CVAE_CONTEXT_DROPOUT:-0.05}
ADAPTIVE_CVAE_ROUTE_ENTROPY_FLOOR_RATIO=${ADAPTIVE_CVAE_ROUTE_ENTROPY_FLOOR_RATIO:-0.15}
ADAPTIVE_CVAE_FUNCTION_ADAPTERS=${ADAPTIVE_CVAE_FUNCTION_ADAPTERS:-0}
ADAPTIVE_CVAE_FUNCTION_RANK=${ADAPTIVE_CVAE_FUNCTION_RANK:-32}
ADAPTIVE_CVAE_PROGRESS_ROLE_DIM=${ADAPTIVE_CVAE_PROGRESS_ROLE_DIM:-16}
ADAPTIVE_CVAE_ROUTE_TOPK=${ADAPTIVE_CVAE_ROUTE_TOPK:-0}
ADAPTIVE_CVAE_ROUTE_SPARSEMAX=${ADAPTIVE_CVAE_ROUTE_SPARSEMAX:-1}
ADAPTIVE_CVAE_ROUTE_ADAPTIVE_TEMPERATURE=${ADAPTIVE_CVAE_ROUTE_ADAPTIVE_TEMPERATURE:-1}
ADAPTIVE_CVAE_ROUTE_MIN_TEMPERATURE=${ADAPTIVE_CVAE_ROUTE_MIN_TEMPERATURE:-0.35}
ADAPTIVE_CVAE_ROUTE_MAX_TEMPERATURE=${ADAPTIVE_CVAE_ROUTE_MAX_TEMPERATURE:-1.25}
ADAPTIVE_CVAE_ROLE_QUERY=${ADAPTIVE_CVAE_ROLE_QUERY:-1}
ADAPTIVE_CVAE_STEP_ROLES=${ADAPTIVE_CVAE_STEP_ROLES:-1}
ADAPTIVE_CVAE_COARSE_STRIDE=${ADAPTIVE_CVAE_COARSE_STRIDE:-4}
ADAPTIVE_CVAE_COARSE_STRENGTH=${ADAPTIVE_CVAE_COARSE_STRENGTH:-0.35}
ADAPTIVE_CVAE_SEED_SCALE=${ADAPTIVE_CVAE_SEED_SCALE:-0.35}
ADAPTIVE_CVAE_OUTPUT_SCALE=${ADAPTIVE_CVAE_OUTPUT_SCALE:-0.05}
ADAPTIVE_CVAE_CONTEXT_CAPSULES=${ADAPTIVE_CVAE_CONTEXT_CAPSULES:-1}
ADAPTIVE_CVAE_CONTEXT_CAPSULE_COUNT=${ADAPTIVE_CVAE_CONTEXT_CAPSULE_COUNT:-6}
LATENT_CVAE_TRAJECTORY_SUPERVISION_WEIGHT=${LATENT_CVAE_TRAJECTORY_SUPERVISION_WEIGHT:-0.04}
LATENT_CVAE_TRAJECTORY_COEFF_WEIGHT=${LATENT_CVAE_TRAJECTORY_COEFF_WEIGHT:-0.04}
LATENT_CVAE_TRAJECTORY_MONOTONIC_WEIGHT=${LATENT_CVAE_TRAJECTORY_MONOTONIC_WEIGHT:-0.01}
LATENT_CVAE_TRAJECTORY_SMOOTHNESS_WEIGHT=${LATENT_CVAE_TRAJECTORY_SMOOTHNESS_WEIGHT:-0.0}
LATENT_CVAE_TRAJECTORY_UPDATE_SMOOTHNESS_WEIGHT=${LATENT_CVAE_TRAJECTORY_UPDATE_SMOOTHNESS_WEIGHT:-0.0}
LATENT_CVAE_TRAJECTORY_UPDATE_ENERGY_WEIGHT=${LATENT_CVAE_TRAJECTORY_UPDATE_ENERGY_WEIGHT:-0.0}
LATENT_CVAE_TRAJECTORY_PROJECTION_WEIGHT=${LATENT_CVAE_TRAJECTORY_PROJECTION_WEIGHT:-0.0}
LATENT_CVAE_PROPOSAL_RESIDUAL_COEFF_WEIGHT=${LATENT_CVAE_PROPOSAL_RESIDUAL_COEFF_WEIGHT:-0.06}
LATENT_CVAE_PROPOSAL_RESIDUAL_MID_COEFF_WEIGHT=${LATENT_CVAE_PROPOSAL_RESIDUAL_MID_COEFF_WEIGHT:-0.03}
LATENT_CVAE_PROPOSAL_RESIDUAL_BOUND_WEIGHT=${LATENT_CVAE_PROPOSAL_RESIDUAL_BOUND_WEIGHT:-0.002}
LATENT_CVAE_PROPOSAL_RESIDUAL_BOUND_RATIO=${LATENT_CVAE_PROPOSAL_RESIDUAL_BOUND_RATIO:-1.25}
LATENT_CVAE_PROPOSAL_RESIDUAL_COEFF_RIDGE=${LATENT_CVAE_PROPOSAL_RESIDUAL_COEFF_RIDGE:-1e-2}
LATENT_CVAE_PROPOSAL_RESIDUAL_ARM_ONLY=${LATENT_CVAE_PROPOSAL_RESIDUAL_ARM_ONLY:-1}
LATENT_CVAE_LEGACY_ANCHOR_WEIGHT=${LATENT_CVAE_LEGACY_ANCHOR_WEIGHT:-0.03}
LATENT_CVAE_LEGACY_ANCHOR_DECAY_STEPS=${LATENT_CVAE_LEGACY_ANCHOR_DECAY_STEPS:-2500}
LATENT_CVAE_LEGACY_ANCHOR_MIN_WEIGHT=${LATENT_CVAE_LEGACY_ANCHOR_MIN_WEIGHT:-0.0}
STAGE1_RESET_DIRTY_ADAPTERS=${STAGE1_RESET_DIRTY_ADAPTERS:-1}
LAYER_CONTRACT_ADAPTER_POLICY_LR_SCALE=${LAYER_CONTRACT_ADAPTER_POLICY_LR_SCALE:-${STAGE1_RESET_DIRTY_ADAPTERS}}
STAGE1_CHECKPOINT=${STAGE1_CHECKPOINT:-runs/v40_1_k6a6_statecf1_norm_full_b8/checkpoints/best_contract.pt}
if [[ ! -f "${STAGE1_CHECKPOINT}" && -f "/home/sen.wang/workspace/robotics/clear/clearvla_v40_2_goodcheck/runs/v40_1_k6a6_statecf1_norm_full_b8/checkpoints/best_contract.pt" ]]; then
  STAGE1_CHECKPOINT=/home/sen.wang/workspace/robotics/clear/clearvla_v40_2_goodcheck/runs/v40_1_k6a6_statecf1_norm_full_b8/checkpoints/best_contract.pt
elif [[ ! -f "${STAGE1_CHECKPOINT}" && -f "/home/sen.wang/workspace/robotics/clear/clearvla_v40_2/runs/v40_1_k6a6_statecf1_norm_full_b8/checkpoints/best_contract.pt" ]]; then
  STAGE1_CHECKPOINT=/home/sen.wang/workspace/robotics/clear/clearvla_v40_2/runs/v40_1_k6a6_statecf1_norm_full_b8/checkpoints/best_contract.pt
fi

python -u -m clearvla.cli.train_v40_policy \
  --data-root "${DATA_ROOT}" \
  --glob '*.hdf5' \
  --decoded-image-cache-dir "${CACHE_DIR}" \
  --cameras top wrist \
  --action-key action \
  --state-key qpos \
  --top-key observations/images/cam_high \
  --wrist-key observations/images/cam_right_wrist \
  --cache-resize 336 336 \
  --episode-split-mode ordered-counts \
  --train-episode-count 63 \
  --val-episode-count 5 \
  --test-episode-count 5 \
  --seed 0 \
  --batch-size 8 \
  --num-workers 4 \
  --normalizer zscore \
  --out-dir "${OUT_DIR}" \
  --stage1-checkpoint "${STAGE1_CHECKPOINT}" \
  --stage1-reset-dirty-adapters "${STAGE1_RESET_DIRTY_ADAPTERS}" \
  --condition-mode dinov2-cache \
  --dinov2-model facebook/dinov2-base \
  --dinov2-token-cache-dir "${DINO_CACHE_DIR}" \
  --prefetch-dinov2-tokens \
  --dtype bf16 \
  --world-horizon 48 \
  --policy-horizon 24 \
  --segment-length 4 \
  --history-offsets=-8,-4,0 \
  --executed-action-offsets=-8,-4,-1 \
  --target-history-offsets=-8,-4,0 \
  --stride 1 \
  --hidden-size 512 \
  --heads 8 \
  --depth 8 \
  --midcut-layer 3 \
  --midcut-future-gain-init 0.1 \
  --layer-contract-adapters 1 \
  --layer-contract-adapter-dim 128 \
  --layer-contract-grad-scale 1.0 \
  --layer-contract-residual-scale 0.5 \
  --layer-shared-fm-probe 0 \
  --layer-fm-probe-hidden 256 \
  --layer-recurrent-consequence 1 \
  --layer-consequence-steps 6 \
  --layer-consequence-hidden 256 \
  --layer-consequence-delta-scale 1.0 \
  --layer-consequence-initial-gain 0.1 \
  --layer-causal-feedback-depth 0 \
  --layer-causal-memory-tokens 4 \
  --layer-low-causal-weight 1.0 \
  --layer-high-causal-weight 1.0 \
  --layer-low-latent-weight 1.0 \
  --layer-high-latent-weight 1.0 \
  --layer-causal-event-from-effect 1 \
  --layer-state-counterfactual 1 \
  --proposal-depth 2 \
  --proposal-dropout 0.25 \
  --dropout 0.05 \
  --event-tokens 3 \
  --canvas-registers 12 \
  --future-anchors 6 \
  --future-grid-size 4 \
  --action-basis-tokens 4 \
  --rollout-tail-start-step 8 \
  --rollout-tail-full-step 13 \
  --controlled-delta-rank 8 \
  --base-effect-hidden 128 \
  --latent-action-tokens 8 \
  --neutral-action-tokens 4 \
  --controlled-delta-dropout 0.0 \
  --role-dropout 0.1 \
  --visual-memory-dropout 0.0 \
  --canvas-dropout 0.0 \
  --inference-steps 5 \
  --gripper-dim-index -1 \
  --first-execution-steps 4 \
  --mid-execution-steps 8 \
  --physical-decode-delta-blend 0.25 \
  --final-action-decoder adaptive_recurrent_cvae_action \
  --latent-cvae-z-dim 64 \
  --latent-cvae-decoder-depth 3 \
  --latent-cvae-ffn-expansion 2.0 \
  --latent-cvae-layer-memory 1 \
  --latent-cvae-transition-memory 1 \
  --latent-cvae-context-memory 0 \
  --latent-cvae-visual-memory 0 \
  --latent-cvae-transition-detach "${LATENT_CVAE_TRANSITION_DETACH}" \
  --latent-cvae-layer-detach "${LATENT_CVAE_LAYER_DETACH}" \
  --latent-cvae-layer-grad-scale "${LATENT_CVAE_LAYER_GRAD_SCALE}" \
  --latent-cvae-event-gripper-gate 1 \
  --latent-cvae-inference-sample 0 \
  --latent-cvae-output-init-std 1e-3 \
  --latent-cvae-mu-bound "${LATENT_CVAE_MU_BOUND}" \
  --latent-cvae-min-std "${LATENT_CVAE_MIN_STD}" \
  --latent-cvae-causal-attention "${LATENT_CVAE_CAUSAL_ATTENTION}" \
  --latent-cvae-trajectory-denoise "${LATENT_CVAE_TRAJECTORY_DENOISE}" \
  --latent-cvae-trajectory-control-points "${LATENT_CVAE_TRAJECTORY_CONTROL_POINTS}" \
  --latent-cvae-trajectory-context "${LATENT_CVAE_TRAJECTORY_CONTEXT}" \
  --latent-cvae-trajectory-mid-supervision "${LATENT_CVAE_TRAJECTORY_MID_SUPERVISION}" \
  --latent-cvae-trajectory-update-scale "${LATENT_CVAE_TRAJECTORY_UPDATE_SCALE}" \
  --latent-cvae-trajectory-context-scale "${LATENT_CVAE_TRAJECTORY_CONTEXT_SCALE}" \
  --adaptive-cvae-refine-steps "${ADAPTIVE_CVAE_REFINE_STEPS}" \
  --adaptive-cvae-progress-memory "${ADAPTIVE_CVAE_PROGRESS_MEMORY}" \
  --adaptive-cvae-progress-steps "${ADAPTIVE_CVAE_PROGRESS_STEPS}" \
  --adaptive-cvae-prefix-memory "${ADAPTIVE_CVAE_PREFIX_MEMORY}" \
  --adaptive-cvae-layer-routing "${ADAPTIVE_CVAE_LAYER_ROUTING}" \
  --adaptive-cvae-route-cosine "${ADAPTIVE_CVAE_ROUTE_COSINE}" \
  --adaptive-cvae-route-temperature "${ADAPTIVE_CVAE_ROUTE_TEMPERATURE}" \
  --adaptive-cvae-prefix-detach "${ADAPTIVE_CVAE_PREFIX_DETACH}" \
  --adaptive-cvae-progress-z-injection "${ADAPTIVE_CVAE_PROGRESS_Z_INJECTION}" \
  --adaptive-cvae-route-query-bias "${ADAPTIVE_CVAE_ROUTE_QUERY_BIAS}" \
  --adaptive-cvae-token-semantic-adapter "${ADAPTIVE_CVAE_TOKEN_SEMANTIC_ADAPTER}" \
  --adaptive-cvae-context-dropout "${ADAPTIVE_CVAE_CONTEXT_DROPOUT}" \
  --adaptive-cvae-route-entropy-floor-ratio "${ADAPTIVE_CVAE_ROUTE_ENTROPY_FLOOR_RATIO}" \
  --adaptive-cvae-function-adapters "${ADAPTIVE_CVAE_FUNCTION_ADAPTERS}" \
  --adaptive-cvae-function-rank "${ADAPTIVE_CVAE_FUNCTION_RANK}" \
  --adaptive-cvae-progress-role-dim "${ADAPTIVE_CVAE_PROGRESS_ROLE_DIM}" \
  --adaptive-cvae-route-topk "${ADAPTIVE_CVAE_ROUTE_TOPK}" \
  --adaptive-cvae-route-sparsemax "${ADAPTIVE_CVAE_ROUTE_SPARSEMAX}" \
  --adaptive-cvae-route-adaptive-temperature "${ADAPTIVE_CVAE_ROUTE_ADAPTIVE_TEMPERATURE}" \
  --adaptive-cvae-route-min-temperature "${ADAPTIVE_CVAE_ROUTE_MIN_TEMPERATURE}" \
  --adaptive-cvae-route-max-temperature "${ADAPTIVE_CVAE_ROUTE_MAX_TEMPERATURE}" \
  --adaptive-cvae-role-query "${ADAPTIVE_CVAE_ROLE_QUERY}" \
  --adaptive-cvae-step-roles "${ADAPTIVE_CVAE_STEP_ROLES}" \
  --adaptive-cvae-coarse-stride "${ADAPTIVE_CVAE_COARSE_STRIDE}" \
  --adaptive-cvae-coarse-strength "${ADAPTIVE_CVAE_COARSE_STRENGTH}" \
  --adaptive-cvae-seed-scale "${ADAPTIVE_CVAE_SEED_SCALE}" \
  --adaptive-cvae-output-scale "${ADAPTIVE_CVAE_OUTPUT_SCALE}" \
  --adaptive-cvae-context-capsules "${ADAPTIVE_CVAE_CONTEXT_CAPSULES}" \
  --adaptive-cvae-context-capsule-count "${ADAPTIVE_CVAE_CONTEXT_CAPSULE_COUNT}" \
  --epochs 8 \
  --lr 8e-5 \
  --proposal-lr 5e-5 \
  --latent-cvae-action-decoder-lr-scale 0.7 \
  --latent-cvae-kl-weight 5e-4 \
  --latent-cvae-legacy-anchor-weight "${LATENT_CVAE_LEGACY_ANCHOR_WEIGHT}" \
  --latent-cvae-legacy-anchor-decay-steps "${LATENT_CVAE_LEGACY_ANCHOR_DECAY_STEPS}" \
  --latent-cvae-legacy-anchor-min-weight "${LATENT_CVAE_LEGACY_ANCHOR_MIN_WEIGHT}" \
  --latent-cvae-posterior-recon-weight 0.05 \
  --latent-cvae-adaptive-regularizer-weight 0.002 \
  --latent-cvae-adaptive-route-entropy-weight 0.0003 \
  --latent-cvae-trajectory-supervision-weight "${LATENT_CVAE_TRAJECTORY_SUPERVISION_WEIGHT}" \
  --latent-cvae-trajectory-coeff-weight "${LATENT_CVAE_TRAJECTORY_COEFF_WEIGHT}" \
  --latent-cvae-trajectory-monotonic-weight "${LATENT_CVAE_TRAJECTORY_MONOTONIC_WEIGHT}" \
  --latent-cvae-trajectory-smoothness-weight "${LATENT_CVAE_TRAJECTORY_SMOOTHNESS_WEIGHT}" \
  --latent-cvae-trajectory-update-smoothness-weight "${LATENT_CVAE_TRAJECTORY_UPDATE_SMOOTHNESS_WEIGHT}" \
  --latent-cvae-trajectory-update-energy-weight "${LATENT_CVAE_TRAJECTORY_UPDATE_ENERGY_WEIGHT}" \
  --latent-cvae-trajectory-projection-weight "${LATENT_CVAE_TRAJECTORY_PROJECTION_WEIGHT}" \
  --latent-cvae-proposal-residual-coeff-weight "${LATENT_CVAE_PROPOSAL_RESIDUAL_COEFF_WEIGHT}" \
  --latent-cvae-proposal-residual-mid-coeff-weight "${LATENT_CVAE_PROPOSAL_RESIDUAL_MID_COEFF_WEIGHT}" \
  --latent-cvae-proposal-residual-bound-weight "${LATENT_CVAE_PROPOSAL_RESIDUAL_BOUND_WEIGHT}" \
  --latent-cvae-proposal-residual-bound-ratio "${LATENT_CVAE_PROPOSAL_RESIDUAL_BOUND_RATIO}" \
  --latent-cvae-proposal-residual-coeff-ridge "${LATENT_CVAE_PROPOSAL_RESIDUAL_COEFF_RIDGE}" \
  --latent-cvae-proposal-residual-arm-only "${LATENT_CVAE_PROPOSAL_RESIDUAL_ARM_ONLY}" \
  --weight-decay 0.01 \
  --beta1 0.9 \
  --beta2 0.999 \
  --eps 1e-8 \
  --grad-clip 1.0 \
  --warmup-steps 500 \
  --min-lr-ratio 0.1 \
  --proposal-loss-weight 0.05 \
  --first-weight 1.5 \
  --first4-weight 1.3 \
  --first8-weight 1.15 \
  --tail-weight 1.1 \
  --event-loss-weight 0.08 \
  --event-positive-weight 4.0 \
  --event-focal-gamma 1.0 \
  --gripper-transition-l1-weight 0.06 \
  --smooth-delta-weight 0.0 \
  --decoded-action-loss-weight 0.08 \
  --physical-delta-consistency-weight 0.03 \
  --transition-gripper-flow-weight 0.06 \
  --event-delta-consistency-weight 0.03 \
  --event-magnitude-weight 0.03 \
  --event-off-delta-weight 0.03 \
  --arm-motion-loss-weight 0.03 \
  --arm-motion-threshold 0.02 \
  --gripper-event-threshold 0.1 \
  --deploy-min-recall 0.4 \
  --deploy-min-event-ratio 0.7 \
  --deploy-max-event-ratio 1.8 \
  --deploy-max-tail-first-ratio 2.6 \
  --eval-inference-steps 5 \
  --log-every 20 \
  --max-train-batches 0 \
  --max-val-batches 0 \
  --rollout-dynamics-loss-weight 0.03 \
  --rollout-delta-loss-weight 0.01 \
  --rollout-contrast-loss-weight 0.06 \
  --rollout-contrast-margin 0.02 \
  --future-latent-loss-weight 0.0 \
  --action-effect-loss-weight 0.0 \
  --future-latent-loss-start-epoch 1 \
  --future-latent-max-batches 0 \
  --memory-report-every 0 \
  --memory-report-detail 0 \
  --memory-report-sync 0 \
  --training-stage policy \
  --upper-lr-scale 0.20 \
  --midcut-head-lr-scale 1.0 \
  --midcut-aux-loss-weight 0.03 \
  --midcut-aux-final-ratio 0.15 \
  --midcut-aux-decay-epochs 4 \
  --midcut-rollout-dynamics-loss-weight 0.03 \
  --midcut-rollout-delta-loss-weight 0.01 \
  --midcut-rollout-contrast-loss-weight 0.03 \
  --contract-mode layer_adapter \
  --layer-contract-loss-weight 1.0 \
  --layer-contract-final-action-loss-weight 0.0 \
  --layer-contract-final-action-lr-scale 0.3 \
  --layerwise-lr-min-scale 0.3 \
  --layer-contract-adapter-policy-lr-scale "${LAYER_CONTRACT_ADAPTER_POLICY_LR_SCALE}" \
  --layer-latent-loss-weight 1.0 \
  --layer-fm-probe-loss-weight 0.0 \
  --layer-event-loss-weight 0.05 \
  --layer-motion-loss-weight 0.03 \
  --layer-decoded-action-loss-weight 0.0 \
  --layer-contrast-loss-weight 0.03 \
  --layer-variance-loss-weight 0.05 \
  --layer-norm-loss-weight 0.02 \
  --layer-delta-match-loss-weight 0.15 \
  "$@"
