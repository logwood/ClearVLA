#!/usr/bin/env bash
# V103: one-stage typed predictive Flow-DINO/JEPA policy.
#
# This launcher keeps V100's strict visual ownership and V99's observable-flow
# supervision, then activates the post-V102 model repairs together:
#   - observation-only multi-resolution address lattice;
#   - positive soft flow-prior floor (not a hard address gate);
#   - real G/W/P residual values through typed AttnRes bridges;
#   - late protected high-resolution detail with query-conditioned cameras;
#   - exact-null T5/action-history conditioning;
#   - stateless phase and condition contexts on selector queries only;
#   - early masked raw future context and explicit teacher-chart delta target;
#   - +48 as an independent typed W->P context, never action step 24;
#   - one common bottom scale for G/W/P typed values (carrier scales do not
#     multiply across ownership boundaries);
#   - a fixed-cell soft flow floor plus an uncertainty-adaptive flow expert.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export OUT_DIR="${OUT_DIR:-runs/v103_typed_predictive_flow_jepa}"
export V103_BATCH_SIZE="${V103_BATCH_SIZE:-8}"
export V100_BATCH_SIZE="${V103_BATCH_SIZE}"
export FLOW_JEPA_PARENT_VERSION="${FLOW_JEPA_PARENT_VERSION:-v103}"
export CLEARVLA_REQUIRED_MODEL_CONTRACT="${CLEARVLA_REQUIRED_MODEL_CONTRACT:-v103}"
# The primary future loss is already the explicit teacher-chart delta under
# this contract; the older duplicate change auxiliary must remain disabled.
export FLOW_JEPA_FUTURE_CHANGE_WEIGHT=0

export FLOW_JEPA_ADDRESS_SLOTS="${FLOW_JEPA_ADDRESS_SLOTS:-4}"
export FLOW_JEPA_ADDRESS_ROUTE_DIM="${FLOW_JEPA_ADDRESS_ROUTE_DIM:-32}"
export FLOW_JEPA_ADDRESS_QUERY_CHUNK="${FLOW_JEPA_ADDRESS_QUERY_CHUNK:-4}"
export FLOW_JEPA_ADDRESS_FLOW_PRIOR_FLOOR="${FLOW_JEPA_ADDRESS_FLOW_PRIOR_FLOOR:-0.25}"
export ROLE_ATTNRES_KEY_DIM="${ROLE_ATTNRES_KEY_DIM:-32}"
export STATELESS_PHASE_COUNT="${STATELESS_PHASE_COUNT:-4}"
export STATELESS_PHASE_QUERY_SCALE="${STATELESS_PHASE_QUERY_SCALE:-0.10}"
export FLOW_JEPA_LATE_DETAIL_SCALE="${FLOW_JEPA_LATE_DETAIL_SCALE:-0.25}"

  printf '[v103] stage=single_end_to_end top=typed_predictive_flow_dino_jepa role_blocks=3/3/2 world_write=coarse_spatial address=soft_multires slots=%s flow_prior_floor=%s flow_floor_width=fixed_coarse_cell typed_bridges=g2w,w2p,p2mmdit typed_scale=single_bottom far_context=plus48_separate early_raw_mask=before_trainable_mixing future_target=teacher_delta condition_null=exact phase=stateless batch=%s stage1_init=off\n' \
  "${FLOW_JEPA_ADDRESS_SLOTS}" \
  "${FLOW_JEPA_ADDRESS_FLOW_PRIOR_FLOOR}" \
  "${V103_BATCH_SIZE}"

# The V48 base launcher still enables the historical recurrent consequence
# cell. V103's terminal P contracts and typed bottom bridge replace that
# parallel rollout/action path, so the command below overrides it explicitly.
exec bash "${SCRIPT_DIR}/current_v100_strict_complementary_flow_jepa.sh" \
  "$@" \
  --batch-size "${V103_BATCH_SIZE}" \
  --information-balanced-sampling 1 \
  --information-uniform-fraction 0.50 \
  --information-event-fraction 0.125 \
  --information-motion-quantile 0.70 \
  --horizon-weight-mode anchor_bands \
  --horizon-tail-emphasis 0.20 \
  --horizon-first-step-protection 0.05 \
  --trajectory-information-weight 0.0 \
  --flow-jepa-horizon-balance-mode per_horizon \
  --flow-jepa-teacher-balanced-target-mask 0 \
  --flow-jepa-teacher-mask-past-fraction 0.25 \
  --flow-jepa-teacher-mask-change-fraction 0.50 \
  --flow-jepa-source-aligned-raw-fusion 1 \
  --flow-jepa-policy-workspace-fixed-fusion 0 \
  --flow-jepa-world-anchor-write-only 0 \
  --flow-jepa-late-policy-detail 1 \
  --flow-jepa-late-policy-detail-scale "${FLOW_JEPA_LATE_DETAIL_SCALE}" \
  --flow-jepa-policy-workspace-horizon-pool 1 \
  --flow-jepa-soft-address-lattice 1 \
  --flow-jepa-address-slots "${FLOW_JEPA_ADDRESS_SLOTS}" \
  --flow-jepa-address-route-dim "${FLOW_JEPA_ADDRESS_ROUTE_DIM}" \
  --flow-jepa-address-query-chunk "${FLOW_JEPA_ADDRESS_QUERY_CHUNK}" \
  --flow-jepa-address-flow-prior-floor "${FLOW_JEPA_ADDRESS_FLOW_PRIOR_FLOOR}" \
  --role-attnres-enabled 1 \
  --role-attnres-key-dim "${ROLE_ATTNRES_KEY_DIM}" \
  --role-attnres-ground-to-world 1 \
  --role-attnres-world-to-policy 1 \
  --role-attnres-policy-to-mmdit 1 \
  --layer-shared-fm-probe 0 \
  --layer-recurrent-consequence 0 \
  --action-history-condition-dropout 0.10 \
  --action-history-condition-exact-null 1 \
  --action-history-proposal-detach 0 \
  --goal-condition-dropout 0.05 \
  --goal-condition-exact-null 1 \
  --stateless-phase-enabled 1 \
  --stateless-phase-count "${STATELESS_PHASE_COUNT}" \
  --stateless-phase-query-scale "${STATELESS_PHASE_QUERY_SCALE}" \
  --flow-jepa-predictive-change-contract 1
