#!/usr/bin/env bash
# V101: information-balanced, long-horizon single-stage Flow-DINO JEPA policy.
#
# This inherits V100's strict grounding -> world -> policy visual ownership and
# changes only the plateau-related contracts:
#   1. half of every epoch remains an unbiased shuffled lane; the remainder is
#      bounded motion/event coverage from action-only dataset statistics;
#   2. action flow is weighted by the real 1-4 / 5-12 / 13-24 anchor bands;
#   3. every JEPA horizon is reduced independently before horizons are averaged;
#   4. future teacher change chooses loss locations only, with exact disjoint
#      past/change/uniform quotas and no future feature entering the forward;
#   5. source-indexed flow detail fuses with the matching source DINO chart,
#      while the policy workspace removes its historical 0.10 bottleneck;
#   6. small independent condition dropout prevents smooth action history or a
#      fixed language condition from becoming an exclusive shortcut.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export OUT_DIR="${OUT_DIR:-runs/v101_information_balanced_long_horizon}"
export V101_BATCH_SIZE="${V101_BATCH_SIZE:-8}"
export V100_BATCH_SIZE="${V101_BATCH_SIZE}"
export FLOW_JEPA_PARENT_VERSION="${FLOW_JEPA_PARENT_VERSION:-v101}"

if [[ "${FLOW_JEPA_PARENT_VERSION}" == "v101" ]]; then
  printf '[v101] stage=single_end_to_end top=strict_complementary_raw_flow_jepa sampling=uniform:0.50/event:0.125/motion:0.375 action_bands=4/12/24 horizon_reduce=per_horizon teacher_mask=past:0.25/change:0.50/uniform:0.25 raw_fusion=source_aligned policy_workspace=fixed_variance history_drop=0.10 goal_drop=0.05 batch=%s stage1_init=off\n' \
    "${V101_BATCH_SIZE}"
fi

exec bash "${SCRIPT_DIR}/current_v100_strict_complementary_flow_jepa.sh" \
  "$@" \
  --batch-size "${V101_BATCH_SIZE}" \
  --information-balanced-sampling 1 \
  --information-uniform-fraction 0.50 \
  --information-event-fraction 0.125 \
  --information-motion-quantile 0.70 \
  --horizon-weight-mode anchor_bands \
  --horizon-tail-emphasis 0.20 \
  --horizon-first-step-protection 0.05 \
  --trajectory-information-weight 0.0 \
  --flow-jepa-horizon-balance-mode per_horizon \
  --flow-jepa-teacher-balanced-target-mask 1 \
  --flow-jepa-teacher-mask-past-fraction 0.25 \
  --flow-jepa-teacher-mask-change-fraction 0.50 \
  --flow-jepa-source-aligned-raw-fusion 1 \
  --flow-jepa-policy-workspace-fixed-fusion 1 \
  --action-history-condition-dropout 0.10 \
  --goal-condition-dropout 0.05
