#!/usr/bin/env bash
# V114: one shared factual P1 read per horizon, four basis-aware P2 consumers.
#
# The full two-camera 8x8/4-slot/49-candidate lattice and all four 3x3 factual
# glimpses are preserved.  P1 cannot read the noisy action.  P2 retains all
# four action basis tokens and protects RGB/detail base and precision carriers
# outside the optional typed-delta router.  FP32 owns logits, softmax and
# geometry; BF16 owns factual value contraction.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export OUT_DIR="${OUT_DIR:-runs/v114_shared_factual_utility_precision}"
export V114_BATCH_SIZE="${V114_BATCH_SIZE:-8}"
export V113_BATCH_SIZE="${V114_BATCH_SIZE}"
export FLOW_JEPA_PARENT_VERSION="${FLOW_JEPA_PARENT_VERSION:-v114}"
export CLEARVLA_REQUIRED_MODEL_CONTRACT="${CLEARVLA_REQUIRED_MODEL_CONTRACT:-v114}"

# Full action validation still covers the complete validation loader.  These
# small budgets only cap auxiliary fan-out: proposal adds one five-step sample,
# execution adds four matched five-step samples, and representation adds one
# teacher-forced pass.  Full path interventions remain available through the
# standalone probe scripts instead of being paid every epoch.
V114_EVAL_SAMPLING_DIAGNOSTIC_BATCHES="${V114_EVAL_SAMPLING_DIAGNOSTIC_BATCHES:-4}"
V114_EVAL_PROPOSAL_ABLATION_BATCHES="${V114_EVAL_PROPOSAL_ABLATION_BATCHES:-2}"
V114_EVAL_EXECUTION_ABLATION_BATCHES="${V114_EVAL_EXECUTION_ABLATION_BATCHES:-1}"
V114_EVAL_REPRESENTATION_BATCHES="${V114_EVAL_REPRESENTATION_BATCHES:-4}"

for value_name in \
  V114_EVAL_SAMPLING_DIAGNOSTIC_BATCHES \
  V114_EVAL_PROPOSAL_ABLATION_BATCHES \
  V114_EVAL_EXECUTION_ABLATION_BATCHES \
  V114_EVAL_REPRESENTATION_BATCHES; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[v114] ${value_name} must be a positive integer, got ${value}" >&2
    exit 2
  fi
done

printf '[v114] base=v113 P1=shared_factual_24x4 P1_action=noisy_excluded P2=basis_aware_96 P2_value=rgb+detail_base+precision posterior=single_joint microgrid=3x3_tiled dtype=fp32_route+bf16_value batch=%s eval_aux=%s/%s/%s/%s\n' \
  "${V114_BATCH_SIZE}" \
  "${V114_EVAL_SAMPLING_DIAGNOSTIC_BATCHES}" \
  "${V114_EVAL_PROPOSAL_ABLATION_BATCHES}" \
  "${V114_EVAL_EXECUTION_ABLATION_BATCHES}" \
  "${V114_EVAL_REPRESENTATION_BATCHES}"

exec bash "${SCRIPT_DIR}/current_v113_functional_mainline_routing.sh" \
  --eval-sampling-diagnostic-batches \
  "${V114_EVAL_SAMPLING_DIAGNOSTIC_BATCHES}" \
  --eval-proposal-ablation-batches \
  "${V114_EVAL_PROPOSAL_ABLATION_BATCHES}" \
  --eval-execution-ablation-batches \
  "${V114_EVAL_EXECUTION_ABLATION_BATCHES}" \
  --eval-representation-batches \
  "${V114_EVAL_REPRESENTATION_BATCHES}" \
  "$@" \
  --flow-jepa-utility-precision-mainline 1 \
  --flow-jepa-action-free-world-factual 1 \
  --flow-jepa-address-query-batch-budget 32 \
  --flow-jepa-microgrid-tile 3 \
  --flow-jepa-p1-mixed-precision 1 \
  --flow-jepa-checkpoint-min-batch 4
