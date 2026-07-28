#!/usr/bin/env bash
# V95 Stage2 policy training must consume the checkpoint produced by V95 Stage1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V95_STAGE1_CHECKPOINT="${V95_STAGE1_CHECKPOINT:-}"
if [[ -z "${V95_STAGE1_CHECKPOINT}" || ! -f "${V95_STAGE1_CHECKPOINT}" ]]; then
  echo "[v95-policy] V95_STAGE1_CHECKPOINT must point to best_stage1_representation.pt" >&2
  exit 2
fi

export STAGE1_CHECKPOINT="${V95_STAGE1_CHECKPOINT}"
export V95_TRAINING_STAGE=policy
export OUT_DIR="${OUT_DIR:-runs/v95_flow_dino_jepa_policy}"

exec bash "${SCRIPT_DIR}/current_v95_flow_dino_jepa.sh" "$@"
