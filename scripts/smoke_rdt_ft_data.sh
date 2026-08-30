#!/usr/bin/env bash
# CPU loader-only acceptance: no model, optimizer, checkpoint, or training step.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ARTIFACT_ROOT="${RDT_ARTIFACT_ROOT:-/data/senwang/data/rdt_ft_data}"

exec python -u -m clearvla.tools.smoke_mainline_data \
  --config "${RDT_DATA_CONFIG:-configs/mainline/rdt_right_arm_data_v1.json}" \
  --split "${RDT_SMOKE_SPLIT:-val}" \
  --batch-size "${RDT_SMOKE_BATCH_SIZE:-1}" \
  --episode-limit "${RDT_SMOKE_EPISODE_LIMIT:-1}" \
  --num-workers "${RDT_SMOKE_NUM_WORKERS:-0}" \
  --data-root "${RDT_DATA_ROOT:-/data/rdt-ft-data}" \
  --dino-cache "${RDT_DINO_CACHE:-${ARTIFACT_ROOT}/dinov2_rgb_336}" \
  --t5-condition "${RDT_T5_CONDITION:-${ARTIFACT_ROOT}/t5_v1_1_xxl_32.pt}" \
  --split-manifest "${RDT_SPLIT_MANIFEST:-${ARTIFACT_ROOT}/split_seed0.json}" \
  "$@"
