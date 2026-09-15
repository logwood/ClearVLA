#!/usr/bin/env bash
# Build one manifest-selected DINO cache scope and prove a typed batch; no model/training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${RDT_DATA_ROOT:-/data/rdt-ft-data}"
ARTIFACT_ROOT="${RDT_ARTIFACT_ROOT:-/data/senwang/data/rdt_ft_data}"
SPLIT_PATH="${RDT_SPLIT_MANIFEST:-${ARTIFACT_ROOT}/split_seed0.json}"
T5_PATH="${RDT_T5_CONDITION:-${ARTIFACT_ROOT}/t5_v1_1_xxl_32.pt}"
SMOKE_SPLIT="${RDT_SMOKE_SPLIT:-val}"
EPISODE_LIMIT="${RDT_SMOKE_EPISODE_LIMIT:-1}"
DINO_CACHE="${RDT_SMOKE_DINO_CACHE:-${ARTIFACT_ROOT}/bounded_smoke_${SMOKE_SPLIT}_${EPISODE_LIMIT}/dinov2_rgb_336}"
SOURCE_COMMIT="$(git rev-parse HEAD)"
REPORT_PATH="${RDT_SMOKE_REPORT:-$(dirname "${DINO_CACHE}")/typed_batch_smoke_$(date +%Y%m%d_%H%M%S).json}"

case "${SMOKE_SPLIT}" in
  train|val|test|external_test) ;;
  *) printf 'RDT_SMOKE_SPLIT must be train, val, test, or external_test\n' >&2; exit 2 ;;
esac
if ! [[ "${EPISODE_LIMIT}" =~ ^[1-8]$ ]]; then
  printf 'RDT_SMOKE_EPISODE_LIMIT must be an integer from 1 through 8\n' >&2
  exit 2
fi

# Metadata and the instruction bank remain global: a one-episode DINO scope
# must not weaken source/split/language identity.
RDT_PREPARE_THROUGH=language \
RDT_DATA_ROOT="${DATA_ROOT}" \
RDT_ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
RDT_SPLIT_MANIFEST="${SPLIT_PATH}" \
RDT_T5_CONDITION="${T5_PATH}" \
bash "${SCRIPT_DIR}/prepare_rdt_ft_data.sh"

DINO_ARGS=(
  --data-root "${DATA_ROOT}"
  --glob '**/*.hdf5'
  --state-key observations/qpos
  --out-dir "${DINO_CACHE}"
  --cameras high left_wrist right_wrist
  --camera-key high=observations/images/cam_high
  --camera-key left_wrist=observations/images/cam_left_wrist
  --camera-key right_wrist=observations/images/cam_right_wrist
  --cache-resize 336 336
  --dinov2-model "${RDT_DINOV2_MODEL:-facebook/dinov2-base}"
  --split-manifest "${SPLIT_PATH}"
  --manifest-split "${SMOKE_SPLIT}"
  --max-episodes "${EPISODE_LIMIT}"
  --batch-size "${RDT_DINO_BATCH_SIZE:-32}"
  --device "${RDT_DINO_DEVICE:-auto}"
  --dtype "${RDT_DINO_DTYPE:-bf16}"
)
if [[ "${RDT_LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  DINO_ARGS+=(--dinov2-local-files-only)
fi
python -u -m clearvla.cli.build_dinov2_token_cache "${DINO_ARGS[@]}"

RDT_DATA_ROOT="${DATA_ROOT}" \
RDT_ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
RDT_SPLIT_MANIFEST="${SPLIT_PATH}" \
RDT_T5_CONDITION="${T5_PATH}" \
RDT_DINO_CACHE="${DINO_CACHE}" \
RDT_SMOKE_SPLIT="${SMOKE_SPLIT}" \
RDT_SMOKE_EPISODE_LIMIT="${EPISODE_LIMIT}" \
bash "${SCRIPT_DIR}/smoke_rdt_ft_data.sh" \
  --source-commit "${SOURCE_COMMIT}" \
  --output "${REPORT_PATH}"
