#!/usr/bin/env bash
# Prepare immutable external artifacts for /data/rdt-ft-data without starting training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${RDT_DATA_ROOT:-/data/rdt-ft-data}"
ARTIFACT_ROOT="${RDT_ARTIFACT_ROOT:-/data/senwang/data/rdt_ft_data}"
AUDIT_PATH="${RDT_AUDIT_PATH:-${ARTIFACT_ROOT}/audit_full.json}"
SPLIT_PATH="${RDT_SPLIT_MANIFEST:-${ARTIFACT_ROOT}/split_seed0.json}"
T5_PATH="${RDT_T5_CONDITION:-${ARTIFACT_ROOT}/t5_v1_1_xxl_32.pt}"
DINO_CACHE="${RDT_DINO_CACHE:-${ARTIFACT_ROOT}/dinov2_rgb_336}"
DECODED_CACHE="${RDT_DECODED_CACHE:-${ARTIFACT_ROOT}/decoded_rgb_336}"
THROUGH="${RDT_PREPARE_THROUGH:-manifest}"

case "${THROUGH}" in
  manifest|language|dino) ;;
  *) printf 'RDT_PREPARE_THROUGH must be manifest, language, or dino\n' >&2; exit 2 ;;
esac

mkdir -p "${ARTIFACT_ROOT}"

OVERWRITE_ARGS=()
if [[ "${RDT_OVERWRITE_METADATA:-0}" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite)
fi

if [[ ! -e "${AUDIT_PATH}" || "${RDT_OVERWRITE_METADATA:-0}" == "1" ]]; then
  python -u -m clearvla.tools.audit_rdt_ft_data \
    "${DATA_ROOT}" \
    --format json \
    --output "${AUDIT_PATH}" \
    "${OVERWRITE_ARGS[@]}"
else
  printf '[rdt-prepare] reuse audit=%s; loader acceptance will recheck inventory identity\n' \
    "${AUDIT_PATH}"
fi

if [[ ! -e "${SPLIT_PATH}" || "${RDT_OVERWRITE_METADATA:-0}" == "1" ]]; then
  python -u -m clearvla.tools.build_rdt_split_manifest \
    "${DATA_ROOT}" \
    --glob '**/*.hdf5' \
    --minimum-episode-length 73 \
    --seed "${RDT_SPLIT_SEED:-0}" \
    --output "${SPLIT_PATH}" \
    "${OVERWRITE_ARGS[@]}"
else
  printf '[rdt-prepare] reuse split=%s; loader acceptance will verify its content digest\n' \
    "${SPLIT_PATH}"
fi

if [[ "${THROUGH}" == "manifest" ]]; then
  printf '[rdt-prepare] metadata complete; no language encoding, image cache, or DINO cache was started\n'
  exit 0
fi

if [[ ! -e "${T5_PATH}" ]]; then
  T5_ARGS=(
    --data-root "${DATA_ROOT}"
    --glob '**/*.hdf5'
    --output "${T5_PATH}"
    --device "${RDT_T5_DEVICE:-auto}"
    --dtype "${RDT_T5_DTYPE:-bf16}"
    --batch-size "${RDT_T5_BATCH_SIZE:-1}"
    --policy-max-tokens 32
  )
  if [[ "${RDT_LOCAL_FILES_ONLY:-0}" == "1" ]]; then
    T5_ARGS+=(--local-files-only)
  fi
  python -u -m clearvla.tools.build_t5_instruction_cache "${T5_ARGS[@]}"
else
  printf '[rdt-prepare] reuse T5 bank=%s; loader acceptance will verify all mappings\n' \
    "${T5_PATH}"
fi

if [[ "${THROUGH}" == "language" ]]; then
  printf '[rdt-prepare] language complete; no RGB or DINO cache was started\n'
  exit 0
fi

if [[ "${RDT_CONFIRM_MULTI_TB_DINO_CACHE:-}" != "YES" ]]; then
  printf 'Refusing the potentially multi-TB DINO build. Inspect %s and set RDT_CONFIRM_MULTI_TB_DINO_CACHE=YES after selecting storage/scope.\n' \
    "${AUDIT_PATH}" >&2
  exit 2
fi

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
  --manifest-split all
  --batch-size "${RDT_DINO_BATCH_SIZE:-32}"
  --device "${RDT_DINO_DEVICE:-auto}"
  --dtype "${RDT_DINO_DTYPE:-bf16}"
)
if [[ "${RDT_DINO_MAX_EPISODES:-0}" != "0" ]]; then
  DINO_ARGS+=(--max-episodes "${RDT_DINO_MAX_EPISODES:-0}")
fi
if [[ "${RDT_LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  DINO_ARGS+=(--dinov2-local-files-only)
fi
python -u -m clearvla.cli.build_dinov2_token_cache "${DINO_ARGS[@]}"

if [[ "${RDT_BUILD_DECODED_CACHE:-0}" == "1" ]]; then
  if [[ "${RDT_CONFIRM_MULTI_TB_DECODED_CACHE:-}" != "YES" ]]; then
    printf 'Decoded RGB materialization also requires RDT_CONFIRM_MULTI_TB_DECODED_CACHE=YES.\n' >&2
    exit 2
  fi
  DECODED_ARGS=(
    --data-root "${DATA_ROOT}"
    --glob '**/*.hdf5'
    --state-key observations/qpos
    --cache-dir "${DECODED_CACHE}"
    --cameras high left_wrist right_wrist
    --camera-key high=observations/images/cam_high
    --camera-key left_wrist=observations/images/cam_left_wrist
    --camera-key right_wrist=observations/images/cam_right_wrist
    --resize 336 336
    --split-manifest "${SPLIT_PATH}"
    --manifest-split all
  )
  if [[ "${RDT_DINO_MAX_EPISODES:-0}" != "0" ]]; then
    DECODED_ARGS+=(--max-episodes "${RDT_DINO_MAX_EPISODES:-0}")
  fi
  python -u -m clearvla.cli.build_decoded_image_cache "${DECODED_ARGS[@]}"
fi

printf '[rdt-prepare] requested external artifacts completed; no training was started\n'
