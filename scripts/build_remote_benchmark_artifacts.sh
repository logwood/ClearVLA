#!/usr/bin/env bash
# Build /data-owned decoded/DINO caches and a frozen per-instruction T5 bank.
# This is intentionally separate from conversion: full CALVIN caches are large.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${CLEARVLA_BENCH_DATA_ROOT:-/data/senwang/data}"
CALVIN_DATA="${CLEARVLA_CALVIN_DATA:-${DATA_ROOT}/calvin}"
LIBERO_DATA="${CLEARVLA_LIBERO_DATA:-${DATA_ROOT}/libero}"
PYTHON_BIN="${CLEARVLA_ARTIFACT_PYTHON:-${CLEARVLA_SIM_ENV:-/data/senwang/envs/clearvla-sim}/bin/python}"
MODEL_NAME="${CLEARVLA_T5_MODEL:-google/t5-v1_1-xxl}"
DEVICE="${CLEARVLA_ARTIFACT_DEVICE:-auto}"
DTYPE="${CLEARVLA_ARTIFACT_DTYPE:-bf16}"
MAX_TOKENS="${CLEARVLA_T5_MAX_TOKENS:-32}"
T5_LOCAL_FILES_ONLY="${CLEARVLA_T5_LOCAL_FILES_ONLY:-0}"
DINO_LOCAL_FILES_ONLY="${CLEARVLA_DINOV2_LOCAL_FILES_ONLY:-0}"
DINO_BATCH_SIZE="${CLEARVLA_DINO_BATCH_SIZE:-32}"
HF_HOME_ROOT="${CLEARVLA_HF_HOME:-${DATA_ROOT}/huggingface}"
HF_CACHE_ROOT="${CLEARVLA_HUGGINGFACE_HUB_CACHE:-${HF_HOME_ROOT}/hub}"
TORCH_HOME_ROOT="${CLEARVLA_TORCH_HOME:-${DATA_ROOT}/torch}"
REBUILD=0
BUILD_CACHE=0
BUILD_LANGUAGE=0

usage() {
  cat >&2 <<'EOF'
usage: build_remote_benchmark_artifacts.sh TARGET [--language] [--caches] [--rebuild]
TARGET: calvin | libero

The selected converted root must already contain instructions.json. Outputs,
Hugging Face caches, decoded images and DINO tokens are kept under /data.
Set CLEARVLA_ARTIFACT_DEVICE=cpu and CLEARVLA_ARTIFACT_DTYPE=fp32 for a CPU
cache build (the default is CUDA/bfloat16 on the configured server).
Set CLEARVLA_T5_LOCAL_FILES_ONLY=1 or CLEARVLA_DINOV2_LOCAL_FILES_ONLY=1 to
fail fast when the corresponding Hugging Face model is not already cached.
Set CLEARVLA_DINO_BATCH_SIZE to tune the bounded DINO frame batch for the
available GPU; the default 32 is conservative.
EOF
}

if [[ $# -lt 1 ]]; then usage; exit 2; fi
TARGET="$1"
shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --language) BUILD_LANGUAGE=1 ;;
    --caches) BUILD_CACHE=1 ;;
    --rebuild) REBUILD=1 ;;
    *) usage; exit 2 ;;
  esac
  shift
done
if [[ "${BUILD_LANGUAGE}" == 0 && "${BUILD_CACHE}" == 0 ]]; then
  BUILD_LANGUAGE=1
fi

case "${TARGET}" in
  calvin)
    CONVERTED="${CLEARVLA_CALVIN_CONVERTED:-${CALVIN_DATA}/converted/abc_d}"
    CACHE_ROOT="${CLEARVLA_CALVIN_CACHE_ROOT:-${CALVIN_DATA}/caches/abc_d}"
    LANGUAGE="${CLEARVLA_CALVIN_LANGUAGE_BANK:-${CALVIN_DATA}/language/t5_xxl_bank.pt}"
    ;;
  libero)
    SUITE_SLUG="${CLEARVLA_LIBERO_SUITE_SLUG:-libero_spatial}"
    CONVERTED="${CLEARVLA_LIBERO_CONVERTED:-${LIBERO_DATA}/converted/${SUITE_SLUG}}"
    CACHE_ROOT="${CLEARVLA_LIBERO_CACHE_ROOT:-${LIBERO_DATA}/caches/${SUITE_SLUG}}"
    LANGUAGE="${CLEARVLA_LIBERO_LANGUAGE_BANK:-${LIBERO_DATA}/language/${SUITE_SLUG}_t5_xxl_bank.pt}"
    ;;
  *) usage; exit 2 ;;
esac

[[ -x "${PYTHON_BIN}" ]] || { echo "artifact Python is missing: ${PYTHON_BIN}" >&2; exit 2; }
[[ -f "${CONVERTED}/instructions.json" ]] || {
  echo "converted benchmark root is missing instructions.json: ${CONVERTED}" >&2
  exit 3
}
mkdir -p "${CACHE_ROOT}" "$(dirname "${LANGUAGE}")" \
  "${HF_HOME_ROOT}" "${HF_CACHE_ROOT}" "${TORCH_HOME_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME_ROOT}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_ROOT}"
export TORCH_HOME="${TORCH_HOME_ROOT}"

if [[ "${BUILD_LANGUAGE}" == 1 ]]; then
  if [[ -f "${LANGUAGE}" && "${REBUILD}" == 0 ]]; then
    echo "[artifact] reuse language bank ${LANGUAGE}"
  else
    language_local_args=()
    [[ "${T5_LOCAL_FILES_ONLY}" == 1 ]] && language_local_args+=(--local-files-only)
    "${PYTHON_BIN}" -m clearvla.benchmarks.language_bank \
      --inventory "${CONVERTED}/instructions.json" \
      --output "${LANGUAGE}" \
      --model "${MODEL_NAME}" \
      --max-tokens "${MAX_TOKENS}" \
      --device "${DEVICE}" \
      "${language_local_args[@]}"
  fi
fi

if [[ "${BUILD_CACHE}" == 1 ]]; then
  decoded="${CACHE_ROOT}/decoded_336"
  dino="${CACHE_ROOT}/dinov2_base_336"
  rebuild_args=()
  [[ "${REBUILD}" == 1 ]] && rebuild_args+=(--rebuild)
  dino_local_args=()
  [[ "${DINO_LOCAL_FILES_ONLY}" == 1 ]] && dino_local_args+=(--dinov2-local-files-only)
  "${PYTHON_BIN}" -m clearvla.cli.build_decoded_image_cache \
    --data-root "${CONVERTED}" \
    --cache-dir "${decoded}" \
    --cameras top wrist \
    --top-key observations/images/cam_high \
    --wrist-key observations/images/cam_right_wrist \
    --resize 336 336 \
    "${rebuild_args[@]}"
  "${PYTHON_BIN}" -m clearvla.cli.build_dinov2_token_cache \
    --data-root "${CONVERTED}" \
    --decoded-image-cache-dir "${decoded}" \
    --out-dir "${dino}" \
    --cameras top wrist \
    --state-key state \
    --top-key observations/images/cam_high \
    --wrist-key observations/images/cam_right_wrist \
    --cache-resize 336 336 \
    --dinov2-model facebook/dinov2-base \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --batch-size "${DINO_BATCH_SIZE}" \
    "${dino_local_args[@]}" \
    "${rebuild_args[@]}"
fi

printf '[clearvla-benchmark-artifacts] target=%s converted=%s language=%s caches=%s\n' \
  "${TARGET}" "${CONVERTED}" "${LANGUAGE}" "${BUILD_CACHE}"

