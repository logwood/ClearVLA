#!/usr/bin/env bash
# DINOv3 online CALVIN deployment helper.
# Usage:
#   ./scripts/launch_dinov3_online_calvin.sh preflight
#   RUN_DIR=/data/.../train-YYYYMMDD-HHMMSS ./scripts/launch_dinov3_online_calvin.sh train
set -euo pipefail

ROOT="/data/senwang/clearvla/checkouts/dinov3-online-aligned-vision-20260928"
PYTHON="${CLEARVLA_PYTHON:-/data/senwang/clearvla/third_party/dinov3/.venv/bin/python}"
CONFIG="${CLEARVLA_CONFIG:-${ROOT}/configs/mainline/dinov3_online_cumulative_calvin.json}"
MODEL="${DINOV3_MODEL:-/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
DTYPE="${CLEARVLA_DTYPE:-bf16}"
MICROBATCH="${CLEARVLA_DINOV3_MICROBATCH:-2}"
BATCH_SIZE="${CLEARVLA_BATCH_SIZE:-}"
EPOCHS="${CLEARVLA_EPOCHS:-}"
MODE="${1:-preflight}"
RUN_ROOT="/data/senwang/clearvla/experiments/dinov3-online-aligned-vision-20260928"

if [[ ! -d "${ROOT}" || ! -f "${CONFIG}" ]]; then
  echo "ClearVLA DINOv3 checkout/config is missing" >&2
  exit 2
fi
if [[ ! -f "${MODEL}/config.json" || ! -f "${MODEL}/model.safetensors" ]]; then
  echo "Local DINOv3 HF model is incomplete: ${MODEL}" >&2
  exit 2
fi

case "${MODE}" in
  preflight)
    OUTPUT="${RUN_DIR:-${RUN_ROOT}/preflight-$(date +%Y%m%d-%H%M%S)}"
    if [[ -e "${OUTPUT}" ]]; then
      echo "Refusing to overwrite preflight evidence: ${OUTPUT}" >&2
      exit 2
    fi
    cd "${ROOT}"
    cmd=(env CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${ROOT}" "${PYTHON}" -u
      scripts/probe_dinov3_deployment_ready.py
      --config "${CONFIG}"
      --model "${MODEL}"
      --device cuda
      --dtype "${DTYPE}"
      --microbatch "${MICROBATCH}"
      --steps "${PREFLIGHT_STEPS:-1}"
      --batch-size 1
      --output "${OUTPUT}")
    exec "${cmd[@]}"
    ;;
  train)
    OUTPUT="${RUN_DIR:-${RUN_ROOT}/train-$(date +%Y%m%d-%H%M%S)}"
    if [[ -e "${OUTPUT}" ]]; then
      echo "Refusing to overwrite training output: ${OUTPUT}" >&2
      exit 2
    fi
    cd "${ROOT}"
    cmd=(env CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${ROOT}" "${PYTHON}" -u
      -m clearvla.mainline.train
      --config "${CONFIG}"
      --device cuda
      --dtype "${DTYPE}"
      --output-dir "${OUTPUT}"
      --dinov3-model "${MODEL}"
      --dinov3-microbatch "${MICROBATCH}")
    if [[ -n "${EPOCHS}" ]]; then
      cmd+=(--epochs "${EPOCHS}")
    fi
    if [[ -n "${BATCH_SIZE}" ]]; then
      cmd+=(--batch-size "${BATCH_SIZE}")
    fi
    exec "${cmd[@]}"
    ;;
  *)
    echo "Usage: ${0} {preflight|train}" >&2
    exit 2
    ;;
esac
