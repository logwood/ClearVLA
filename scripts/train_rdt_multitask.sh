#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ADOPTED_RDT_GRIPPER_EVENT_THRESHOLD="0.18310546875"
if [[ -n "${RDT_GRIPPER_EVENT_THRESHOLD:-}" && "${RDT_GRIPPER_EVENT_THRESHOLD}" != "${ADOPTED_RDT_GRIPPER_EVENT_THRESHOLD}" ]]; then
  printf 'RDT_GRIPPER_EVENT_THRESHOLD is fixed at %s by the train-only p95 audit; use a separate experimental config for an ablation\n' \
    "${ADOPTED_RDT_GRIPPER_EVENT_THRESHOLD}" >&2
  exit 2
fi
readonly RDT_GRIPPER_EVENT_THRESHOLD="${ADOPTED_RDT_GRIPPER_EVENT_THRESHOLD}"

for argument in "$@"; do
  if [[ "${argument}" == "--gripper-event-threshold" || "${argument}" == --gripper-event-threshold=* ]]; then
    printf '%s\n' 'train_rdt_multitask.sh does not allow a threshold override; use a separate experimental config for an ablation' >&2
    exit 2
  fi
done

export MAINLINE_CONFIG="${MAINLINE_CONFIG:-configs/mainline/rdt_multitask8_data_v1.json}"
export MAINLINE_BATCH_SIZE="${MAINLINE_BATCH_SIZE:-8}"
export MAINLINE_NUM_WORKERS="${MAINLINE_NUM_WORKERS:-4}"
export DATA_ROOT="${DATA_ROOT:-/data/rdt-ft-data}"
export CACHE_DIR="${CACHE_DIR:-/data/senwang/data/rdt_ft_data/multitask_v1/decoded_rgb_336_not_materialized}"
export DINO_CACHE_DIR="${DINO_CACHE_DIR:-/data/senwang/data/rdt_ft_data/multitask_v1/dinov2_rgb_336}"
export T5_CONDITION_PATH="${T5_CONDITION_PATH:-/data/senwang/data/rdt_ft_data/multitask_v1/t5_v1_1_xxl_32.pt}"
export OUT_DIR="${OUT_DIR:-runs/clearvla_rdt_multitask8_v1}"

printf '[mainline-multitask] tasks=8 batch=%s threshold=%s cameras=high,right_wrist action_chart=right_arm_7d out=%s\n' \
  "${MAINLINE_BATCH_SIZE}" \
  "${RDT_GRIPPER_EVENT_THRESHOLD}" \
  "${OUT_DIR}"

exec bash "${SCRIPT_DIR}/train_mainline.sh" \
  --gripper-event-threshold "${RDT_GRIPPER_EVENT_THRESHOLD}" \
  --max-val-batches "${RDT_MAX_VAL_BATCHES:-64}" \
  "$@"
