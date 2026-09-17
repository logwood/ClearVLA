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

printf '[mainline-multitask] tasks=8 batch=%s threshold=%s cameras=high,right_wrist action_chart=right_arm_7d out=%s\n' \
  "${MAINLINE_BATCH_SIZE:-<config>}" \
  "${RDT_GRIPPER_EVENT_THRESHOLD}" \
  "${OUT_DIR:-<config>}"

exec bash "${SCRIPT_DIR}/train_mainline.sh" \
  --gripper-event-threshold "${RDT_GRIPPER_EVENT_THRESHOLD}" \
  --max-val-batches "${RDT_MAX_VAL_BATCHES:-64}" \
  "$@"
