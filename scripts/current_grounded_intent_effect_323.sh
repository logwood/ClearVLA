#!/usr/bin/env bash
# Grounded Intent-Effect 3-2-3 mainline.
#
# "v119" is only the run/log label.  Source selection and checkpoint identity
# use the capability name grounded_intent_effect_323.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

for argument in "$@"; do
  case "${argument}" in
    --resume|--resume=*)
      echo "[grounded-intent-effect-323] changed top ownership requires a fresh run" >&2
      exit 2
      ;;
  esac
done

export OUT_DIR="${OUT_DIR:-runs/v119_grounded_intent_effect_323}"
export GROUNDED_323_BATCH_SIZE="${GROUNDED_323_BATCH_SIZE:-8}"
export V116_BATCH_SIZE="${GROUNDED_323_BATCH_SIZE}"
export FLOW_JEPA_PARENT_VERSION=grounded_intent_effect_323
export CLEARVLA_REQUIRED_MODEL_CONTRACT=grounded_intent_effect_323

# Current server storage layout.  The raw HDF5 default was already under
# /data and remains unchanged; only the former /home cache/weight defaults
# move to Sen Wang's storage roots.
export CLEARVLA_DATA_CACHE_ROOT="${CLEARVLA_DATA_CACHE_ROOT:-/data/senwang/data}"
export CLEARVLA_CHECKPOINT_ROOT="${CLEARVLA_CHECKPOINT_ROOT:-/data/senwang/checkpoint}"
export DATA_ROOT="${DATA_ROOT:-/data/liang.zhang/dataset/grab_pen_single/grab_pen_single}"
export CACHE_DIR="${CACHE_DIR:-${CLEARVLA_DATA_CACHE_ROOT}/cache_336}"
export DINO_CACHE_DIR="${DINO_CACHE_DIR:-${CLEARVLA_DATA_CACHE_ROOT}/dinov2_cache_336}"
export T5_CONDITION_PATH="${T5_CONDITION_PATH:-${CLEARVLA_CHECKPOINT_ROOT}/grasp_pen_embed.pt}"

printf '[v119] capability=grounded_intent_effect_323 topology=3-2-3 S=stateless_set_intent W=four_interval_object_effect P1=single_precision_read P2=bounded_effect_read P3=consequence_precision+temporal teacher=G_aligned_no_grad batch=%s\n' \
  "${GROUNDED_323_BATCH_SIZE}"
printf '[v119-paths] data=%s decoded_cache=%s dino_cache=%s t5=%s\n' \
  "${DATA_ROOT}" "${CACHE_DIR}" "${DINO_CACHE_DIR}" \
  "${T5_CONDITION_PATH}"

exec bash "${SCRIPT_DIR}/current_v116_supervised_effect_mainline.sh" \
  "$@" \
  --flow-jepa-grounded-intent-effect-mainline 1 \
  --flow-jepa-differential-intent-effect-mainline 0 \
  --flow-jepa-stateless-intent-controller 0 \
  --flow-jepa-window-effect-bank 0 \
  --flow-jepa-effect-read-in-p2 0 \
  --flow-jepa-future-slots 4
