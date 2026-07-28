#!/usr/bin/env bash
# Dataset-only diagnostic for the V95 Flow-DINO matching geometry.
# No checkpoint is loaded and no training state is changed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Keep these defaults aligned with current_v48_justok.sh and
# current_v95_flow_dino_jepa.sh. Every value remains overridable on the server.
DATA_ROOT="${DATA_ROOT:-/data/liang.zhang/dataset/grab_pen_single/grab_pen_single}"
DINO_CACHE_DIR="${DINO_CACHE_DIR:-/home/sen.wang/workspace/robotics/clear/data/dinov2_cache_336}"
PROBE_OUTPUT="${PROBE_OUTPUT:-runs/v95_flow_dataset_probe.json}"
PROBE_SPLIT="${PROBE_SPLIT:-val}"
PROBE_MAX_EPISODES="${PROBE_MAX_EPISODES:-0}"
PROBE_STRIDE="${PROBE_STRIDE:-1}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-16}"
PROBE_DEVICE="${PROBE_DEVICE:-auto}"
PROBE_BOOTSTRAP_SAMPLES="${PROBE_BOOTSTRAP_SAMPLES:-2000}"

FLOW_JEPA_GRID_SIZE="${FLOW_JEPA_GRID_SIZE:-8}"
FLOW_JEPA_CORR_RADIUS="${FLOW_JEPA_CORR_RADIUS:-2}"
FLOW_JEPA_CORR_TEMPERATURE="${FLOW_JEPA_CORR_TEMPERATURE:-0.07}"
FLOW_JEPA_WINDOW_OFFSETS="${FLOW_JEPA_WINDOW_OFFSETS:-4 12 24}"
FLOW_JEPA_STAGE_OFFSET="${FLOW_JEPA_STAGE_OFFSET:-48}"
FLOW_JEPA_HISTORY_OFFSETS="${FLOW_JEPA_HISTORY_OFFSETS:--8 -4 0}"
ACTION_HISTORY_OFFSETS="${ACTION_HISTORY_OFFSETS:--24 -16 -12 -8 -6 -4 -2 -1}"
# Offset +1 is diagnostic-only and exposes the actual per-frame change scale.
PROBE_EXTRA_TARGET_OFFSETS="${PROBE_EXTRA_TARGET_OFFSETS:-1}"

printf '[v95-flow-data-probe] split=%s data=%s cache=%s grid=%s radius=%s temperature=%s history=%s windows=%s stage=%s extra=%s output=%s\n' \
  "${PROBE_SPLIT}" \
  "${DATA_ROOT}" \
  "${DINO_CACHE_DIR}" \
  "${FLOW_JEPA_GRID_SIZE}" \
  "${FLOW_JEPA_CORR_RADIUS}" \
  "${FLOW_JEPA_CORR_TEMPERATURE}" \
  "${FLOW_JEPA_HISTORY_OFFSETS}" \
  "${FLOW_JEPA_WINDOW_OFFSETS}" \
  "${FLOW_JEPA_STAGE_OFFSET}" \
  "${PROBE_EXTRA_TARGET_OFFSETS}" \
  "${PROBE_OUTPUT}"

exec python -u -m clearvla.tools.probe_flow_dino_dataset_motion \
  --data-root "${DATA_ROOT}" \
  --dino-cache "${DINO_CACHE_DIR}" \
  --output "${PROBE_OUTPUT}" \
  --glob '*.hdf5' \
  --split "${PROBE_SPLIT}" \
  --train-episodes 63 \
  --val-episodes 5 \
  --test-episodes 5 \
  --max-episodes "${PROBE_MAX_EPISODES}" \
  --history-offsets "${FLOW_JEPA_HISTORY_OFFSETS}" \
  --action-history-offsets "${ACTION_HISTORY_OFFSETS}" \
  --window-offsets "${FLOW_JEPA_WINDOW_OFFSETS}" \
  --stage-offset "${FLOW_JEPA_STAGE_OFFSET}" \
  --extra-target-offsets "${PROBE_EXTRA_TARGET_OFFSETS}" \
  --grid-size "${FLOW_JEPA_GRID_SIZE}" \
  --correlation-radius "${FLOW_JEPA_CORR_RADIUS}" \
  --correlation-temperature "${FLOW_JEPA_CORR_TEMPERATURE}" \
  --stride "${PROBE_STRIDE}" \
  --batch-size "${PROBE_BATCH_SIZE}" \
  --bootstrap-samples "${PROBE_BOOTSTRAP_SAMPLES}" \
  --seed 0 \
  --device "${PROBE_DEVICE}" \
  "$@"
