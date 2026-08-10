#!/usr/bin/env bash
# Sequential, same-data early-loss comparison for an 8 GiB development GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

COMPARE_BATCHES="${COMPARE_BATCHES:-20}"
COMPARE_BATCH_SIZE="${COMPARE_BATCH_SIZE:-1}"
COMPARE_VAL_BATCHES="${COMPARE_VAL_BATCHES:-1}"
COMPARE_ROOT="${COMPARE_ROOT:-runs/v122_mainline_early_loss_compare}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_ROOT="${DATA_ROOT:-/data/liang.zhang/dataset/grab_pen_single/grab_pen_single}"
CACHE_DIR="${CACHE_DIR:-/data/senwang/data/cache_336}"
DINO_CACHE_DIR="${DINO_CACHE_DIR:-/data/senwang/data/dinov2_cache_336}"
T5_CONDITION_PATH="${T5_CONDITION_PATH:-/data/senwang/checkpoint/grasp_pen_embed.pt}"

for pair in \
  "COMPARE_BATCHES:${COMPARE_BATCHES}" \
  "COMPARE_BATCH_SIZE:${COMPARE_BATCH_SIZE}" \
  "COMPARE_VAL_BATCHES:${COMPARE_VAL_BATCHES}"; do
  name="${pair%%:*}"
  value="${pair#*:}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[early-loss-compare] ${name} must be a positive integer, got ${value}" >&2
    exit 2
  fi
done

for directory in "${DATA_ROOT}" "${CACHE_DIR}" "${DINO_CACHE_DIR}"; do
  if [[ ! -d "${directory}" ]]; then
    echo "[early-loss-compare] required directory is missing: ${directory}" >&2
    exit 2
  fi
done
if [[ ! -f "${T5_CONDITION_PATH}" ]]; then
  echo "[early-loss-compare] T5 condition is missing: ${T5_CONDITION_PATH}" >&2
  exit 2
fi

LEGACY_RUN="${COMPARE_ROOT}/v122_run"
MAINLINE_RUN="${COMPARE_ROOT}/mainline_run"
LEGACY_LOG="${COMPARE_ROOT}/v122.stdout.log"
MAINLINE_LOG="${COMPARE_ROOT}/mainline.stdout.log"
for path in "${LEGACY_RUN}" "${MAINLINE_RUN}" "${LEGACY_LOG}" "${MAINLINE_LOG}"; do
  if [[ -e "${path}" ]]; then
    echo "[early-loss-compare] fresh comparison requires an unused path: ${path}" >&2
    exit 2
  fi
done
mkdir -p "${COMPARE_ROOT}"

resolved_python="$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
export PATH="$(dirname "${resolved_python}"):${PATH}"
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

printf '[early-loss-compare] batches=%s batch_size=%s cuda=%s root=%s\n' \
  "${COMPARE_BATCHES}" "${COMPARE_BATCH_SIZE}" \
  "${CUDA_VISIBLE_DEVICES:-<runtime-default>}" "${COMPARE_ROOT}"
printf '[early-loss-compare] criterion=shared_scale_and_early_direction_not_bitwise_equality\n'

OUT_DIR="${LEGACY_RUN}" \
OBJECT_323_BATCH_SIZE="${COMPARE_BATCH_SIZE}" \
DATA_ROOT="${DATA_ROOT}" \
CACHE_DIR="${CACHE_DIR}" \
DINO_CACHE_DIR="${DINO_CACHE_DIR}" \
T5_CONDITION_PATH="${T5_CONDITION_PATH}" \
bash scripts/current_object_intent_dynamics_323_smoke.sh \
  --max-train-batches "${COMPARE_BATCHES}" \
  --max-val-batches "${COMPARE_VAL_BATCHES}" \
  --log-every 1 \
  --eval-sampling-diagnostic-batches 0 \
  --eval-proposal-ablation-batches 0 \
  --eval-execution-ablation-batches 0 \
  --eval-representation-batches 0 \
  2>&1 | tee "${LEGACY_LOG}"

# The first process has exited here, so the two full models never coexist on
# the 8 GiB card.
OUT_DIR="${MAINLINE_RUN}" \
MAINLINE_BATCH_SIZE="${COMPARE_BATCH_SIZE}" \
MAINLINE_NUM_WORKERS=0 \
DATA_ROOT="${DATA_ROOT}" \
CACHE_DIR="${CACHE_DIR}" \
DINO_CACHE_DIR="${DINO_CACHE_DIR}" \
T5_CONDITION_PATH="${T5_CONDITION_PATH}" \
bash scripts/smoke_mainline.sh \
  --max-train-batches "${COMPARE_BATCHES}" \
  --max-val-batches "${COMPARE_VAL_BATCHES}" \
  2>&1 | tee "${MAINLINE_LOG}"

"${PYTHON_BIN}" -m clearvla.tools.compare_mainline_early_losses \
  --legacy-log "${LEGACY_LOG}" \
  --mainline-metrics "${MAINLINE_RUN}/metrics.jsonl" \
  --legacy-manifest "${LEGACY_RUN}/run_manifest.json" \
  --mainline-context "${MAINLINE_RUN}/run_context.json" \
  --steps "${COMPARE_BATCHES}" \
  --json-output "${COMPARE_ROOT}/comparison.json" \
  --markdown-output "${COMPARE_ROOT}/comparison.md"

printf '[early-loss-compare] complete report=%s\n' "${COMPARE_ROOT}/comparison.md"
