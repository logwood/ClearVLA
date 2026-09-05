#!/usr/bin/env bash
# One fresh hybrid Pen run. Logs are visible beside the artifact directory.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
: "${RUN_ROOT:?RUN_ROOT must name a new, unique visible run directory}"
: "${CUDA_VISIBLE_DEVICES:?select exactly one idle GPU by UUID}"
: "${HYBRID_PYTHON:?HYBRID_PYTHON must name the verified training interpreter}"
if [[ -e "${RUN_ROOT}" ]]; then
  printf 'Refusing existing run root: %s\n' "${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}"
exec > "${RUN_ROOT}/train.log" 2>&1
printf '[hybrid-v1] fresh initialization; no resume; source=%s; gpu=%s\n' "$(git rev-parse HEAD)" "${CUDA_VISIBLE_DEVICES}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
exec "${HYBRID_PYTHON}" -B -u -m clearvla.mainline.train \
  --config configs/mainline/object_intent_dynamics_323_pen_hybrid_v1.json \
  --device cuda --dtype bf16 --batch-size 8 --num-workers 4 \
  --output-dir "${RUN_ROOT}/artifacts"
