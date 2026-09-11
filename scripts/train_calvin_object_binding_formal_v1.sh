#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${CLEARVLA_PYTHON:-python}"
CONFIG_PATH="${CALVIN_CONFIG:-configs/mainline/calvin_object_binding_formal_v1.json}"
OUTPUT_PATH="${CALVIN_OUT_DIR:-/data/senwang/data/calvin/runs/clearvla_calvin_object_binding_formal_v1_20260911}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON_BIN}" -c \
  'from clearvla.mainline.config import load_config; import sys; c=load_config(sys.argv[1]); assert c.data.data_profile == "calvin_relative_7d_v1"; assert c.top.calvin_object_binding == "calvin_primary_v1"' \
  "${CONFIG_PATH}"

printf '[calvin-object-binding-v1] config=%s out=%s cuda=%s\\n' \
  "${CONFIG_PATH}" "${OUTPUT_PATH}" "${CUDA_VISIBLE_DEVICES}"

exec "${PYTHON_BIN}" -B -u -m clearvla.mainline.train \
  --config "${CONFIG_PATH}" \
  --device cuda \
  --dtype bf16 \
  --batch-size 8 \
  --num-workers 4 \
  --output-dir "${OUTPUT_PATH}"
