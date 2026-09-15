#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

uv run --frozen --no-sync python -m compileall -q clearvla tests
uv run --frozen --no-sync python -m pytest -q \
  tests/test_audit_policy_logs.py \
  tests/test_probe_flow_dino_dataset_motion.py \
  tests/test_physical_action_codec.py \
  tests/test_temporal_dct.py
