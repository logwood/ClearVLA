#!/usr/bin/env bash
# No dataset, pretrained weights or simulator. These are explicit interface tests.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?Pass a directory for logs and JUnit XML}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TERM="${TERM:-dumb}"
"$PYTHON_BIN" - <<'PY' > "$OUT/environment.txt"
import sys, platform, torch
print(sys.version)
print(platform.platform())
print('torch', torch.__version__, 'CUDA available', torch.cuda.is_available())
print('Scope: synthetic compact transports and explicit marker encoder, not pretrained/robot success')
PY
git rev-parse HEAD > "$OUT/tested-commit.txt"
STATUS=0
run_group() {
  local name="$1"; shift
  "$PYTHON_BIN" -m pytest -q "$@" --junitxml="$OUT/$name.xml" > "$OUT/$name.log" 2>&1
  local code=$?
  printf '%s\t%s\n' "$name" "$code" >> "$OUT/exit-codes.tsv"
  cat "$OUT/$name.log"
  if [[ "$code" -ne 0 ]]; then STATUS=1; fi
}
: > "$OUT/exit-codes.tsv"
run_group native-owners tests/test_mainline_spatial_effect_v2.py
run_group integrated-policy tests/test_mainline_spatial_effect_integration.py
run_group legacy-task tests/test_mainline_task_execution.py \
  tests/test_mainline_task_execution_boundaries.py \
  tests/test_mainline_task_execution_language_values.py \
  tests/test_mainline_task_execution_integration.py
run_group native-marker tests/test_dinov3_spatial_effect_integration.py
"$PYTHON_BIN" -m compileall -q clearvla/mainline tests/test_mainline_spatial_effect_v2.py \
  tests/test_mainline_spatial_effect_integration.py tests/test_dinov3_spatial_effect_integration.py
CODE=$?
printf 'compile\t%s\n' "$CODE" >> "$OUT/exit-codes.tsv"
if [[ "$CODE" -ne 0 ]]; then STATUS=1; fi
exit "$STATUS"
