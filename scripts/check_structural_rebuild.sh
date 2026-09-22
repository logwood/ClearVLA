#!/usr/bin/env bash
# Read-only checks. Never trains on user data, updates Git refs, or migrates checkpoints.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${1:?provide an output directory outside the repository}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
case "$OUT/" in "$ROOT/"*) echo "audit output must be outside the repository" >&2; exit 2;; esac
BASE=0f07160d692ec8c8302880420a3d93d74512c39b
BASELINE="${CLEARVLA_AUDIT_BASELINE:-$OUT/baseline}"
if [[ ! -d "$BASELINE/clearvla" ]]; then
  mkdir -p "$BASELINE"
  git archive "$BASE" | tar -x -C "$BASELINE"
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python - <<'PY' > "$OUT/runtime.json"
import json, platform
import torch
print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                  "cuda_available": torch.cuda.is_available()}, indent=2))
PY
ruff --version > "$OUT/ruff-version.txt"
pyright --version > "$OUT/pyright-version.txt"
python -m compileall -q clearvla tests scripts/audit_structural_static.py
if [[ -n "${CLEARVLA_AUDIT_CHANGED_FILES:-}" ]]; then
  cp "$CLEARVLA_AUDIT_CHANGED_FILES" "$OUT/changed-python-files.txt"
else
  git diff --name-only --diff-filter=ACMR "$BASE" -- '*.py' > "$OUT/changed-python-files.txt"
fi
python scripts/audit_structural_static.py --baseline "$BASELINE" --current "$ROOT" \
  --output "$OUT" --files "$OUT/changed-python-files.txt"
TESTS=(tests/test_mainline*.py tests/test_structural_static_audit.py tests/test_calvin_raw_reader.py
       tests/test_calvin_schema30_adapter.py tests/test_calvin_chunked_execution.py
       tests/test_maniskill_binary_gripper.py tests/test_maniskill_rl_admission.py
       tests/test_libero_boundaries.py tests/test_camera_coordinate_role_candidate.py)
# Run identical selected inventories in separate processes per file. This
# bounds allocator lifetime, not test coverage. A failed/killed child is fatal.
OLD=()
for path in "${TESTS[@]}"; do [[ ! -f "$BASELINE/$path" ]] || OLD+=("$path"); done
python scripts/run_structural_tests.py --root "$BASELINE" --output "$OUT" \
  --label baseline "${OLD[@]}" | tee "$OUT/baseline-tests.txt"
python scripts/run_structural_tests.py --root "$ROOT" --output "$OUT" \
  --label current "${TESTS[@]}" | tee "$OUT/current-tests.txt"
