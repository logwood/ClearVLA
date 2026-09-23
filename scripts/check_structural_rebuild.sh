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
import importlib.util
import json, platform
import torch
print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                  "cuda_available": torch.cuda.is_available(),
                  "optional_video_modules": {name: importlib.util.find_spec(name) is not None
                    for name in ("imageio", "imageio_ffmpeg", "cv2")}}, indent=2))
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
TESTS=(tests/test_mainline*.py tests/test_structural_static_audit.py
       tests/test_structural_audit_orchestration.py tests/test_calvin*.py
       tests/test_benchmark*.py tests/test_deployment_candidate_prefix_reuse.py
       tests/test_maniskill_binary_gripper.py tests/test_maniskill_rl_admission.py
       tests/test_libero_boundaries.py tests/test_camera_coordinate_role_candidate.py
       tests/test_stackcube_v2.py tests/test_residual_sac.py tests/test_simulation_video.py)
# Run source-appropriate selected inventories in isolated processes per file.
# Historical failures remain failures, but MUST NOT prevent current-source
# checks. Preserve pipefail (including log-write errors), then fail the overall
# gate after both inventories have completed. Source/static preparation errors
# above are still fatal: an unprepared inventory cannot be labeled as checked.
OLD=()
for path in "${TESTS[@]}"; do [[ ! -f "$BASELINE/$path" ]] || OLD+=("$path"); done
baseline_status=0
python scripts/run_structural_tests.py --root "$BASELINE" --output "$OUT" \
  --label baseline "${OLD[@]}" | tee "$OUT/baseline-tests.txt" || baseline_status=$?
current_status=0
python scripts/run_structural_tests.py --root "$ROOT" --output "$OUT" \
  --label current "${TESTS[@]}" | tee "$OUT/current-tests.txt" || current_status=$?
python - "$OUT/regression-exits.json" "$baseline_status" "$current_status" <<'PY_STATUS'
import json
import pathlib
import sys
baseline, current = map(int, sys.argv[2:])
report = {"baseline_exit": baseline, "current_exit": current,
          "both_inventories_attempted": True, "passed": baseline == current == 0}
pathlib.Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report))
PY_STATUS
if [[ "$baseline_status" -ne 0 || "$current_status" -ne 0 ]]; then
  exit 1
fi
