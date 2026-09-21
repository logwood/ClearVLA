#!/usr/bin/env bash
set -euo pipefail

# Matched closed-loop intervention for diagnosing a fixed CALVIN arm drift.
# The same frozen checkpoint is loaded once.  The intervention is changed only
# at the episode-reset boundary; no checkpoint or training state is written.

MODEL_REPO_ROOT="${MODEL_REPO_ROOT:-/home/sen.wang/workspace/robotics/clear/calvin-object-binding-push-20260913}"
EVAL_REPO_ROOT="${EVAL_REPO_ROOT:-/home/sen.wang/workspace/robotics/clear/clearvla_sim_mainline}"
CHECKPOINT="${CHECKPOINT:-/data/senwang/data/calvin/runs/clearvla_calvin_object_binding_push_all_20260913_3ep/checkpoints/best.pt}"
DATASET_ROOT="${DATASET_ROOT:-/data/senwang/data/calvin/raw/task_ABC_D}"
MODEL_PYTHON="${MODEL_PYTHON:-/data/senwang/envs/clearvla-sim/bin/python}"
CALVIN_PYTHON="${CALVIN_PYTHON:-/home/sen.wang/.venvs/clearvla-calvin/bin/python}"
BRIDGE_GPU="${BRIDGE_GPU:-1}"
EVAL_GPU="${EVAL_GPU:-6}"
PORT="${PORT:-18774}"
MAX_STEPS="${MAX_STEPS:-80}"
EXECUTE_ROWS="${EXECUTE_ROWS:-4}"
RUN_ROOT="${RUN_ROOT:-/data/senwang/data/calvin/rollouts/terminal_bias_ab_$(date +%Y%m%d_%H%M%S)}"
PROBE_SCRIPT="${PROBE_SCRIPT:-${RUN_ROOT}/serve_calvin_terminal_bias_probe.py}"
MODE_FILE="${RUN_ROOT}/mode.txt"
INITIAL_STATE="${RUN_ROOT}/initial_state.json"
BRIDGE_LOG="${RUN_ROOT}/bridge.log"

for required in \
  "${MODEL_REPO_ROOT}/clearvla/benchmarks/bridge.py" \
  "${MODEL_REPO_ROOT}/clearvla/simulation/clearvla_policy.py" \
  "${EVAL_REPO_ROOT}/clearvla/benchmarks/calvin_eval.py" \
  "${CHECKPOINT}" \
  "${DATASET_ROOT}" \
  "${MODEL_PYTHON}" \
  "${CALVIN_PYTHON}" \
  "${PROBE_SCRIPT}"; do
  if [[ ! -e "${required}" ]]; then
    printf 'missing required path: %s\n' "${required}" >&2
    exit 2
  fi
done

if [[ -e "${RUN_ROOT}/started.json" ]]; then
  printf 'refusing to reuse an existing run root: %s\n' "${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}"

cat >"${INITIAL_STATE}" <<'JSON'
{
  "blue_block": "table",
  "drawer": "closed",
  "grasped": 0,
  "led": 0,
  "lightbulb": 0,
  "pink_block": "slider_right",
  "red_block": "table",
  "slider": "right"
}
JSON
printf 'baseline\n' >"${MODE_FILE}"

"${MODEL_PYTHON}" -c \
  "import json, pathlib, time; pathlib.Path('${RUN_ROOT}/started.json').write_text(json.dumps({'started_unix': time.time(), 'run_root': '${RUN_ROOT}', 'checkpoint': '${CHECKPOINT}', 'max_steps': int('${MAX_STEPS}'), 'execute_rows': int('${EXECUTE_ROWS}')}, indent=2))"

printf '[terminal-bias-ab] run_root=%s bridge_gpu=%s eval_gpu=%s max_steps=%s execute_rows=%s\n' \
  "${RUN_ROOT}" "${BRIDGE_GPU}" "${EVAL_GPU}" "${MAX_STEPS}" "${EXECUTE_ROWS}"

CUDA_VISIBLE_DEVICES="${BRIDGE_GPU}" \
PYTHONPATH="${MODEL_REPO_ROOT}" \
PYTHONUNBUFFERED=1 \
"${MODEL_PYTHON}" "${PROBE_SCRIPT}" \
  --checkpoint "${CHECKPOINT}" \
  --mode-file "${MODE_FILE}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --device cuda:0 \
  --seed 0 \
  --dinov2-local-files-only \
  >"${BRIDGE_LOG}" 2>&1 &
BRIDGE_PID=$!

cleanup() {
  if kill -0 "${BRIDGE_PID}" 2>/dev/null; then
    kill "${BRIDGE_PID}" 2>/dev/null || true
    wait "${BRIDGE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

WAIT_SECONDS=0
until curl --silent --show-error --fail --max-time 5 \
  "http://127.0.0.1:${PORT}/health" >"${RUN_ROOT}/health_ready.json" 2>/dev/null; do
  if ! kill -0 "${BRIDGE_PID}" 2>/dev/null; then
    printf 'bridge exited before becoming healthy\n' >&2
    tail -n 100 "${BRIDGE_LOG}" >&2 || true
    exit 3
  fi
  sleep 5
  WAIT_SECONDS=$((WAIT_SECONDS + 5))
  if (( WAIT_SECONDS % 60 == 0 )); then
    printf '[terminal-bias-ab] model still loading, elapsed=%ss\n' "${WAIT_SECONDS}"
  fi
done
printf '[terminal-bias-ab] bridge healthy after %ss\n' "${WAIT_SECONDS}"

run_rollout() {
  local mode="$1"
  local output_dir="${RUN_ROOT}/${mode}"
  printf '%s\n' "${mode}" >"${MODE_FILE}"
  printf '[terminal-bias-ab] starting mode=%s\n' "${mode}"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
  PYTHONPATH="${EVAL_REPO_ROOT}" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  QT_QPA_PLATFORM=offscreen \
  "${CALVIN_PYTHON}" -m clearvla.benchmarks.calvin_eval \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${output_dir}" \
    --endpoint "http://127.0.0.1:${PORT}" \
    --timeout 300 \
    --task push_red_block_left \
    --instruction "go push the red block left" \
    --initial-state-json "${INITIAL_STATE}" \
    --max-subtask-steps "${MAX_STEPS}" \
    --execute-rows "${EXECUTE_ROWS}" \
    --record-video \
    --video-fps 12 \
    >"${RUN_ROOT}/${mode}.log" 2>&1
  printf '[terminal-bias-ab] finished mode=%s\n' "${mode}"
}

run_rollout baseline
run_rollout zero_terminal_bias

"${CALVIN_PYTHON}" - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
rows = {}
for mode in ("baseline", "zero_terminal_bias"):
    result = json.loads((root / mode / "result.json").read_text())
    arrays = np.load(root / mode / "actions.npz")
    action = np.asarray(arrays["executed"], dtype=np.float64)
    robot = np.asarray(arrays["robot_obs"], dtype=np.float64)
    first = action[: min(20, action.shape[0]), :6]
    rows[mode] = {
        "success": bool(result["success"]),
        "steps": int(result["steps"]),
        "planning_decisions": int(result["execution"]["planning_decisions"]),
        "elapsed_seconds": float(result["elapsed_seconds"]),
        "arm_signed_mean": action[:, :6].mean(axis=0).tolist(),
        "arm_first20_signed_mean": first.mean(axis=0).tolist(),
        "arm_mean_l2": float(np.linalg.norm(action[:, :6], axis=1).mean()),
        "raw_robot_obs_first6_delta": (
            (robot[-1, :6] - robot[0, :6]).tolist()
            if robot.shape[0] > 1 and robot.shape[1] >= 6
            else None
        ),
        "video": result["video_recording"].get("video"),
        "intervention_before": result["bridge_health_before"].get(
            "diagnostic_intervention"
        ),
        "intervention_after": result["bridge_health_after"].get(
            "diagnostic_intervention"
        ),
    }

base = np.asarray(rows["baseline"]["arm_signed_mean"])
zero = np.asarray(rows["zero_terminal_bias"]["arm_signed_mean"])
summary = {
    "schema": "clearvla-calvin-terminal-bias-ab-v1",
    "run_root": str(root),
    "matched_controls": {
        "checkpoint": "/data/senwang/data/calvin/runs/clearvla_calvin_object_binding_push_all_20260913_3ep/checkpoints/best.pt",
        "policy_seed": 0,
        "task": "push_red_block_left",
        "instruction": "go push the red block left",
        "initial_state": json.loads((root / "initial_state.json").read_text()),
        "execute_rows": 4,
    },
    "modes": rows,
    "zero_minus_baseline_arm_signed_mean": (zero - base).tolist(),
}
(root / "comparison.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY

"${MODEL_PYTHON}" -c \
  "import json, pathlib, time; pathlib.Path('${RUN_ROOT}/complete.json').write_text(json.dumps({'completed_unix': time.time(), 'run_root': '${RUN_ROOT}'}, indent=2))"
printf '[terminal-bias-ab] complete run_root=%s\n' "${RUN_ROOT}"
