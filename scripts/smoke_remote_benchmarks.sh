#!/usr/bin/env bash
# Run small official-environment rollouts through the loopback bridge.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH_ROOT="${CLEARVLA_BENCH_ROOT:-/home/sen.wang/workspace/robotics/benchmarks}"
CALVIN_ROOT="${CLEARVLA_CALVIN_ROOT:-${BENCH_ROOT}/calvin}"
LIBERO_ROOT="${CLEARVLA_LIBERO_ROOT:-${BENCH_ROOT}/LIBERO}"
DATA_ROOT="${CLEARVLA_BENCH_DATA_ROOT:-/data/senwang/data}"
CALVIN_DATA="${CLEARVLA_CALVIN_DATA:-${DATA_ROOT}/calvin}"
LIBERO_DATA="${CLEARVLA_LIBERO_DATA:-${DATA_ROOT}/libero}"
SIM_ENV="${CLEARVLA_SIM_ENV:-/data/senwang/envs/clearvla-sim}"
CALVIN_ENV="${CLEARVLA_CALVIN_ENV:-/home/sen.wang/.venvs/clearvla-calvin}"
LIBERO_ENV="${CLEARVLA_LIBERO_ENV:-/home/sen.wang/.venvs/clearvla-libero}"
BRIDGE_PORT="${CLEARVLA_BRIDGE_PORT:-8765}"
RUN_ID="${CLEARVLA_BENCH_SMOKE_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${CLEARVLA_BENCH_SMOKE_ROOT:-${DATA_ROOT}/benchmark_smoke/${RUN_ID}}"

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
  TARGETS=(calvin libero)
fi
mkdir -p "${OUT_ROOT}" "${LIBERO_DATA}/cache/huggingface"

if [[ ! -x "${SIM_ENV}/bin/python" ]]; then
  echo "missing ClearVLA simulator Python for the bridge: ${SIM_ENV}/bin/python" >&2
  exit 2
fi
if [[ ! -x "${CALVIN_ENV}/bin/python" && " ${TARGETS[*]} " == *" calvin "* ]]; then
  echo "missing CALVIN environment; run setup_remote_benchmarks.sh calvin" >&2
  exit 2
fi
if [[ ! -x "${LIBERO_ENV}/bin/python" && " ${TARGETS[*]} " == *" libero "* ]]; then
  echo "missing LIBERO environment; run setup_remote_benchmarks.sh libero" >&2
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${DATA_ROOT}/benchmark_smoke/huggingface"
export XDG_CACHE_HOME="${DATA_ROOT}/benchmark_smoke/xdg-cache"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
# CALVIN's legacy OpenCV wheel is Qt-linked even when ``show_gui`` is false.
# Keep the smoke lane headless by default; a user must explicitly opt into
# X11 with QT_QPA_PLATFORM/xvfb when they want visual debugging.
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}"

BRIDGE_LOG="${OUT_ROOT}/bridge.log"
"${SIM_ENV}/bin/python" -m clearvla.benchmarks.bridge \
  --smoke-zero-policy \
  --host 127.0.0.1 \
  --port "${BRIDGE_PORT}" \
  > "${BRIDGE_LOG}" 2>&1 &
BRIDGE_PID=$!
cleanup() {
  kill "${BRIDGE_PID}" 2>/dev/null || true
  wait "${BRIDGE_PID}" 2>/dev/null || true
}
trap cleanup EXIT

for attempt in $(seq 1 30); do
  if curl -fsS --max-time 2 "http://127.0.0.1:${BRIDGE_PORT}/health" > "${OUT_ROOT}/bridge_health.json"; then
    break
  fi
  if ! kill -0 "${BRIDGE_PID}" 2>/dev/null; then
    cat "${BRIDGE_LOG}" >&2 || true
    exit 7
  fi
  sleep 1
  if [[ "${attempt}" == 30 ]]; then
    echo "bridge did not become healthy" >&2
    exit 7
  fi
done

find_calvin_source() {
  local candidate
  for candidate in \
    "${CALVIN_DATA}/raw/calvin_debug_dataset/task_D_D" \
    "${CALVIN_DATA}/raw/calvin_debug_dataset" \
    "${CALVIN_DATA}/raw/task_D_D" \
    "${CALVIN_DATA}/raw/task_ABC_D"; do
    if [[ -d "${candidate}/validation" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  echo "no downloaded CALVIN validation source found" >&2
  exit 3
}

for target in "${TARGETS[@]}"; do
  case "${target}" in
    calvin)
      source="$(find_calvin_source)"
      CALVIN_OUT="${OUT_ROOT}/calvin"
      mkdir -p "${CALVIN_OUT}"
      calvin_debug_args=()
      if [[ "${CLEARVLA_CALVIN_SMOKE_DEBUG:-0}" == 1 ]]; then
        calvin_debug_args+=(--debug)
      fi
      (cd "${REPO_ROOT}" && "${CALVIN_ENV}/bin/python" -m clearvla.benchmarks.calvin_eval \
        --dataset-root "${source}" \
        --output-dir "${CALVIN_OUT}" \
        --endpoint "http://127.0.0.1:${BRIDGE_PORT}" \
        --allow-smoke-policy \
        --num-sequences "${CLEARVLA_CALVIN_SMOKE_SEQUENCES:-1}" \
        --max-subtask-steps "${CLEARVLA_CALVIN_SMOKE_STEPS:-2}" \
        --sequence-workers 1 \
        "${calvin_debug_args[@]}") | tee "${CALVIN_OUT}/stdout.json"
      ;;
    libero)
      LIBERO_OUT="${OUT_ROOT}/libero.json"
      (cd "${REPO_ROOT}" && LIBERO_CONFIG_PATH="${LIBERO_DATA}/config" \
        PYTHONPATH="${LIBERO_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}" \
        "${LIBERO_ENV}/bin/python" -m clearvla.benchmarks.libero_eval \
        --suite "${CLEARVLA_LIBERO_SMOKE_SUITE:-libero_spatial}" \
        --output "${LIBERO_OUT}" \
        --endpoint "http://127.0.0.1:${BRIDGE_PORT}" \
        --allow-smoke-policy \
        --task-ids "${CLEARVLA_LIBERO_SMOKE_TASK:-0}" \
        --episodes-per-task "${CLEARVLA_LIBERO_SMOKE_EPISODES:-1}" \
        --max-steps "${CLEARVLA_LIBERO_SMOKE_STEPS:-3}" \
        --warmup-steps "${CLEARVLA_LIBERO_SMOKE_WARMUP:-1}" \
        --image-side 128) | tee "${OUT_ROOT}/libero.stdout.json"
      ;;
    *) echo "unknown smoke target ${target}; use calvin or libero" >&2; exit 2 ;;
  esac
done

printf '[clearvla-benchmark-smoke] run_id=%s output=%s bridge=%s\n' \
  "${RUN_ID}" "${OUT_ROOT}" "${BRIDGE_PORT}"
