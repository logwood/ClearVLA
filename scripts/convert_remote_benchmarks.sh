#!/usr/bin/env bash
# Convert official CALVIN/LIBERO demonstrations into isolated ClearVLA HDF5 roots.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH_ROOT="${CLEARVLA_BENCH_ROOT:-/home/sen.wang/workspace/robotics/benchmarks}"
CALVIN_ROOT="${CLEARVLA_CALVIN_ROOT:-${BENCH_ROOT}/calvin}"
LIBERO_ROOT="${CLEARVLA_LIBERO_ROOT:-${BENCH_ROOT}/LIBERO}"
DATA_ROOT="${CLEARVLA_BENCH_DATA_ROOT:-/data/senwang/data}"
CALVIN_DATA="${CLEARVLA_CALVIN_DATA:-${DATA_ROOT}/calvin}"
LIBERO_DATA="${CLEARVLA_LIBERO_DATA:-${DATA_ROOT}/libero}"
CALVIN_ENV="${CLEARVLA_CALVIN_ENV:-/home/sen.wang/.venvs/clearvla-calvin}"
LIBERO_ENV="${CLEARVLA_LIBERO_ENV:-/home/sen.wang/.venvs/clearvla-libero}"
SIM_ENV="${CLEARVLA_SIM_ENV:-/data/senwang/envs/clearvla-sim}"
CONVERTER_PYTHON="${CLEARVLA_BENCH_CONVERTER_PYTHON:-${SIM_ENV}/bin/python}"
# Keep benchmark profiles explicit. A shared Pen base used to be copied for
# every target, which could silently construct CALVIN with a continuous
# gripper head. CLEARVLA_BASE_CONFIG remains a compatibility override.
SHARED_BASE_CONFIG="${CLEARVLA_BASE_CONFIG:-}"
CALVIN_BASE_CONFIG="${CLEARVLA_CALVIN_BASE_CONFIG:-${SHARED_BASE_CONFIG:-${REPO_ROOT}/configs/mainline/calvin_object_intent_dynamics_323_binary.json}}"
LIBERO_BASE_CONFIG="${CLEARVLA_LIBERO_BASE_CONFIG:-${SHARED_BASE_CONFIG:-${REPO_ROOT}/configs/mainline/object_intent_dynamics_323.json}}"
CALVIN_PROFILE="${CLEARVLA_CALVIN_PROFILE:-calvin_relative_7d_v1}"
CALVIN_ARM_FLOW_MODE="${CLEARVLA_CALVIN_ARM_FLOW_MODE:-relative_command_adapter}"
CALVIN_GRIPPER_MODE="${CLEARVLA_CALVIN_GRIPPER_OUTPUT_MODE:-calvin_binary_command}"
CALVIN_COMMAND_WEIGHT="${CLEARVLA_CALVIN_COMMAND_WEIGHT:-0.10}"
CALVIN_EVENT_THRESHOLD="${CLEARVLA_CALVIN_EVENT_THRESHOLD:-0.10}"
CALVIN_TASK_FILTER="${CLEARVLA_CALVIN_TASK_FILTER:-}"
CALVIN_VAL_FRACTION="${CLEARVLA_CALVIN_VAL_FRACTION:-0.10}"
CALVIN_SPLIT_SEED="${CLEARVLA_CALVIN_SPLIT_SEED:-0}"
LIBERO_PROFILE="${CLEARVLA_LIBERO_PROFILE:-libero_relative_7d_v1}"
LIBERO_ARM_FLOW_MODE="${CLEARVLA_LIBERO_ARM_FLOW_MODE:-relative_command_adapter}"
LIBERO_GRIPPER_MODE="${CLEARVLA_LIBERO_GRIPPER_OUTPUT_MODE:-continuous}"
LIBERO_EVENT_THRESHOLD="${CLEARVLA_LIBERO_EVENT_THRESHOLD:-0.10}"

DO_CALVIN=0
DO_LIBERO=0
CALVIN_SOURCE_OVERRIDE=""
CALVIN_OUTPUT="${CLEARVLA_CALVIN_CONVERTED:-${CALVIN_DATA}/converted/abc_d}"
CALVIN_CACHE_ROOT="${CLEARVLA_CALVIN_CACHE_ROOT:-${CALVIN_DATA}/caches/abc_d}"
CALVIN_LANGUAGE_BANK="${CLEARVLA_CALVIN_LANGUAGE_BANK:-${CALVIN_DATA}/language/t5_xxl_bank.pt}"
CALVIN_CONFIG="${CLEARVLA_CALVIN_CONFIG:-${CALVIN_DATA}/configs/mainline.json}"
CALVIN_RUN_ROOT="${CLEARVLA_CALVIN_RUN_ROOT:-${CALVIN_DATA}/runs/clearvla_calvin_abc_d}"
LIBERO_OUTPUT_OVERRIDE="${CLEARVLA_LIBERO_CONVERTED:-}"
read -r -a LIBERO_SUITES <<< "${CLEARVLA_LIBERO_SUITES:-libero_spatial}"
if [[ ${#LIBERO_SUITES[@]} -eq 0 ]]; then
  echo "CLEARVLA_LIBERO_SUITES must contain at least one suite" >&2
  exit 2
fi
LIMIT_PER_SPLIT="${CLEARVLA_CALVIN_LIMIT_PER_SPLIT:-}"
LIMIT_TASKS="${CLEARVLA_LIBERO_LIMIT_TASKS:-}"
LIMIT_DEMOS="${CLEARVLA_LIBERO_LIMIT_DEMOS:-}"

usage() {
  cat >&2 <<'EOF'
usage: convert_remote_benchmarks.sh TARGET [...]

TARGET choices:
  calvin-debug  auto-detect the downloaded debug source
  calvin-abc    convert task_ABC_D (default formal primary)
  libero-spatial | libero-object | libero-goal | libero-90 | libero-10
  libero        use CLEARVLA_LIBERO_SUITES (space-separated suite names)

The converter refuses non-empty output roots. Set explicit CLEARVLA_* paths to
resume a known conversion rather than mixing benchmark rows.
EOF
}

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi
for target in "$@"; do
  case "${target}" in
    calvin-debug) DO_CALVIN=1; CALVIN_SOURCE_OVERRIDE="debug" ;;
    calvin-abc) DO_CALVIN=1; CALVIN_SOURCE_OVERRIDE="abc" ;;
    libero-spatial) DO_LIBERO=1; LIBERO_SUITES=("libero_spatial") ;;
    libero-object) DO_LIBERO=1; LIBERO_SUITES=("libero_object") ;;
    libero-goal) DO_LIBERO=1; LIBERO_SUITES=("libero_goal") ;;
    libero-90) DO_LIBERO=1; LIBERO_SUITES=("libero_90") ;;
    libero-10) DO_LIBERO=1; LIBERO_SUITES=("libero_10") ;;
    libero) DO_LIBERO=1 ;;
    all) DO_CALVIN=1; DO_LIBERO=1 ;;
    *) usage; exit 2 ;;
  esac
done

resolve_calvin_source() {
  local requested="$1"
  local candidate
  if [[ -n "${CLEARVLA_CALVIN_SOURCE:-}" ]]; then
    candidate="${CLEARVLA_CALVIN_SOURCE}"
    [[ -d "${candidate}/training" && -d "${candidate}/validation" ]] || {
      echo "CLEARVLA_CALVIN_SOURCE lacks training/validation: ${candidate}" >&2
      exit 3
    }
    printf '%s\n' "${candidate}"
    return
  fi
  if [[ "${requested}" == abc && -d "${CALVIN_DATA}/raw/task_ABC_D/training" ]]; then
    printf '%s\n' "${CALVIN_DATA}/raw/task_ABC_D"
    return
  fi
  if [[ "${requested}" == debug ]]; then
    for candidate in \
      "${CALVIN_DATA}/raw/calvin_debug_dataset/task_D_D" \
      "${CALVIN_DATA}/raw/calvin_debug_dataset" \
      "${CALVIN_DATA}/raw/task_D_D"; do
      if [[ -d "${candidate}/training" && -d "${candidate}/validation" ]]; then
        printf '%s\n' "${candidate}"
        return
      fi
    done
  fi
  echo "cannot locate downloaded CALVIN source under ${CALVIN_DATA}/raw" >&2
  exit 3
}

run_calvin() {
  local source
  source="$(resolve_calvin_source "${CALVIN_SOURCE_OVERRIDE}")"
  [[ -x "${CALVIN_ENV}/bin/python" ]] || {
    echo "missing CALVIN environment ${CALVIN_ENV}; run setup_remote_benchmarks.sh calvin" >&2
    exit 2
  }
  [[ -x "${CONVERTER_PYTHON}" ]] || {
    echo "missing converter Python ${CONVERTER_PYTHON}; set CLEARVLA_BENCH_CONVERTER_PYTHON" >&2
    exit 2
  }
  mkdir -p "$(dirname "${CALVIN_OUTPUT}")"
  local limit_args=()
  if [[ -n "${LIMIT_PER_SPLIT}" ]]; then
    limit_args+=(--limit-per-split "${LIMIT_PER_SPLIT}")
  fi
  local task_args=()
  if [[ -n "${CALVIN_TASK_FILTER}" ]]; then
    task_args+=(--task-filter "${CALVIN_TASK_FILTER}")
  fi
  cd "${REPO_ROOT}"
  "${CONVERTER_PYTHON}" -m clearvla.benchmarks.calvin \
    --source "${source}" \
    --output "${CALVIN_OUTPUT}" \
    --validation-annotations \
      "${CALVIN_ROOT}/calvin_models/conf/annotations/new_playtable_validation.yaml" \
    --val-fraction "${CALVIN_VAL_FRACTION}" \
    --split-seed "${CALVIN_SPLIT_SEED}" \
    "${task_args[@]}" \
    "${limit_args[@]}"
  "${CONVERTER_PYTHON}" -m clearvla.benchmarks.config \
    --base "${CALVIN_BASE_CONFIG}" \
    --output "${CALVIN_CONFIG}" \
    --hdf5-root "${CALVIN_OUTPUT}" \
    --cache-root "${CALVIN_CACHE_ROOT}" \
    --language-bank "${CALVIN_LANGUAGE_BANK}" \
    --run-root "${CALVIN_RUN_ROOT}" \
    --num-workers "${CLEARVLA_MAINLINE_WORKERS:-4}" \
    --data-profile "${CALVIN_PROFILE}" \
    --arm-flow-mode "${CALVIN_ARM_FLOW_MODE}" \
    --gripper-output-mode "${CALVIN_GRIPPER_MODE}" \
    --gripper-command-weight "${CALVIN_COMMAND_WEIGHT}" \
    --gripper-event-threshold "${CALVIN_EVENT_THRESHOLD}"
  echo "[calvin-convert] output=${CALVIN_OUTPUT} config=${CALVIN_CONFIG} cache=${CALVIN_CACHE_ROOT} language=${CALVIN_LANGUAGE_BANK}"
}

run_libero() {
  [[ -x "${LIBERO_ENV}/bin/python" ]] || {
    echo "missing LIBERO environment ${LIBERO_ENV}; run setup_remote_benchmarks.sh libero" >&2
    exit 2
  }
  [[ -x "${CONVERTER_PYTHON}" ]] || {
    echo "missing converter Python ${CONVERTER_PYTHON}; set CLEARVLA_BENCH_CONVERTER_PYTHON" >&2
    exit 2
  }
  [[ -d "${LIBERO_DATA}/raw/datasets" ]] || {
    echo "missing LIBERO downloaded datasets: ${LIBERO_DATA}/raw/datasets" >&2
    exit 3
  }
  local limit_args=()
  [[ -n "${LIMIT_TASKS}" ]] && limit_args+=(--limit-tasks "${LIMIT_TASKS}")
  [[ -n "${LIMIT_DEMOS}" ]] && limit_args+=(--limit-demos-per-task "${LIMIT_DEMOS}")
  local suite_slug output cache_root language_bank config run_root
  suite_slug="$(IFS=_; echo "${LIBERO_SUITES[*]}")"
  output="${LIBERO_OUTPUT_OVERRIDE:-${LIBERO_DATA}/converted/${suite_slug}}"
  cache_root="${CLEARVLA_LIBERO_CACHE_ROOT:-${LIBERO_DATA}/caches/${suite_slug}}"
  language_bank="${CLEARVLA_LIBERO_LANGUAGE_BANK:-${LIBERO_DATA}/language/${suite_slug}_t5_xxl_bank.pt}"
  config="${CLEARVLA_LIBERO_CONFIG:-${LIBERO_DATA}/configs/${suite_slug}.json}"
  run_root="${CLEARVLA_LIBERO_RUN_ROOT:-${LIBERO_DATA}/runs/clearvla_${suite_slug}}"
  mkdir -p "$(dirname "${output}")"
  cd "${REPO_ROOT}"
  LIBERO_CONFIG_PATH="${LIBERO_DATA}/config" \
    "${CONVERTER_PYTHON}" -m clearvla.benchmarks.libero \
    --source "${LIBERO_DATA}/raw/datasets" \
    --output "${output}" \
    --suites "${LIBERO_SUITES[@]}" \
    --evaluator-bddl-root "${LIBERO_ROOT}/libero/libero/bddl_files" \
    "${limit_args[@]}"
  LIBERO_CONFIG_PATH="${LIBERO_DATA}/config" \
    "${CONVERTER_PYTHON}" -m clearvla.benchmarks.config \
    --base "${LIBERO_BASE_CONFIG}" \
    --output "${config}" \
    --hdf5-root "${output}" \
    --cache-root "${cache_root}" \
    --language-bank "${language_bank}" \
    --run-root "${run_root}" \
    --num-workers "${CLEARVLA_MAINLINE_WORKERS:-4}" \
    --data-profile "${LIBERO_PROFILE}" \
    --arm-flow-mode "${LIBERO_ARM_FLOW_MODE}" \
    --gripper-output-mode "${LIBERO_GRIPPER_MODE}" \
    --gripper-event-threshold "${LIBERO_EVENT_THRESHOLD}"
  echo "[libero-convert] suites=${suite_slug} output=${output} config=${config} cache=${cache_root} language=${language_bank}"
}

if [[ "${DO_CALVIN}" == 1 ]]; then run_calvin; fi
if [[ "${DO_LIBERO}" == 1 ]]; then run_libero; fi

printf '[clearvla-benchmark-convert] calvin=%s libero=%s\n' \
  "${DO_CALVIN}" "${DO_LIBERO}"
