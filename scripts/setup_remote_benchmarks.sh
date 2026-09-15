#!/usr/bin/env bash
# Install CALVIN and LIBERO in independent legacy environments.
# Nothing from these environments is installed into the ClearVLA/ManiSkill env.
set -euo pipefail

BENCH_ROOT="${CLEARVLA_BENCH_ROOT:-/home/sen.wang/workspace/robotics/benchmarks}"
CALVIN_ROOT="${CLEARVLA_CALVIN_ROOT:-${BENCH_ROOT}/calvin}"
LIBERO_ROOT="${CLEARVLA_LIBERO_ROOT:-${BENCH_ROOT}/LIBERO}"
CALVIN_ENV="${CLEARVLA_CALVIN_ENV:-/home/sen.wang/.venvs/clearvla-calvin}"
LIBERO_ENV="${CLEARVLA_LIBERO_ENV:-/home/sen.wang/.venvs/clearvla-libero}"
DATA_ROOT="${CLEARVLA_BENCH_DATA_ROOT:-/data/senwang/data}"
UV_BIN="${CLEARVLA_UV_BIN:-/data/senwang/tools/uv/uv}"
UV_CACHE="${CLEARVLA_UV_CACHE:-${DATA_ROOT}/uv-cache}"
UV_PYTHONS="${CLEARVLA_UV_PYTHONS:-${DATA_ROOT}/uv-python}"
PACKAGE_CACHE="${CLEARVLA_PACKAGE_CACHE:-${DATA_ROOT}/package-cache}"
PYTHON_VERSION="${CLEARVLA_BENCH_PYTHON:-3.9}"
LIBERO_CONFIG_PATH="${CLEARVLA_LIBERO_CONFIG_PATH:-${DATA_ROOT}/libero/config}"

DO_CALVIN=1
DO_LIBERO=1
if [[ $# -gt 0 ]]; then
  DO_CALVIN=0
  DO_LIBERO=0
  for target in "$@"; do
    case "$target" in
      calvin) DO_CALVIN=1 ;;
      libero) DO_LIBERO=1 ;;
      all) DO_CALVIN=1; DO_LIBERO=1 ;;
      *) echo "usage: $0 [calvin] [libero] [all]" >&2; exit 2 ;;
    esac
  done
fi

if [[ ! -x "${UV_BIN}" ]]; then
  echo "uv is missing at ${UV_BIN}; install it first or set CLEARVLA_UV_BIN" >&2
  exit 2
fi
for required in "${CALVIN_ROOT}" "${LIBERO_ROOT}"; do
  if [[ ! -d "${required}" ]]; then
    echo "benchmark checkout is missing: ${required}" >&2
    exit 2
  fi
done

mkdir -p "${UV_CACHE}" "${UV_PYTHONS}" "${PACKAGE_CACHE}" \
  "${CALVIN_ENV%/*}" "${LIBERO_ENV%/*}" "${LIBERO_CONFIG_PATH}"
export UV_CACHE_DIR="${UV_CACHE}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHONS}"
export PIP_CACHE_DIR="${PACKAGE_CACHE}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1

"${UV_BIN}" python install "${PYTHON_VERSION}"

make_env() {
  local env_path="$1"
  if [[ ! -x "${env_path}/bin/python" ]]; then
    "${UV_BIN}" venv --python "${PYTHON_VERSION}" "${env_path}"
  fi
}

pip_install() {
  local python_bin="$1"
  shift
  "${UV_BIN}" pip install --python "${python_bin}" "$@"
}

filtered_requirements() {
  local source="$1"
  local output="$2"
  # A few optional plotting packages in the legacy files have no wheel on the
  # current host (notably MulticoreTSNE). They are not imported by the
  # official rollout path, so leave them out while retaining every runtime
  # dependency. Pins for the numeric/torch stack are installed explicitly.
  grep -Eiv \
    '^[[:space:]]*(multicoretsne|torch|torchvision|torchaudio|numpy|setuptools|pyhash|sentence-transformers|wandb|moviepy|plotly|lightning_lite|mujoco|robosuite)([<=>~!]|[[:space:]]|$)' \
    "${source}" > "${output}"
}

install_calvin() {
  make_env "${CALVIN_ENV}"
  local python_bin="${CALVIN_ENV}/bin/python"
  local env_requirements model_requirements constraints
  env_requirements="$(mktemp /tmp/clearvla-calvin-env-req.XXXXXX)"
  model_requirements="$(mktemp /tmp/clearvla-calvin-model-req.XXXXXX)"
  constraints="$(mktemp /tmp/clearvla-calvin-constraints.XXXXXX)"
  filtered_requirements "${CALVIN_ROOT}/calvin_env/requirements.txt" "${env_requirements}"
  filtered_requirements "${CALVIN_ROOT}/calvin_models/requirements.txt" "${model_requirements}"
  printf '%s\n' \
    'numpy==1.24.4' \
    'torch==1.13.1' \
    'torchvision==0.14.1' \
    'torchaudio==0.13.1' \
    'setuptools==57.5.0' > "${constraints}"
  # Keep the official dependency graph, including the legacy torch expected by
  # calvin_models, inside this venv only.
  pip_install "${python_bin}" "numpy==1.24.4"
  pip_install "${python_bin}" "torch==1.13.1" "torchvision==0.14.1" "torchaudio==0.13.1"
  # The evaluator imports ClearVLA's dependency-light benchmark boundary
  # (which audits HDF5 episodes) from this legacy interpreter as well.
  # Install h5py explicitly so that boundary does not reach into the
  # ManiSkill environment for a transitive dependency.
  pip_install "${python_bin}" "h5py==3.8.0"
  pip_install "${python_bin}" -c "${constraints}" -r "${env_requirements}"
  pip_install "${python_bin}" -c "${constraints}" -r "${model_requirements}"
  pip_install "${python_bin}" -c "${constraints}" "pytorch-lightning==1.8.6"
  pip_install "${python_bin}" "setuptools==57.5.0"
  if ! "${python_bin}" -c 'import pyhash' 2>/dev/null; then
    pip_install "${python_bin}" --no-build-isolation "pyhash==0.9.3"
  fi
  pip_install "${python_bin}" "numpy==1.24.4"
  rm -f "${env_requirements}" "${model_requirements}" "${constraints}"
  pip_install "${python_bin}" --no-deps -e "${CALVIN_ROOT}/calvin_env"
  pip_install "${python_bin}" --no-deps -e "${CALVIN_ROOT}/calvin_models"
  "${python_bin}" - <<'PY'
import importlib.metadata as md
import sys
import calvin_env
import calvin_agent
print({
    "python": sys.version.split()[0],
    "calvin_env": getattr(calvin_env, "__version__", "unknown"),
    "calvin_agent": getattr(calvin_agent, "__version__", "unknown"),
    "torch": md.version("torch"),
})
PY
}

install_libero() {
  make_env "${LIBERO_ENV}"
  local python_bin="${LIBERO_ENV}/bin/python"
  local libero_requirements constraints
  libero_requirements="$(mktemp /tmp/clearvla-libero-req.XXXXXX)"
  constraints="$(mktemp /tmp/clearvla-libero-constraints.XXXXXX)"
  filtered_requirements "${LIBERO_ROOT}/requirements.txt" "${libero_requirements}"
  printf '%s\n' 'numpy==1.24.4' 'setuptools==57.5.0' > "${constraints}"
  pip_install "${python_bin}" "numpy==1.24.4"
  # LIBERO's robosuite pin otherwise makes the resolver select the newest
  # MuJoCo release, which has no wheel for this legacy Python/Torch stack and
  # falls back to a source build requiring MUJOCO_PATH.  Install the known
  # wheel first and keep both packages out of the broad requirements solve.
  pip_install "${python_bin}" --no-deps "mujoco==2.3.7"
  pip_install "${python_bin}" --no-deps "robosuite==1.4.0"
  pip_install "${python_bin}" -c "${constraints}" -r "${libero_requirements}"
  # The repository's README pins a CUDA-11.3 torch that has no wheel for every
  # host Python. Try the official pin first; use a modern CUDA wheel only when
  # the pin is unavailable. The environment remains isolated either way.
  if ! pip_install "${python_bin}" \
      --index-url "https://download.pytorch.org/whl/cu113" \
      "torch==1.11.0+cu113" "torchvision==0.12.0+cu113" "torchaudio==0.11.0"; then
    echo "[libero-setup] official torch 1.11/cu113 wheel unavailable; using torch 2.2 CUDA fallback" >&2
    pip_install "${python_bin}" \
      --index-url "https://download.pytorch.org/whl/cu121" \
      "torch==2.2.2" "torchvision==0.17.2" "torchaudio==2.2.2"
  fi
  # The official HF dataset path is non-interactive and keeps its cache under
  # /data. This package is intentionally absent from the ClearVLA env.
  pip_install "${python_bin}" "huggingface_hub>=0.20,<1"
  # robosuite's off-screen MuJoCo binding imports PyOpenGL directly, but the
  # legacy LIBERO requirements omit it because the original conda setup pulled
  # it in indirectly. Keep the headless EGL dependency inside this venv.
  # ``mujoco`` is installed with ``--no-deps`` above to keep its version
  # pinned, so install both headless rendering dependencies explicitly.
  # Importing ``mujoco`` still imports its GLFW backend module even when EGL
  # is selected at runtime; omitting ``glfw`` therefore breaks LIBERO before
  # the evaluator can create an off-screen environment.
  pip_install "${python_bin}" "PyOpenGL==3.1.6" "glfw>=2.5,<3"
  pip_install "${python_bin}" "numba==0.57.1" "scipy==1.10.1"
  pip_install "${python_bin}" --no-deps -e "${LIBERO_ROOT}"
  pip_install "${python_bin}" "setuptools==57.5.0" "numpy==1.24.4"
  rm -f "${libero_requirements}" "${constraints}"
  # LIBERO reads this file at import time. BDDL/assets/init files stay with the
  # checkout; only demonstrations and cache material live under /data.
  printf '%s\n' \
    "benchmark_root: ${LIBERO_ROOT}/libero/libero" \
    "bddl_files: ${LIBERO_ROOT}/libero/libero/bddl_files" \
    "init_states: ${LIBERO_ROOT}/libero/libero/init_files" \
    "datasets: ${DATA_ROOT}/libero/raw/datasets" \
    "assets: ${LIBERO_ROOT}/libero/libero/assets" \
    > "${LIBERO_CONFIG_PATH}/config.yaml"
  LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}" \
    MUJOCO_GL="${MUJOCO_GL:-egl}" \
    PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" \
    PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH:-}" "${python_bin}" - <<'PY'
import importlib.metadata as md
import os
import sys
import mujoco
import numba
import robosuite
import libero
from libero.libero import get_libero_path
print({
    "python": sys.version.split()[0],
    "libero": md.version("libero"),
    "torch": md.version("torch"),
    "mujoco": md.version("mujoco"),
    "robosuite": md.version("robosuite"),
    "numba": md.version("numba"),
    "datasets": get_libero_path("datasets"),
    "config": os.environ["LIBERO_CONFIG_PATH"],
})
PY
}

if [[ "${DO_CALVIN}" == 1 ]]; then
  install_calvin
fi
if [[ "${DO_LIBERO}" == 1 ]]; then
  install_libero
fi

printf '[clearvla-benchmark-setup] calvin_env=%s libero_env=%s python=%s data=%s\n' \
  "${CALVIN_ENV}" "${LIBERO_ENV}" "${PYTHON_VERSION}" "${DATA_ROOT}"

