#!/usr/bin/env bash
# Idempotent isolated simulator setup for senwang-server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ENV="${CLEARVLA_SIM_ENV:-/data/senwang/envs/clearvla-sim}"
UV_INSTALL_DIR="${CLEARVLA_UV_INSTALL_DIR:-/data/senwang/tools/uv}"
UV_BIN="${UV_INSTALL_DIR}/uv"
UV_CACHE="${CLEARVLA_UV_CACHE:-/data/senwang/envs/uv-cache}"
UV_PYTHONS="${CLEARVLA_UV_PYTHONS:-/data/senwang/envs/uv-python}"

mkdir -p "${UV_INSTALL_DIR}" "${UV_CACHE}" "${UV_PYTHONS}" "$(dirname "${SIM_ENV}")"
if [[ ! -x "${UV_BIN}" ]]; then
  INSTALLER="$(mktemp /tmp/clearvla-uv-install.XXXXXX.sh)"
  trap 'rm -f "${INSTALLER}"' EXIT
  curl -LsSf --retry 3 https://astral.sh/uv/install.sh -o "${INSTALLER}"
  env UV_INSTALL_DIR="${UV_INSTALL_DIR}" UV_NO_MODIFY_PATH=1 sh "${INSTALLER}"
fi

export UV_CACHE_DIR="${UV_CACHE}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHONS}"
export UV_PROJECT_ENVIRONMENT="${SIM_ENV}"
cd "${REPO_ROOT}"

"${UV_BIN}" python install 3.12
"${UV_BIN}" sync \
  --frozen \
  --managed-python \
  --python 3.12 \
  --no-dev \
  --extra simulation \
  --extra simulation-maniskill \
  --extra reference-models

"${SIM_ENV}/bin/python" -c \
  'from importlib.metadata import version; import gymnasium, mani_skill, mujoco, torch, transformers; from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, T5EncoderModel; print({"python": __import__("sys").version.split()[0], "torch": torch.__version__, "mujoco": mujoco.__version__, "maniskill": version("mani-skill"), "gymnasium": gymnasium.__version__, "transformers": transformers.__version__})'

printf '[clearvla-sim-setup] env=%s uv=%s cache=%s\n' \
  "${SIM_ENV}" "${UV_BIN}" "${UV_CACHE}"


