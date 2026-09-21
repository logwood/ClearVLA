#!/usr/bin/env bash
# Configure CALVIN and LIBERO in local, isolated legacy environments.
#
# The benchmark source checkouts and Python environments stay on the host
# filesystem; demonstrations, decoded images, DINO/T5 artifacts and run
# outputs stay below CLEARVLA_BENCH_DATA_ROOT (default: /data/senwang/data).
# This wrapper supplies local defaults and delegates the pinned installation
# logic to setup_remote_benchmarks.sh, which is also used on senwang-server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${CLEARVLA_BENCH_DATA_ROOT:-/data/senwang/data}"
BENCH_ROOT="${CLEARVLA_BENCH_ROOT:-${REPO_ROOT}/third_party/benchmarks}"
CALVIN_ROOT="${CLEARVLA_CALVIN_ROOT:-${BENCH_ROOT}/calvin}"
LIBERO_ROOT="${CLEARVLA_LIBERO_ROOT:-${BENCH_ROOT}/LIBERO}"
CALVIN_ENV="${CLEARVLA_CALVIN_ENV:-${REPO_ROOT}/.venvs/clearvla-calvin}"
LIBERO_ENV="${CLEARVLA_LIBERO_ENV:-${REPO_ROOT}/.venvs/clearvla-libero}"

UV_DEFAULT="$(command -v uv 2>/dev/null || true)"
if [[ -z "${UV_DEFAULT}" ]]; then
  UV_DEFAULT="${REPO_ROOT}/.tools/uv/uv"
fi
UV_BIN="${CLEARVLA_UV_BIN:-${UV_DEFAULT}}"

if [[ ! -x "${UV_BIN}" ]]; then
  cat >&2 <<EOF
uv is required for the local benchmark environments.
Install uv or set CLEARVLA_UV_BIN to its executable path.
Current path: ${UV_BIN}
EOF
  exit 2
fi

for path in "${CALVIN_ROOT}" "${LIBERO_ROOT}"; do
  if [[ ! -d "${path}" ]]; then
    echo "benchmark checkout is missing: ${path}" >&2
    echo "Set CLEARVLA_BENCH_ROOT/CLEARVLA_CALVIN_ROOT/CLEARVLA_LIBERO_ROOT or clone the official checkouts there." >&2
    exit 2
  fi
done

exec env \
  CLEARVLA_BENCH_ROOT="${BENCH_ROOT}" \
  CLEARVLA_CALVIN_ROOT="${CALVIN_ROOT}" \
  CLEARVLA_LIBERO_ROOT="${LIBERO_ROOT}" \
  CLEARVLA_CALVIN_ENV="${CALVIN_ENV}" \
  CLEARVLA_LIBERO_ENV="${LIBERO_ENV}" \
  CLEARVLA_BENCH_DATA_ROOT="${DATA_ROOT}" \
  CLEARVLA_UV_BIN="${UV_BIN}" \
  bash "${SCRIPT_DIR}/setup_remote_benchmarks.sh" "$@"


