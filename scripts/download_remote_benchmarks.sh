#!/usr/bin/env bash
# Download benchmark archives directly into /data with resume and extraction markers.
set -euo pipefail

BENCH_ROOT="${CLEARVLA_BENCH_ROOT:-/home/sen.wang/workspace/robotics/benchmarks}"
CALVIN_ROOT="${CLEARVLA_CALVIN_ROOT:-${BENCH_ROOT}/calvin}"
LIBERO_ROOT="${CLEARVLA_LIBERO_ROOT:-${BENCH_ROOT}/LIBERO}"
DATA_ROOT="${CLEARVLA_BENCH_DATA_ROOT:-/data/senwang/data}"
CALVIN_DATA="${CLEARVLA_CALVIN_DATA:-${DATA_ROOT}/calvin}"
LIBERO_DATA="${CLEARVLA_LIBERO_DATA:-${DATA_ROOT}/libero}"
CALVIN_ENV="${CLEARVLA_CALVIN_ENV:-/home/sen.wang/.venvs/clearvla-calvin}"
LIBERO_ENV="${CLEARVLA_LIBERO_ENV:-/home/sen.wang/.venvs/clearvla-libero}"
LIBERO_CONFIG_PATH="${CLEARVLA_LIBERO_CONFIG_PATH:-${LIBERO_DATA}/config}"
HF_ENDPOINT_VALUE="${CLEARVLA_HF_ENDPOINT:-${HF_ENDPOINT:-}}"
HF_MAX_WORKERS="${CLEARVLA_HF_MAX_WORKERS:-4}"
DOWNLOAD_TOOL="${CLEARVLA_DOWNLOAD_TOOL:-auto}"
DOWNLOAD_CONNECTIONS="${CLEARVLA_DOWNLOAD_CONNECTIONS:-16}"
DOWNLOAD_SPLIT="${CLEARVLA_DOWNLOAD_SPLIT:-16}"

case "${DOWNLOAD_CONNECTIONS}" in
  ''|*[!0-9]*)
    echo "CLEARVLA_DOWNLOAD_CONNECTIONS must be a positive integer" >&2
    exit 2
    ;;
esac
case "${DOWNLOAD_SPLIT}" in
  ''|*[!0-9]*)
    echo "CLEARVLA_DOWNLOAD_SPLIT must be a positive integer" >&2
    exit 2
    ;;
esac
if (( DOWNLOAD_CONNECTIONS < 1 || DOWNLOAD_SPLIT < 1 )); then
  echo "CLEARVLA_DOWNLOAD_CONNECTIONS and CLEARVLA_DOWNLOAD_SPLIT must be positive" >&2
  exit 2
fi

mkdir -p "${CALVIN_DATA}/archives" "${CALVIN_DATA}/raw" \
  "${LIBERO_DATA}/raw/datasets" "${LIBERO_DATA}/cache" "${LIBERO_CONFIG_PATH}"

if [[ -n "${HF_ENDPOINT_VALUE}" ]]; then
  # huggingface_hub reads HF_ENDPOINT at import time. Keep the endpoint
  # explicit so restricted hosts can use an approved mirror without editing
  # the official LIBERO checkout; the default remains the upstream service.
  export HF_ENDPOINT="${HF_ENDPOINT_VALUE}"
fi

usage() {
  cat >&2 <<'EOF'
usage: download_remote_benchmarks.sh TARGET [...]

TARGET choices:
  calvin-debug  1.3 GB CALVIN debug archive
  calvin-d      166 GB CALVIN D->D archive
  calvin-abc    517 GB CALVIN ABC->D archive
  calvin-abcd   656 GB CALVIN ABCD->D archive
  libero-spatial | libero-object | libero-goal | libero-100
  libero-90 | libero-10
  libero-all    all four official LIBERO archives
EOF
}

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

require_space_gb() {
  local required_gb="$1"
  local available_kb
  available_kb="$(df -Pk "${DATA_ROOT}" | awk 'NR==2 {print $4}')"
  if [[ -z "${available_kb}" ]] || (( available_kb < required_gb * 1024 * 1024 )); then
    echo "insufficient /data space: need approximately ${required_gb} GiB" >&2
    df -h "${DATA_ROOT}" >&2 || true
    exit 3
  fi
}

verify_calvin_checksum() {
  local archive="$1"
  local sums="${CALVIN_DATA}/archives/sha256sum.txt"
  if [[ ! -s "${sums}" ]]; then
    if ! wget -q --tries=3 --timeout=60 \
      "http://calvin.cs.uni-freiburg.de/dataset/sha256sum.txt" -O "${sums}"; then
      echo "[calvin-download] checksum list unavailable; archive is retained for later verification" >&2
      return 0
    fi
  fi
  local line
  line="$(grep -E "[[:space:]]\*?$(basename "${archive}")$" "${sums}" | head -n 1 || true)"
  if [[ -z "${line}" ]]; then
    echo "[calvin-download] no checksum entry for $(basename "${archive}")" >&2
    return 0
  fi
  if ! printf '%s\n' "${line}" | (cd "$(dirname "${archive}")" && sha256sum -c -); then
    echo "CALVIN checksum failed for ${archive}" >&2
    exit 4
  fi
}

download_calvin_archive() {
  local url="$1"
  local archive="$2"
  local tool="${DOWNLOAD_TOOL}"
  case "${tool}" in
    auto|aria2|aria2c)
      if [[ "${tool}" == auto && ! -x "$(command -v aria2c 2>/dev/null || true)" ]]; then
        tool="wget"
      elif command -v aria2c >/dev/null 2>&1; then
        # aria2 understands both an ordinary partial file left by wget and its
        # own .aria2 control file.  --file-allocation=none avoids reserving the
        # full 517--656 GB archive before the first byte is received.
        aria2c \
          --continue=true \
          --allow-overwrite=false \
          --auto-file-renaming=false \
          --max-connection-per-server="${DOWNLOAD_CONNECTIONS}" \
          --split="${DOWNLOAD_SPLIT}" \
          --min-split-size=16M \
          --max-tries=20 \
          --retry-wait=10 \
          --timeout=120 \
          --connect-timeout=30 \
          --file-allocation=none \
          --dir="$(dirname "${archive}")" \
          --out="$(basename "${archive}")" \
          "${url}"
        return
      else
        if [[ "${tool}" != auto ]]; then
          echo "CLEARVLA_DOWNLOAD_TOOL=${DOWNLOAD_TOOL} requested aria2c, but aria2c is missing" >&2
          exit 2
        fi
        tool="wget"
      fi
      ;;
    wget)
      ;;
    *)
      printf 'CLEARVLA_DOWNLOAD_TOOL must be auto, aria2c, or wget; got %s\n' \
        "${DOWNLOAD_TOOL}" >&2
      exit 2
      ;;
  esac
  wget --continue --tries=20 --timeout=120 --waitretry=10 "${url}" -O "${archive}"
}

download_calvin() {
  local kind="$1"
  local folder url archive marker staging extracted target
  case "${kind}" in
    debug)
      folder="calvin_debug_dataset"
      url="http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip"
      ;;
    d)
      folder="task_D_D"
      url="http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip"
      ;;
    abc)
      folder="task_ABC_D"
      url="http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip"
      ;;
    abcd)
      folder="task_ABCD_D"
      url="http://calvin.cs.uni-freiburg.de/dataset/task_ABCD_D.zip"
      ;;
    *) echo "unknown CALVIN archive: ${kind}" >&2; exit 2 ;;
  esac
  archive="${CALVIN_DATA}/archives/${folder}.zip"
  marker="${CALVIN_DATA}/raw/.${folder}.complete"
  target="${CALVIN_DATA}/raw/${folder}"
  if [[ -f "${marker}" ]]; then
    echo "[calvin-download] reuse ${target}"
    return 0
  fi
  if [[ -e "${target}" ]]; then
    echo "partial CALVIN extraction exists without marker: ${target}" >&2
    echo "inspect it and remove only that directory before retrying" >&2
    exit 5
  fi
  case "${kind}" in
    debug) require_space_gb 5 ;;
    d) require_space_gb 220 ;;
    abc) require_space_gb 650 ;;
    abcd) require_space_gb 820 ;;
  esac
  echo "[calvin-download] fetching ${url} -> ${archive} (tool=${DOWNLOAD_TOOL}, connections=${DOWNLOAD_CONNECTIONS}, split=${DOWNLOAD_SPLIT})"
  download_calvin_archive "${url}" "${archive}"
  verify_calvin_checksum "${archive}"
  staging="${CALVIN_DATA}/raw/.extract_${folder}"
  if [[ -e "${staging}" ]]; then
    echo "partial CALVIN extraction staging exists: ${staging}" >&2
    exit 5
  fi
  mkdir -p "${staging}"
  unzip -q "${archive}" -d "${staging}"
  extracted="$(find "${staging}" -mindepth 1 -maxdepth 3 -type d -name training -print -quit || true)"
  if [[ -n "${extracted}" ]]; then
    extracted="$(dirname "${extracted}")"
  elif [[ -d "${staging}/training" && -d "${staging}/validation" ]]; then
    extracted="${staging}"
  else
    echo "cannot locate CALVIN training/validation folders in ${staging}" >&2
    exit 6
  fi
  mv "${extracted}" "${target}"
  if [[ -d "${staging}" ]] && [[ -z "$(find "${staging}" -mindepth 1 -print -quit)" ]]; then
    rmdir "${staging}"
  fi
  printf 'archive=%s\nurl=%s\n' "${archive}" "${url}" > "${marker}"
  echo "[calvin-download] extracted ${target}"
}

download_libero() {
  local dataset="$1"
  local marker="${LIBERO_DATA}/raw/.${dataset}.complete"
  local target="${LIBERO_DATA}/raw/datasets/${dataset}"
  if [[ -f "${marker}" ]]; then
    echo "[libero-download] reuse ${target}"
    return 0
  fi
  if [[ ! -x "${LIBERO_ENV}/bin/python" ]]; then
    echo "missing LIBERO environment ${LIBERO_ENV}; run setup_remote_benchmarks.sh libero" >&2
    exit 2
  fi
  export LIBERO_CONFIG_PATH
  export HF_HOME="${LIBERO_DATA}/cache/huggingface"
  export HUGGINGFACE_HUB_CACHE="${LIBERO_DATA}/cache/huggingface/hub"
  mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}"
  require_space_gb 20
  printf '[libero-download] fetching %s into %s\n' "${dataset}" "${LIBERO_DATA}/raw/datasets"
  # Use snapshot_download directly instead of the upstream helper's
  # force_download=True call.  This preserves completed files and resumes a
  # truncated .incomplete object after a network interruption.  The official
  # checkout remains the source of the dataset id and task metadata.
  (cd "${LIBERO_ROOT}" && PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH:-}" \
    "${LIBERO_ENV}/bin/python" - "${dataset}" "${LIBERO_DATA}/raw/datasets" \
    "${HF_MAX_WORKERS}" <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

dataset, destination, max_workers = sys.argv[1:]
patterns = [f"{dataset}/*"]
if dataset == "libero_100":
    # The maintained HF mirror stores LIBERO-100 as its official 90+10
    # directories rather than a synthetic libero_100 directory.
    patterns = ["libero_90/*", "libero_10/*"]
snapshot_download(
    repo_id="yifengzhu-hf/LIBERO-datasets",
    repo_type="dataset",
    local_dir=destination,
    allow_patterns=patterns,
    force_download=False,
    max_workers=int(max_workers),
)
PY
  )
  if [[ ! -d "${target}" ]]; then
    if [[ "${dataset}" != "libero_100" ]]; then
      echo "LIBERO downloader completed but expected directory is absent: ${target}" >&2
      echo "inspect ${LIBERO_DATA}/raw/datasets before retrying" >&2
      exit 6
    fi
  fi
  local expected_count actual_count
  case "${dataset}" in
    libero_100) expected_count=100 ;;
    libero_90) expected_count=90 ;;
    *) expected_count=10 ;;
  esac
  if [[ "${dataset}" == "libero_100" ]]; then
    actual_count="$(( $(find "${LIBERO_DATA}/raw/datasets/libero_90" -maxdepth 1 -type f -name '*.hdf5' 2>/dev/null | wc -l) + $(find "${LIBERO_DATA}/raw/datasets/libero_10" -maxdepth 1 -type f -name '*.hdf5' 2>/dev/null | wc -l) ))"
  else
    actual_count="$(find "${target}" -maxdepth 1 -type f -name '*.hdf5' 2>/dev/null | wc -l)"
  fi
  if (( actual_count < expected_count )); then
    echo "LIBERO ${dataset} is incomplete: ${actual_count}/${expected_count} HDF5 files" >&2
    echo "inspect ${LIBERO_DATA}/raw/datasets before retrying" >&2
    exit 6
  fi
  printf 'dataset=%s\n' "${dataset}" > "${marker}"
  echo "[libero-download] extracted ${target}"
}

for target in "$@"; do
  case "${target}" in
    calvin-debug) download_calvin debug ;;
    calvin-d) download_calvin d ;;
    calvin-abc) download_calvin abc ;;
    calvin-abcd) download_calvin abcd ;;
    libero-spatial) download_libero libero_spatial ;;
    libero-object) download_libero libero_object ;;
    libero-goal) download_libero libero_goal ;;
    libero-100) download_libero libero_100 ;;
    libero-90) download_libero libero_90 ;;
    libero-10) download_libero libero_10 ;;
    libero-all)
      download_libero libero_spatial
      download_libero libero_object
      download_libero libero_goal
      download_libero libero_100
      ;;
    *) usage; exit 2 ;;
  esac
done

printf '[clearvla-benchmark-download] calvin=%s libero=%s data=%s\n' \
  "${CALVIN_DATA}/raw" "${LIBERO_DATA}/raw/datasets" "${DATA_ROOT}"


