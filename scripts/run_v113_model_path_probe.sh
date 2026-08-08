#!/usr/bin/env bash
# Paired ownership/action probe for a completed V113 checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Refuse a mixed V113 checkout before loading a checkpoint.  The probe is
# intentionally stricter than ordinary inference: it requires the matched
# current-context intervention, the exact typed-posterior expansion repair,
# and the v13 result contract to be present together.
declare -a REQUIRED_SOURCE_MARKERS=(
  "${REPO_ROOT}/clearvla/policy/flow_dino_evidence.py|current_context_masked"
  "${REPO_ROOT}/clearvla/policy/trunk.py|uniform_route_weights"
  "${REPO_ROOT}/clearvla/experiments/observed_state_lab/policy_runtime_v39.py|clearvla-v113-model-path-intervention-v13"
)
source_has_marker() {
  local marker="$1"
  local source_file="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -q --fixed-strings "${marker}" "${source_file}"
  else
    grep -Fq -- "${marker}" "${source_file}"
  fi
}
for requirement in "${REQUIRED_SOURCE_MARKERS[@]}"; do
  source_file="${requirement%%|*}"
  source_marker="${requirement#*|}"
  if [[ ! -f "${source_file}" ]] || ! source_has_marker "${source_marker}" "${source_file}"; then
    echo "[v113-model-path-probe] source mismatch: missing ${source_marker} in ${source_file}" >&2
    echo "[v113-model-path-probe] sync the complete V113 source snapshot before rerunning" >&2
    exit 2
  fi
done

export CHECKPOINT="${CHECKPOINT:-runs/v113_functional_mainline_routing/checkpoints/latest.pt}"
export DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-runs/diagnostics/v113_model_path}"
export BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-113}"
export RESULT_JSON="${RESULT_JSON:-${DIAGNOSTICS_DIR}/probe_v113_${PROBE_BATCHES:-10}b.json}"
export MODEL_PATH_PROBE_LABEL=v113
export MODEL_PATH_REQUIRED_CONTRACT=v113

exec bash "${SCRIPT_DIR}/run_v103_model_path_probe.sh" "$@"
