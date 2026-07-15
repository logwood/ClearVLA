#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
printf '%s\n' \
  'current_v81_distinct_block_nested_contraction.sh is a compatibility alias; current source uses the V82 sidecar ownership fix.' \
  'Checkout the historical V81 git version to reproduce V81 itself.' >&2
exec bash "${SCRIPT_DIR}/current_v82_sidecar_nested_contraction.sh" "$@"
