#!/usr/bin/env bash
# Never silently select an archived training graph from the repository root.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
  -h|--help)
    cat <<'HELP'
run_current_policy.sh no longer selects a default policy.

Current managed experiments:
  python scripts/clearvla_workspace.py --help

Historical V48 reproduction only (explicit opt-in):
  bash run_current_policy.sh --legacy-v48 [legacy arguments...]

Select a commit, outlet, config and fresh run ID before launching training.
HELP
    exit 0
    ;;
  --legacy-v48)
    shift
    exec bash "${SCRIPT_DIR}/scripts/current_v48_justok.sh" "$@"
    ;;
  *)
    echo "No default policy selected; use --help for managed experiments or --legacy-v48." >&2
    exit 2
    ;;
esac
