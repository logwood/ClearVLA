#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_policy_golden_current.sh VARIANT OUTPUT_DIR [PYTHON]

Runs two independent captures of the current, possibly uncommitted source tree.
Use this only as a determinism and health self-check. Historical equivalence
still requires run_policy_golden_worktrees.sh with committed refs.
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage >&2
  exit 2
fi

VARIANT=$1
OUTPUT_DIR=$2
PYTHON_BIN=${3:-python}

case $VARIANT in
  v76|v77|v78|v79|v80|v81|v82|v84|v88) ;;
  *)
    echo "VARIANT must be v76, v77, v78, v79, v80, v81, v82, v84, or v88" >&2
    exit 2
    ;;
esac

REPO_ROOT=$(git rev-parse --show-toplevel)
HARNESS=$REPO_ROOT/clearvla/tools/policy_golden.py
SUPPORT_TREE=$REPO_ROOT/clearvla/data
OUTPUT_DIR=$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)

if [[ ! -d $SUPPORT_TREE ]]; then
  echo "required support tree is missing: $SUPPORT_TREE" >&2
  exit 1
fi

PYTHONHASHSEED=0 "$PYTHON_BIN" "$HARNESS" fixture \
  --output "$OUTPUT_DIR/fixture.pt"

for side in baseline candidate; do
  PYTHONHASHSEED=0 "$PYTHON_BIN" "$HARNESS" capture \
    --source-root "$REPO_ROOT" \
    --fixture "$OUTPUT_DIR/fixture.pt" \
    --variant "$VARIANT" \
    --support-tree "clearvla/data=$SUPPORT_TREE" \
    --output "$OUTPUT_DIR/$side"
done

PYTHONHASHSEED=0 "$PYTHON_BIN" "$HARNESS" compare \
  --baseline "$OUTPUT_DIR/baseline" \
  --candidate "$OUTPUT_DIR/candidate" \
  --report "$OUTPUT_DIR/comparison.json"

echo "current-source golden self-check passed: $OUTPUT_DIR/comparison.json"
