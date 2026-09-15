#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_policy_golden_sources.sh \
    BASELINE_SOURCE_ROOT CANDIDATE_SOURCE_ROOT VARIANT OUTPUT_DIR [PYTHON]

Compares two existing source trees, including uncommitted worktrees, with one
fixture and exact tensor equality. Use this before promoting a refactor whose
baseline or candidate has not yet been committed.
EOF
}

if [[ $# -lt 4 || $# -gt 5 ]]; then
  usage >&2
  exit 2
fi

BASELINE_ROOT=$1
CANDIDATE_ROOT=$2
VARIANT=$3
OUTPUT_DIR=$4
PYTHON_BIN=${5:-python}

if [[ ! -d $BASELINE_ROOT || ! -d $CANDIDATE_ROOT ]]; then
  echo "baseline and candidate source roots must both be directories" >&2
  exit 1
fi

BASELINE_ROOT=$(cd "$BASELINE_ROOT" && pwd)
CANDIDATE_ROOT=$(cd "$CANDIDATE_ROOT" && pwd)
HARNESS=$CANDIDATE_ROOT/clearvla/tools/policy_golden.py
if [[ ! -f $HARNESS ]]; then
  echo "candidate golden harness is missing: $HARNESS" >&2
  exit 1
fi

SUPPORT_TREE=
for source_root in "$BASELINE_ROOT" "$CANDIDATE_ROOT"; do
  if [[ -d $source_root/clearvla/data ]]; then
    SUPPORT_TREE=$source_root/clearvla/data
    break
  fi
done
if [[ -z $SUPPORT_TREE ]]; then
  echo "required clearvla/data support tree is missing from both sources" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
for artifact in fixture.pt baseline candidate comparison.json; do
  if [[ -e $OUTPUT_DIR/$artifact ]]; then
    echo "output artifact already exists; use a fresh output directory: $OUTPUT_DIR/$artifact" >&2
    exit 1
  fi
done

PYTHONHASHSEED=0 "$PYTHON_BIN" "$HARNESS" fixture \
  --output "$OUTPUT_DIR/fixture.pt"

PYTHONHASHSEED=0 "$PYTHON_BIN" "$HARNESS" capture \
  --source-root "$BASELINE_ROOT" \
  --fixture "$OUTPUT_DIR/fixture.pt" \
  --variant "$VARIANT" \
  --support-tree "clearvla/data=$SUPPORT_TREE" \
  --output "$OUTPUT_DIR/baseline"

PYTHONHASHSEED=0 "$PYTHON_BIN" "$HARNESS" capture \
  --source-root "$CANDIDATE_ROOT" \
  --fixture "$OUTPUT_DIR/fixture.pt" \
  --variant "$VARIANT" \
  --support-tree "clearvla/data=$SUPPORT_TREE" \
  --output "$OUTPUT_DIR/candidate"

PYTHONHASHSEED=0 "$PYTHON_BIN" "$HARNESS" compare \
  --baseline "$OUTPUT_DIR/baseline" \
  --candidate "$OUTPUT_DIR/candidate" \
  --report "$OUTPUT_DIR/comparison.json"

echo "cross-source golden comparison passed: $OUTPUT_DIR/comparison.json"
